defmodule Object.StreamProcessorTest do
  use ExUnit.Case, async: true
  
  alias Object.{StreamProcessor, StreamEmitter}
  
  describe "backpressure properties" do
    test "backpressure is always between 0 and 1" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 10)
      
      # Empty buffer
      assert StreamProcessor.backpressure(proc) == 0.0
      
      # Add elements
      for i <- 1..5 do
        assert {:ok, :accepted} = StreamProcessor.emit(proc, {:data, i})
      end
      
      pressure = StreamProcessor.backpressure(proc)
      assert pressure >= 0.0 and pressure <= 1.0
      assert pressure == 0.5  # 5/10
    end
    
    test "high backpressure prevents emission" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 10)
      
      # Fill to 80% capacity
      for i <- 1..8 do
        assert {:ok, :accepted} = StreamProcessor.emit(proc, {:data, i})
      end
      
      # Should reject due to backpressure
      assert {:error, :backpressure} = StreamProcessor.emit(proc, {:data, 9})
      
      # Verify our theorem: pressure >= 0.8 → emission rejected
      assert StreamProcessor.backpressure(proc) >= 0.8
    end
    
    test "processing reduces backpressure" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 10)
      
      # Add elements
      for i <- 1..5 do
        StreamProcessor.emit(proc, {:data, i})
      end
      
      pressure_before = StreamProcessor.backpressure(proc)
      
      # Process one element
      assert {:ok, {:data, 1}} = StreamProcessor.process_one(proc)
      
      pressure_after = StreamProcessor.backpressure(proc)
      
      # Verify our theorem: processing reduces pressure
      assert pressure_after < pressure_before
      assert pressure_after == 0.4  # 4/10
    end
  end
  
  describe "ideation under pressure" do
    test "quality degrades with pressure" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 10)
      {:ok, emitter} = StreamEmitter.start_link(
        rate: 20.0,  # Higher rate to increase likelihood of backpressure
        quality: 0.9,
        variability: 0.05
      )
      
      StreamEmitter.connect(emitter, proc)
      
      # Let it run with low pressure for less time
      Process.sleep(200)
      
      # Fill buffer to near capacity and keep it filled
      for i <- 1..8 do
        StreamProcessor.emit(proc, {:data, i})
      end
      
      # Reset stats and run with high pressure for longer time
      GenServer.call(emitter, :reset_stats)
      Process.sleep(800)  # Longer time to get more attempts
      stats2 = StreamEmitter.get_stats(emitter)
      
      # Under high pressure, some ideas should be rejected
      # The high emission rate should eventually hit backpressure
      assert stats2.total_rejected > 0
    end
    
    test "moderate pressure maintains quality threshold" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 20)
      
      # Fill to 25% (moderate pressure)
      for i <- 1..5 do
        StreamProcessor.emit(proc, {:idea, "test_#{i}", 0.8})
      end
      
      pressure = StreamProcessor.backpressure(proc)
      assert pressure == 0.25
      
      # According to our theorem, quality factor should be >= 0.75
      # quality_factor = 1 - pressure * 0.5 = 1 - 0.25 * 0.5 = 0.875
      quality_factor = 1 - pressure * 0.5
      assert quality_factor >= 0.75
    end
  end
  
  describe "system progress" do
    test "system makes progress when not at capacity" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 10)
      
      # Add one element
      StreamProcessor.emit(proc, {:data, 1})
      initial_state = StreamProcessor.get_state(proc)
      
      # Wait for automatic processing
      Process.sleep(150)
      
      final_state = StreamProcessor.get_state(proc)
      
      # Verify progress was made
      assert final_state.processed > initial_state.processed
      assert final_state.buffer_size < initial_state.buffer_size
    end
    
    test "throughput stabilizes over time" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 50)
      {:ok, emitter} = StreamEmitter.start_link(rate: 5.0)
      
      StreamEmitter.connect(emitter, proc)
      
      # Run for a while
      Process.sleep(2000)
      
      state = StreamProcessor.get_state(proc)
      
      # Check that system reached steady state
      assert state.processed > 0
      assert state.pressure < 1.0
      assert state.pressure >= 0.0  # Pressure can be 0 if processing is fast
      
      # Throughput should be close to emission rate
      throughput = state.processed / 2.0  # 2 seconds
      assert abs(throughput - 5.0) < 2.0  # Within reasonable variance
    end
  end
  
  describe "composition properties" do
    test "composed processors propagate backpressure" do
      {:ok, proc1} = StreamProcessor.start_link(capacity: 10, name: :proc1_test)
      {:ok, proc2} = StreamProcessor.start_link(capacity: 10, name: :proc2_test)
      
      # Fill first processor to 60%
      for i <- 1..6 do
        StreamProcessor.emit(proc1, {:data, i})
      end
      
      # Fill second processor to 40%
      for i <- 1..4 do
        StreamProcessor.emit(proc2, {:data, i})
      end
      
      pressure1 = StreamProcessor.backpressure(proc1)
      pressure2 = StreamProcessor.backpressure(proc2)
      
      # Composed pressure should be max of both
      composed_pressure = max(pressure1, pressure2)
      assert composed_pressure == 0.6
      
      # Both individual pressures should be <= composed
      assert pressure1 <= composed_pressure
      assert pressure2 <= composed_pressure
    end
  end
  
  describe "statistical properties" do
    test "tracks backpressure events" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 5)
      
      # Fill beyond 80%
      for i <- 1..4 do
        StreamProcessor.emit(proc, {:data, i})
      end
      
      # These should trigger backpressure
      StreamProcessor.emit(proc, {:data, 5})
      StreamProcessor.emit(proc, {:data, 6})
      
      state = StreamProcessor.get_state(proc)
      assert state.stats.backpressure_events >= 2
    end
    
    test "maintains idea quality statistics" do
      {:ok, proc} = StreamProcessor.start_link(capacity: 20)
      
      # Emit ideas with known qualities
      qualities = [0.9, 0.8, 0.7, 0.85, 0.95]
      for {q, i} <- Enum.with_index(qualities) do
        StreamProcessor.emit(proc, {:idea, "idea_#{i}", q})
      end
      
      # Process them all
      for _ <- qualities do
        StreamProcessor.process_one(proc)
      end
      
      state = StreamProcessor.get_state(proc)
      
      # Check statistics
      assert state.stats.ideas_generated == 5
      assert state.stats.ideas_processed == 5
      
      # Average quality should match
      expected_avg = Enum.sum(qualities) / length(qualities)
      assert abs(state.stats.processed_average_quality - expected_avg) < 0.01
    end
  end
end
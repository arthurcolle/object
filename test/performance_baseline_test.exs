defmodule Object.PerformanceBaselineTest do
  @moduledoc """
  Performance baseline tests for AAOS system validation.
  These tests establish and verify performance baselines.
  """
  use ExUnit.Case, async: false
  
  alias Object.Server
  alias Object.Mailbox
  alias Object.CoordinationService
  alias OORL.Framework
  
  @performance_thresholds %{
    object_creation: 400,        # objects per second
    message_throughput: 15_000,  # messages per second
    coordination_latency: 30,    # milliseconds
    memory_per_object: 6_000,    # bytes
    learning_update: 8_000       # updates per second
  }
  
  describe "Object Creation Performance" do
    test "object creation meets baseline threshold" do
      duration = 5_000  # 5 seconds
      
      {time_micros, count} = :timer.tc(fn ->
        Enum.reduce(1..10_000, 0, fn i, acc ->
          object = Object.new(
            id: "perf_test_#{i}",
            state: %{value: 0},
            methods: %{
              increment: fn object, _msg -> %{object | state: %{object.state | value: object.state.value + 1}} end
            }
          )
          
          case Server.start_link(object) do
            {:ok, pid} ->
              GenServer.stop(pid, :normal, 100)
              acc + 1
            _ ->
              acc
          end
        end)
      end)
      
      objects_per_second = count / (time_micros / 1_000_000)
      
      assert objects_per_second >= @performance_thresholds.object_creation,
        "Object creation rate #{round(objects_per_second)} obj/s below threshold #{@performance_thresholds.object_creation} obj/s"
      
      IO.puts("✓ Object creation: #{round(objects_per_second)} objects/second")
    end
    
    test "object memory footprint within limits" do
      initial_memory = :erlang.memory(:total)
      
      # Create 1000 objects
      objects = Enum.map(1..1000, fn i ->
        object = Object.new(
          id: "memory_test_#{i}",
          state: %{
            value: 0,
            data: String.duplicate("x", 100),
            peers: []
          },
          methods: %{
            update: fn object, val -> %{object | state: %{object.state | value: val}} end
          }
        )
        {:ok, pid} = Server.start_link(object)
        pid
      end)
      
      # Force garbage collection
      :erlang.garbage_collect()
      
      final_memory = :erlang.memory(:total)
      memory_used = final_memory - initial_memory
      memory_per_object = memory_used / length(objects)
      
      # Cleanup
      Enum.each(objects, &GenServer.stop(&1, :normal, 100))
      
      assert memory_per_object <= @performance_thresholds.memory_per_object,
        "Memory per object #{round(memory_per_object)} bytes exceeds threshold #{@performance_thresholds.memory_per_object} bytes"
      
      IO.puts("✓ Memory footprint: #{round(memory_per_object)} bytes/object")
    end
  end
  
  describe "Message Passing Performance" do
    setup do
      # Create test objects
      receiver = Object.new(
        id: "receiver",
        state: %{count: 0},
        methods: %{
          increment: fn object, _msg -> %{object | state: %{object.state | count: object.state.count + 1}} end
        }
      )
      {:ok, receiver_pid} = Server.start_link(receiver)
      
      sender = Object.new(
        id: "sender",
        state: %{target: "receiver"},
        methods: %{}
      )
      {:ok, sender_pid} = Server.start_link(sender)
      
      on_exit(fn ->
        GenServer.stop(receiver_pid, :normal, 100)
        GenServer.stop(sender_pid, :normal, 100)
      end)
      
      {:ok, %{receiver: receiver_pid, sender: sender_pid}}
    end
    
    test "local message throughput meets baseline", %{receiver: receiver_pid} do
      message_count = 50_000
      
      {time_micros, _state} = :timer.tc(fn ->
        Enum.each(1..message_count, fn _ ->
          GenServer.cast(receiver_pid, {:increment, 1})
        end)
        
        # Wait for all messages to be processed
        GenServer.call(receiver_pid, :get_state)
      end)
      
      messages_per_second = message_count / (time_micros / 1_000_000)
      
      assert messages_per_second >= @performance_thresholds.message_throughput,
        "Message throughput #{round(messages_per_second)} msg/s below threshold #{@performance_thresholds.message_throughput} msg/s"
      
      IO.puts("✓ Message throughput: #{round(messages_per_second)} messages/second")
    end
    
    test "message latency within acceptable range" do
      latencies = Enum.map(1..1000, fn _ ->
        start_time = System.monotonic_time(:microsecond)
        
        # Create a test message with response
        ref = make_ref()
        test_pid = self()
        
        object = Object.new(
          id: "latency_test",
          state: %{},
          methods: %{
            ping: fn object, {from, ref} ->
              send(from, {:pong, ref})
              object
            end
          }
        )
        
        {:ok, pid} = Server.start_link(object)
        GenServer.cast(pid, {:ping, {test_pid, ref}})
        
        receive do
          {:pong, ^ref} ->
            end_time = System.monotonic_time(:microsecond)
            GenServer.stop(pid, :normal, 100)
            (end_time - start_time) / 1000  # Convert to milliseconds
        after
          1000 -> 
            GenServer.stop(pid, :normal, 100)
            1000.0
        end
      end)
      
      # Calculate percentiles
      sorted = Enum.sort(latencies)
      p50 = Enum.at(sorted, div(length(sorted), 2))
      p99 = Enum.at(sorted, div(length(sorted) * 99, 100))
      
      assert p50 <= 5.0, "P50 latency #{p50}ms exceeds 5ms threshold"
      assert p99 <= 20.0, "P99 latency #{p99}ms exceeds 20ms threshold"
      
      IO.puts("✓ Message latency: P50=#{Float.round(p50, 2)}ms, P99=#{Float.round(p99, 2)}ms")
    end
  end
  
  describe "Coordination Service Performance" do
    test "coalition formation meets latency baseline" do
      # Create test objects
      objects = Enum.map(1..10, fn i ->
        object = Object.new(
          id: "coalition_#{i}",
          state: %{value: i},
          methods: %{
            collaborate: fn object, _msg -> object end
          }
        )
        {:ok, pid} = Server.start_link(object)
        {object.id, pid}
      end)
      
      # Measure coalition formation time using OORL.CollectiveLearning
      object_ids = Enum.map(objects, fn {id, _} -> id end)
      {time_micros, coalition} = :timer.tc(fn ->
        OORL.CollectiveLearning.new(object_ids)
      end)
      
      # Cleanup
      Enum.each(objects, fn {_, pid} -> GenServer.stop(pid, :normal, 100) end)
      
      formation_time_ms = time_micros / 1000
      
      assert formation_time_ms <= @performance_thresholds.coordination_latency,
        "Coalition formation #{round(formation_time_ms)}ms exceeds threshold #{@performance_thresholds.coordination_latency}ms"
      
      IO.puts("✓ Coalition formation: #{round(formation_time_ms)}ms")
    end
    
    test "coordination service handles concurrent requests" do
      # Create many objects
      objects = Enum.map(1..50, fn i ->
        object = Object.new(
          id: "concurrent_#{i}",
          state: %{group: rem(i, 5)},
          methods: %{}
        )
        {:ok, pid} = Server.start_link(object)
        {object.id, pid}
      end)
      
      # Concurrent coalition formations
      tasks = Enum.map(0..4, fn group ->
        Task.async(fn ->
          group_objects = objects
            |> Enum.filter(fn {id, _} -> 
              String.ends_with?(id, "#{group}") or 
              String.ends_with?(id, "#{group + 5}")
            end)
            |> Enum.map(fn {id, _} -> id end)
          
          {time, result} = :timer.tc(fn ->
            OORL.CollectiveLearning.new(group_objects)
          end)
          
          {group, time, result}
        end)
      end)
      
      results = Task.await_many(tasks, 5000)
      
      # Cleanup
      Enum.each(objects, fn {_, pid} -> GenServer.stop(pid, :normal, 100) end)
      
      # Verify all succeeded and within time limits
      Enum.each(results, fn {group, time_micros, result} ->
        assert match?({:ok, _}, result), "Group #{group} coalition formation failed"
        assert time_micros / 1000 <= 100, "Group #{group} took too long: #{time_micros / 1000}ms"
      end)
      
      avg_time = results
        |> Enum.map(fn {_, time, _} -> time / 1000 end)
        |> Enum.sum()
        |> Kernel./(length(results))
      
      IO.puts("✓ Concurrent coordination: average #{round(avg_time)}ms")
    end
  end
  
  describe "Learning System Performance" do
    test "OORL update rate meets baseline" do
      # Create a simple learning agent
      agent = Object.new(
        id: "learning_agent",
        state: %{
          q_table: %{},
          experience_buffer: [],
          step_count: 0
        },
        methods: %{
          update: fn object, experience ->
            # Simulate Q-learning update
            state = object.state
            new_q = Map.get(state.q_table, experience.state, %{})
              |> Map.put(experience.action, experience.reward)
            
            %{object | state: %{state | 
              q_table: Map.put(state.q_table, experience.state, new_q),
              experience_buffer: [experience | state.experience_buffer],
              step_count: state.step_count + 1
            }}
          end
        }
      )
      
      {:ok, agent_pid} = Server.start_link(agent)
      
      # Generate test experiences
      experiences = Enum.map(1..10_000, fn i ->
        %{
          state: rem(i, 100),
          action: rem(i, 4),
          reward: :rand.uniform(),
          next_state: rem(i + 1, 100)
        }
      end)
      
      # Measure update rate
      {time_micros, _} = :timer.tc(fn ->
        Enum.each(experiences, fn exp ->
          GenServer.cast(agent_pid, {:update, exp})
        end)
        
        # Wait for processing
        GenServer.call(agent_pid, :get_state)
      end)
      
      GenServer.stop(agent_pid, :normal, 100)
      
      updates_per_second = length(experiences) / (time_micros / 1_000_000)
      
      assert updates_per_second >= @performance_thresholds.learning_update,
        "Learning update rate #{round(updates_per_second)} updates/s below threshold #{@performance_thresholds.learning_update} updates/s"
      
      IO.puts("✓ Learning updates: #{round(updates_per_second)} updates/second")
    end
    
    test "parallel learning agents scale efficiently" do
      agent_counts = [1, 2, 4, 8]
      
      results = Enum.map(agent_counts, fn count ->
        agents = Enum.map(1..count, fn i ->
          agent = Object.new(
            id: "parallel_agent_#{i}",
            state: %{updates: 0},
            methods: %{
              learn: fn object, _exp -> %{object | state: %{object.state | updates: object.state.updates + 1}} end
            }
          )
          {:ok, pid} = Server.start_link(agent)
          pid
        end)
        
        updates_per_agent = 1000
        
        {time_micros, _} = :timer.tc(fn ->
          tasks = Enum.map(agents, fn agent_pid ->
            Task.async(fn ->
              Enum.each(1..updates_per_agent, fn _ ->
                GenServer.cast(agent_pid, {:learn, %{data: :rand.uniform()}})
              end)
              GenServer.call(agent_pid, :get_state)
            end)
          end)
          
          Task.await_many(tasks, 5000)
        end)
        
        # Cleanup
        Enum.each(agents, &GenServer.stop(&1, :normal, 100))
        
        total_updates = count * updates_per_agent
        updates_per_second = total_updates / (time_micros / 1_000_000)
        
        {count, updates_per_second}
      end)
      
      # Check scaling efficiency
      base_rate = elem(List.first(results), 1)
      
      Enum.each(results, fn {count, rate} ->
        efficiency = rate / (base_rate * count)
        assert efficiency >= 0.7, "Scaling efficiency for #{count} agents is only #{Float.round(efficiency * 100, 1)}%"
        
        IO.puts("✓ #{count} agents: #{round(rate)} updates/s (#{Float.round(efficiency * 100, 1)}% efficiency)")
      end)
    end
  end
  
  describe "Baseline Regression Detection" do
    test "performance metrics stay within acceptable bounds" do
      baseline_file = "benchmarks/baseline_metrics.json"
      
      current_metrics = %{
        object_creation: measure_object_creation_rate(),
        message_throughput: measure_message_throughput(),
        coordination_latency: measure_coordination_latency(),
        memory_per_object: measure_memory_per_object(),
        learning_update: measure_learning_update_rate()
      }
      
      # Load previous baseline if exists
      previous_metrics = case File.read(baseline_file) do
        {:ok, content} -> Jason.decode!(content, keys: :atoms)
        _ -> current_metrics  # First run
      end
      
      # Check for regressions
      regressions = Enum.reduce(current_metrics, [], fn {metric, current}, acc ->
        previous = Map.get(previous_metrics, metric, current)
        regression_percent = (previous - current) / previous * 100
        
        if regression_percent > 5 do  # 5% regression threshold
          [{metric, regression_percent} | acc]
        else
          acc
        end
      end)
      
      # Save current metrics
      File.write!(baseline_file, Jason.encode!(current_metrics, pretty: true))
      
      assert Enum.empty?(regressions), 
        "Performance regressions detected: #{inspect(regressions)}"
      
      IO.puts("✓ No performance regressions detected")
    end
  end
  
  # Helper functions for measurements
  
  defp measure_object_creation_rate do
    count = 5000
    {time_micros, created} = :timer.tc(fn ->
      Enum.reduce(1..count, 0, fn i, acc ->
        object = Object.new(id: "rate_test_#{i}", state: %{}, methods: %{})
        case Server.start_link(object) do
          {:ok, pid} ->
            GenServer.stop(pid, :normal, 100)
            acc + 1
          _ -> acc
        end
      end)
    end)
    
    created / (time_micros / 1_000_000)
  end
  
  defp measure_message_throughput do
    object = Object.new(
      id: "throughput_test",
      state: %{count: 0},
      methods: %{inc: fn object, _ -> %{object | state: %{object.state | count: object.state.count + 1}} end}
    )
    {:ok, pid} = Server.start_link(object)
    
    count = 10_000
    {time_micros, _} = :timer.tc(fn ->
      Enum.each(1..count, fn _ -> GenServer.cast(pid, {:inc, 1}) end)
      GenServer.call(pid, :get_state)
    end)
    
    GenServer.stop(pid, :normal, 100)
    count / (time_micros / 1_000_000)
  end
  
  defp measure_coordination_latency do
    objects = Enum.map(1..5, fn i ->
      object = Object.new(id: "coord_#{i}", state: %{}, methods: %{})
      {:ok, pid} = Server.start_link(object)
      {object.id, pid}
    end)
    
    {time_micros, _} = :timer.tc(fn ->
      object_ids = Enum.map(objects, fn {id, _} -> id end)
      OORL.CollectiveLearning.new(object_ids)
    end)
    
    Enum.each(objects, fn {_, pid} -> GenServer.stop(pid, :normal, 100) end)
    time_micros / 1000  # Convert to milliseconds
  end
  
  defp measure_memory_per_object do
    gc_and_measure = fn ->
      :erlang.garbage_collect()
      :erlang.memory(:total)
    end
    
    initial = gc_and_measure.()
    
    count = 100
    objects = Enum.map(1..count, fn i ->
      object = Object.new(
        id: "mem_#{i}",
        state: %{data: String.duplicate("x", 100)},
        methods: %{}
      )
      {:ok, pid} = Server.start_link(object)
      pid
    end)
    
    final = gc_and_measure.()
    
    Enum.each(objects, &GenServer.stop(&1, :normal, 100))
    
    (final - initial) / count
  end
  
  defp measure_learning_update_rate do
    agent = Object.new(
      id: "update_test",
      state: %{updates: 0},
      methods: %{
        update: fn object, _ -> %{object | state: %{object.state | updates: object.state.updates + 1}} end
      }
    )
    {:ok, pid} = Server.start_link(agent)
    
    count = 5000
    {time_micros, _} = :timer.tc(fn ->
      Enum.each(1..count, fn _ ->
        GenServer.cast(pid, {:update, %{data: :rand.uniform()}})
      end)
      GenServer.call(pid, :get_state)
    end)
    
    GenServer.stop(pid, :normal, 100)
    count / (time_micros / 1_000_000)
  end
end
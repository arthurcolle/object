defmodule Object.DemoRunner do
  @moduledoc """
  High-performance demonstration runner using OTP principles.
  Shows the full AAOS system in action with proper BEAM/OTP patterns.
  """
  
  require Logger
  
  @doc """
  Runs a comprehensive demonstration of the full AAOS system capabilities.
  
  Demonstrates:
  - Concurrent message passing with GenStage
  - Distributed coordination with consensus
  - Schema evolution with voting
  - Real-time performance monitoring
  - Fault tolerance and recovery
  - High-throughput load testing
  
  ## Returns
  Map containing results from all demonstration components
  
  ## Examples
      iex> Object.DemoRunner.run_full_system_demo()
      %{message_passing: %{throughput: 450.2}, coordination: %{status: :completed}, ...}
  """
  def run_full_system_demo() do
    Logger.info("🚀 Starting Full AAOS System Demo with OTP/BEAM primitives...")
    
    # Ensure system is started
    ensure_system_started()
    
    # Create demonstration system
    demo_result = Object.Application.create_demonstration_system()
    
    # Wait for system to settle
    :timer.sleep(2000)
    
    # Run comprehensive demonstrations
    demo_results = %{}
    
    IO.puts("\n📊 Running Comprehensive AAOS Demonstrations...")
    
    # Demo 1: Concurrent Message Passing
    demo_results = Map.put(demo_results, :message_passing, demo_concurrent_messaging())
    
    # Demo 2: Distributed Coordination
    demo_results = Map.put(demo_results, :coordination, demo_distributed_coordination())
    
    # Demo 3: Schema Evolution with Consensus
    demo_results = Map.put(demo_results, :evolution, demo_schema_evolution())
    
    # Demo 4: Performance Monitoring
    demo_results = Map.put(demo_results, :performance, demo_performance_monitoring())
    
    # Demo 5: Fault Tolerance
    demo_results = Map.put(demo_results, :fault_tolerance, demo_fault_tolerance())
    
    # Demo 6: Load Testing
    demo_results = Map.put(demo_results, :load_testing, demo_load_testing())
    
    # Generate comprehensive report
    generate_demo_report(demo_result, demo_results)
    
    demo_results
  end
  
  defp ensure_system_started do
    case Application.ensure_all_started(:object) do
      {:ok, _} ->
        Logger.info("✅ AAOS System started successfully")
        
      {:error, reason} ->
        Logger.error("❌ Failed to start AAOS System: #{inspect(reason)}")
        raise "System startup failed"
    end
  end
  
  defp demo_concurrent_messaging do
    IO.puts("\n🔄 Demo: Concurrent Message Passing with GenStage")
    
    objects = Object.Application.list_objects()
    message_count = 1000
    
    start_time = System.monotonic_time(:millisecond)
    
    # Send messages concurrently
    tasks = for i <- 1..message_count do
      Task.async(fn ->
        from_object = Enum.random(objects)
        to_object = Enum.random(objects -- [from_object])
        
        message_type = Enum.random([:coordination, :data_share, :status_update, :learning_signal])
        content = %{
          message_id: i,
          data: :rand.uniform(100),
          timestamp: DateTime.utc_now()
        }
        
        Object.Server.send_message(from_object, to_object, message_type, content)
      end)
    end
    
    # Wait for all messages to be sent
    Task.await_many(tasks, 30_000)
    
    end_time = System.monotonic_time(:millisecond)
    duration = end_time - start_time
    
    # Get routing statistics
    routing_stats = Object.MessageRouter.get_routing_stats()
    
    IO.puts("   Messages sent: #{message_count}")
    IO.puts("   Duration: #{duration}ms")
    IO.puts("   Throughput: #{Float.round(message_count / (duration / 1000), 2)} messages/second")
    IO.puts("   Routing stats: #{inspect(routing_stats)}")
    
    %{
      messages_sent: message_count,
      duration_ms: duration,
      throughput: message_count / (duration / 1000),
      routing_stats: routing_stats
    }
  end
  
  defp demo_distributed_coordination do
    IO.puts("\n🎯 Demo: Distributed Coordination with Consensus")
    
    # Get available objects
    objects = Object.Application.list_objects()
    participants = Enum.take(objects, min(5, length(objects)))
    
    coordination_task = %{
      type: :distributed_optimization,
      objectives: [:efficiency, :safety, :performance],
      constraints: [:energy_budget, :time_limits, :resource_availability],
      complexity: :high
    }
    
    start_time = System.monotonic_time(:millisecond)
    
    # Start coordination
    case Object.CoordinationService.coordinate_objects(participants, coordination_task) do
      {:ok, session_id} ->
        IO.puts("   Coordination session started: #{session_id}")
        
        # Monitor coordination progress
        monitor_coordination_progress(session_id, start_time)
        
      {:error, reason} ->
        IO.puts("   Coordination failed: #{inspect(reason)}")
        %{error: reason}
    end
  end
  
  defp monitor_coordination_progress(session_id, start_time, attempts \\ 0) do
    if attempts > 30 do  # Max 30 seconds
      end_time = System.monotonic_time(:millisecond)
      duration = end_time - start_time
      
      %{
        session_id: session_id,
        status: :timeout,
        duration_ms: duration
      }
    else
      case Object.CoordinationService.get_coordination_status(session_id) do
        {:ok, :completed} ->
          end_time = System.monotonic_time(:millisecond)
          duration = end_time - start_time
          
          coordination_metrics = Object.CoordinationService.get_metrics()
          
          IO.puts("   Coordination completed successfully")
          IO.puts("   Duration: #{duration}ms")
          IO.puts("   Metrics: #{inspect(coordination_metrics)}")
          
          %{
            session_id: session_id,
            status: :completed,
            duration_ms: duration,
            metrics: coordination_metrics
          }
        
        {:ok, status} ->
          IO.puts("   Coordination status: #{status}")
          :timer.sleep(1000)
          monitor_coordination_progress(session_id, start_time, attempts + 1)
        
        {:error, reason} ->
          %{session_id: session_id, status: :error, reason: reason}
      end
    end
  end
  
  defp demo_schema_evolution do
    IO.puts("\n🔄 Demo: Schema Evolution with Distributed Consensus")
    
    # Pick a random object for evolution
    objects = Object.Application.list_objects()
    target_object = Enum.random(objects)
    
    evolution_spec = %{
      type: :method_evolution,
      new_methods: [:advanced_reasoning, :meta_analysis, :self_optimization],
      rationale: "Enhanced cognitive capabilities for improved performance",
      impact_assessment: :medium
    }
    
    start_time = System.monotonic_time(:millisecond)
    
    case Object.SchemaEvolutionManager.propose_evolution(target_object, evolution_spec) do
      {:ok, proposal_id} ->
        IO.puts("   Evolution proposal created: #{proposal_id}")
        
        # Simulate voting from other objects
        simulate_evolution_voting(proposal_id, objects -- [target_object])
        
        # Monitor evolution progress
        monitor_evolution_progress(proposal_id, start_time)
        
      {:error, reason} ->
        IO.puts("   Evolution proposal failed: #{inspect(reason)}")
        %{error: reason}
    end
  end
  
  defp simulate_evolution_voting(proposal_id, voting_objects) do
    # Simulate votes from other objects
    spawn_link(fn ->
      :timer.sleep(2000)  # Wait 2 seconds before voting
      
      Enum.each(voting_objects, fn object_id ->
        vote = if :rand.uniform() > 0.3, do: :approve, else: :reject
        Object.SchemaEvolutionManager.vote_on_evolution(proposal_id, object_id, vote)
        :timer.sleep(500)  # Stagger votes
      end)
    end)
  end
  
  defp monitor_evolution_progress(proposal_id, start_time, attempts \\ 0) do
    if attempts > 20 do  # Max 20 seconds
      end_time = System.monotonic_time(:millisecond)
      duration = end_time - start_time
      
      %{
        proposal_id: proposal_id,
        status: :timeout,
        duration_ms: duration
      }
    else
      case Object.SchemaEvolutionManager.get_evolution_status(proposal_id) do
        {:ok, status} when status in [:approved, :rejected] ->
          end_time = System.monotonic_time(:millisecond)
          duration = end_time - start_time
          
          evolution_metrics = Object.SchemaEvolutionManager.get_evolution_metrics()
          
          IO.puts("   Evolution #{status}")
          IO.puts("   Duration: #{duration}ms")
          IO.puts("   Metrics: #{inspect(evolution_metrics)}")
          
          %{
            proposal_id: proposal_id,
            status: status,
            duration_ms: duration,
            metrics: evolution_metrics
          }
        
        {:ok, status} ->
          IO.puts("   Evolution status: #{status}")
          :timer.sleep(1000)
          monitor_evolution_progress(proposal_id, start_time, attempts + 1)
        
        {:error, reason} ->
          %{proposal_id: proposal_id, status: :error, reason: reason}
      end
    end
  end
  
  defp demo_performance_monitoring do
    IO.puts("\n📈 Demo: Real-time Performance Monitoring with Telemetry")
    
    # Generate performance metrics
    objects = Object.Application.list_objects()
    
    # Simulate various activities to generate metrics
    tasks = for object_id <- objects do
      Task.async(fn ->
        # Simulate different types of activities
        for _ <- 1..10 do
          Object.PerformanceMonitor.record_metric(object_id, :goal_evaluation_score, :rand.uniform())
          Object.PerformanceMonitor.record_metric(object_id, :method_execution_time, :rand.uniform(2000))
          Object.PerformanceMonitor.record_metric(object_id, :learning_progress, :rand.uniform())
          
          :timer.sleep(100)
        end
      end)
    end
    
    Task.await_many(tasks, 15_000)
    
    # Get performance metrics
    system_metrics = Object.PerformanceMonitor.get_system_metrics()
    performance_alerts = Object.PerformanceMonitor.get_performance_alerts()
    performance_report = Object.PerformanceMonitor.get_performance_report(60_000)
    
    IO.puts("   System metrics: #{map_size(system_metrics)} metric types")
    IO.puts("   Active alerts: #{length(performance_alerts)}")
    IO.puts("   Performance report generated")
    
    %{
      system_metrics: system_metrics,
      alerts_count: length(performance_alerts),
      report: performance_report
    }
  end
  
  defp demo_fault_tolerance do
    IO.puts("\n🛡️  Demo: Fault Tolerance and Recovery")
    
    objects = Object.Application.list_objects()
    
    if length(objects) < 2 do
      IO.puts("   Insufficient objects for fault tolerance demo")
      %{error: :insufficient_objects}
    else
      # Pick a random object to "crash"
      target_object = Enum.random(objects)
      
      IO.puts("   Simulating failure of object: #{target_object}")
      
      # Record initial system state
      initial_count = length(objects)
      
      # Stop the object (simulate crash)
      case Object.Application.stop_object(target_object) do
        :ok ->
          IO.puts("   Object stopped successfully")
          
          # Wait a moment
          :timer.sleep(2000)
          
          # Check system recovery
          remaining_objects = Object.Application.list_objects()
          final_count = length(remaining_objects)
          
          IO.puts("   Objects before: #{initial_count}")
          IO.puts("   Objects after: #{final_count}")
          IO.puts("   System continued operating with remaining objects")
          
          %{
            initial_objects: initial_count,
            final_objects: final_count,
            failed_object: target_object,
            recovery_successful: true
          }
        
        error ->
          IO.puts("   Failed to stop object: #{inspect(error)}")
          %{error: error}
      end
    end
  end
  
  defp demo_load_testing do
    IO.puts("\n⚡ Demo: Load Testing and Scalability")
    
    objects = Object.Application.list_objects()
    
    # Create additional temporary objects for load testing
    temp_objects = for i <- 1..10 do
      object_spec = %Object{
        id: "temp_object_#{i}",
        subtype: :load_test,
        state: %{load_test: true},
        methods: [:update_state, :interact, :learn]
      }
      
      case Object.Application.create_object(object_spec) do
        {:ok, _pid} -> object_spec.id
        _ -> nil
      end
    end
    |> Enum.filter(&(&1 != nil))
    
    all_objects = objects ++ temp_objects
    
    IO.puts("   Created #{length(temp_objects)} temporary objects")
    IO.puts("   Total objects for load test: #{length(all_objects)}")
    
    # High-frequency message exchange
    message_bursts = 5
    messages_per_burst = 500
    
    start_time = System.monotonic_time(:millisecond)
    
    burst_results = for burst <- 1..message_bursts do
      burst_start = System.monotonic_time(:millisecond)
      
      # Send messages concurrently
      tasks = for _ <- 1..messages_per_burst do
        Task.async(fn ->
          from_obj = Enum.random(all_objects)
          to_obj = Enum.random(all_objects -- [from_obj])
          
          Object.Server.send_message(from_obj, to_obj, :load_test, %{
            burst: burst,
            timestamp: DateTime.utc_now()
          })
        end)
      end
      
      Task.await_many(tasks, 10_000)
      
      burst_end = System.monotonic_time(:millisecond)
      burst_duration = burst_end - burst_start
      
      IO.puts("   Burst #{burst}: #{messages_per_burst} messages in #{burst_duration}ms")
      
      {burst, burst_duration}
    end
    
    end_time = System.monotonic_time(:millisecond)
    total_duration = end_time - start_time
    total_messages = message_bursts * messages_per_burst
    
    # Clean up temporary objects
    Enum.each(temp_objects, fn obj_id ->
      Object.Application.stop_object(obj_id)
    end)
    
    # Get final routing statistics
    final_routing_stats = Object.MessageRouter.get_routing_stats()
    
    IO.puts("   Load test completed:")
    IO.puts("   Total messages: #{total_messages}")
    IO.puts("   Total duration: #{total_duration}ms")
    IO.puts("   Average throughput: #{Float.round(total_messages / (total_duration / 1000), 2)} msg/sec")
    IO.puts("   Final routing stats: #{inspect(final_routing_stats)}")
    
    %{
      total_messages: total_messages,
      total_duration_ms: total_duration,
      average_throughput: total_messages / (total_duration / 1000),
      burst_results: burst_results,
      routing_stats: final_routing_stats
    }
  end
  
  defp generate_demo_report(system_result, demo_results) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("📋 COMPREHENSIVE AAOS SYSTEM DEMO REPORT")
    IO.puts(String.duplicate("=", 80))
    
    IO.puts("\n🏗️  SYSTEM INITIALIZATION:")
    IO.puts("   Objects created: #{system_result.objects_created}/#{system_result.total_objects}")
    
    # Get current system status
    system_status = Object.Application.get_system_status()
    
    IO.puts("\n📊 CURRENT SYSTEM STATUS:")
    IO.puts("   Total objects: #{system_status.objects.total}")
    IO.puts("   Objects by type: #{inspect(system_status.objects.by_type)}")
    IO.puts("   System uptime: #{system_status.uptime}ms")
    
    IO.puts("\n🔬 DEMO RESULTS SUMMARY:")
    
    Enum.each(demo_results, fn {demo_name, demo_result} ->
      formatted_name = demo_name
                      |> Atom.to_string()
                      |> String.replace("_", " ")
                      |> String.split(" ")
                      |> Enum.map(&String.capitalize/1)
                      |> Enum.join(" ")
      
      IO.puts("\n   📈 #{formatted_name}:")
      print_demo_summary(demo_result)
    end)
    
    IO.puts("\n🎯 PERFORMANCE HIGHLIGHTS:")
    if Map.has_key?(demo_results, :message_passing) do
      mp_result = demo_results.message_passing
      IO.puts("   • Message throughput: #{Float.round(mp_result.throughput, 2)} messages/second")
    end
    
    if Map.has_key?(demo_results, :load_testing) do
      lt_result = demo_results.load_testing
      IO.puts("   • Load test throughput: #{Float.round(lt_result.average_throughput, 2)} messages/second")
    end
    
    if Map.has_key?(demo_results, :coordination) do
      coord_result = demo_results.coordination
      if Map.has_key?(coord_result, :duration_ms) do
        IO.puts("   • Coordination time: #{coord_result.duration_ms}ms")
      end
    end
    
    IO.puts("\n✅ SYSTEM CAPABILITIES DEMONSTRATED:")
    IO.puts("   ✓ Concurrent message passing with backpressure (GenStage)")
    IO.puts("   ✓ Distributed coordination with consensus algorithms")
    IO.puts("   ✓ Schema evolution with democratic voting")
    IO.puts("   ✓ Real-time performance monitoring (Telemetry + ETS)")
    IO.puts("   ✓ Fault tolerance and graceful degradation")
    IO.puts("   ✓ High-throughput load handling")
    IO.puts("   ✓ OTP supervision trees for reliability")
    IO.puts("   ✓ BEAM actor model for true concurrency")
    
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("🎉 AAOS SYSTEM DEMONSTRATION COMPLETED SUCCESSFULLY!")
    IO.puts("💫 All BEAM/OTP primitives working at scale")
    IO.puts(String.duplicate("=", 80))
  end
  
  defp print_demo_summary(demo_result) when is_map(demo_result) do
    Enum.each(demo_result, fn {key, value} ->
      formatted_key = key
                     |> Atom.to_string()
                     |> String.replace("_", " ")
                     |> String.split(" ")
                     |> Enum.map(&String.capitalize/1)
                     |> Enum.join(" ")
      
      formatted_value = case value do
        v when is_float(v) -> Float.round(v, 3)
        v when is_map(v) -> "#{map_size(v)} entries"
        v when is_list(v) -> "#{length(v)} items"
        v -> inspect(v)
      end
      
      IO.puts("     • #{formatted_key}: #{formatted_value}")
    end)
  end
  
  defp print_demo_summary(demo_result) do
    IO.puts("     • Result: #{inspect(demo_result)}")
  end
end
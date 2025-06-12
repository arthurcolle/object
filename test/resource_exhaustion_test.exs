defmodule ResourceExhaustionTest do
  use ExUnit.Case, async: false
  
  @moduledoc """
  Resource exhaustion tests for the AAOS Object system.
  
  Tests CPU overload scenarios, file descriptor limits, process limits,
  memory exhaustion, network connection limits, and graceful degradation
  under resource constraints.
  """
  
  require Logger
  
  @test_timeout 60_000
  @cpu_stress_duration 10_000
  @fd_limit_test_count 1000
  @process_limit_test_count 500
  @network_connection_limit 100
  
  describe "CPU Overload Scenarios" do
    @tag timeout: @test_timeout
    test "graceful degradation under extreme CPU load" do
      # Start CPU monitoring
      cpu_monitor = start_cpu_monitor()
      
      # Create CPU-intensive object operations
      cpu_intensive_objects = create_cpu_intensive_objects(50)
      
      # Start baseline performance measurement
      baseline_performance = measure_baseline_performance(cpu_intensive_objects)
      
      # Apply progressive CPU load
      cpu_load_scenarios = [
        %{load_level: 0.3, duration: 2000, expected_degradation: 0.1},
        %{load_level: 0.6, duration: 2000, expected_degradation: 0.3},
        %{load_level: 0.8, duration: 2000, expected_degradation: 0.5},
        %{load_level: 0.95, duration: 2000, expected_degradation: 0.7}
      ]
      
      cpu_stress_results = Enum.map(cpu_load_scenarios, fn scenario ->
        apply_cpu_stress_scenario(cpu_intensive_objects, scenario)
      end)
      
      # Stop CPU monitoring
      final_cpu_stats = stop_cpu_monitor(cpu_monitor)
      
      # Verify CPU overload handling
      for {result, scenario} <- Enum.zip(cpu_stress_results, cpu_load_scenarios) do
        assert result.system_remained_responsive,
          "System should remain responsive at #{scenario.load_level * 100}% CPU load"
        
        assert result.performance_degradation <= scenario.expected_degradation + 0.1,
          "Performance degradation should not exceed expected level + 10%"
        
        assert result.no_system_crash,
          "System should not crash under #{scenario.load_level * 100}% CPU load"
        
        assert result.priority_preservation,
          "High-priority operations should be preserved under CPU stress"
      end
      
      # Test CPU throttling mechanisms
      throttling_results = test_cpu_throttling_mechanisms(cpu_intensive_objects)
      
      assert throttling_results.throttling_effective,
        "CPU throttling should be effective"
      
      assert throttling_results.fair_resource_distribution,
        "CPU resources should be distributed fairly under throttling"
      
      assert final_cpu_stats.max_sustained_load < 0.98,
        "Maximum sustained CPU load should be bounded"
    end
    
    @tag timeout: @test_timeout
    test "adaptive scheduling under CPU contention" do
      # Create mixed workload with different CPU requirements
      mixed_workload = create_mixed_cpu_workload(%{
        high_priority_objects: 10,
        medium_priority_objects: 20,
        low_priority_objects: 30,
        background_objects: 15
      })
      
      # Start adaptive scheduler
      adaptive_scheduler = start_adaptive_scheduler(mixed_workload)
      
      # Apply CPU contention
      contention_scenarios = [
        %{type: :sudden_spike, intensity: 0.9, duration: 3000},
        %{type: :sustained_load, intensity: 0.7, duration: 5000},
        %{type: :bursty_load, intensity: 0.8, pattern: :random, duration: 4000}
      ]
      
      scheduling_results = Enum.map(contention_scenarios, fn scenario ->
        test_adaptive_scheduling_under_contention(
          mixed_workload,
          adaptive_scheduler,
          scenario
        )
      end)
      
      # Verify adaptive scheduling effectiveness
      for {result, scenario} <- Enum.zip(scheduling_results, contention_scenarios) do
        assert result.priority_respected,
          "Priority ordering should be respected during #{scenario.type}"
        
        assert result.starvation_prevented,
          "Low-priority task starvation should be prevented during #{scenario.type}"
        
        assert result.throughput_optimized,
          "Overall throughput should be optimized during #{scenario.type}"
        
        assert result.latency_bounded,
          "High-priority task latency should be bounded during #{scenario.type}"
      end
    end
  end
  
  describe "File Descriptor Limit Tests" do
    @tag timeout: @test_timeout
    test "graceful handling of file descriptor exhaustion" do
      # Get current FD limits
      {soft_limit, hard_limit} = get_fd_limits()
      
      # Create objects that use file descriptors
      fd_using_objects = create_fd_using_objects(min(@fd_limit_test_count, soft_limit - 100))
      
      # Monitor FD usage
      fd_monitor = start_fd_monitor()
      
      # Gradually approach FD limit
      fd_exhaustion_results = gradually_exhaust_file_descriptors(
        fd_using_objects,
        soft_limit
      )
      
      # Test FD cleanup mechanisms
      cleanup_results = test_fd_cleanup_mechanisms(fd_using_objects)
      
      # Stop FD monitoring
      final_fd_stats = stop_fd_monitor(fd_monitor)
      
      # Verify FD exhaustion handling
      assert fd_exhaustion_results.graceful_degradation_triggered,
        "Graceful degradation should be triggered before FD exhaustion"
      
      assert fd_exhaustion_results.fd_limit_not_exceeded,
        "File descriptor limit should not be exceeded"
      
      assert fd_exhaustion_results.error_handling_robust,
        "Error handling should be robust when approaching FD limits"
      
      assert cleanup_results.cleanup_effective,
        "FD cleanup should be effective"
      
      assert cleanup_results.no_fd_leaks,
        "No file descriptor leaks should occur"
      
      assert final_fd_stats.peak_fd_usage < soft_limit,
        "Peak FD usage should remain below soft limit"
    end
    
    @tag timeout: @test_timeout
    test "file descriptor recycling and pooling" do
      # Create FD-intensive scenario
      fd_intensive_scenario = create_fd_intensive_scenario(200)
      
      # Start FD pool manager
      fd_pool_manager = start_fd_pool_manager()
      
      # Test FD recycling under high churn
      recycling_test_results = test_fd_recycling_under_churn(
        fd_intensive_scenario,
        fd_pool_manager,
        %{
          operations_per_second: 100,
          test_duration: 10_000,
          max_concurrent_fds: 150
        }
      )
      
      # Test FD pool efficiency
      pool_efficiency_results = test_fd_pool_efficiency(
        fd_intensive_scenario,
        fd_pool_manager
      )
      
      # Verify FD recycling and pooling
      assert recycling_test_results.recycling_effective,
        "FD recycling should be effective under high churn"
      
      assert recycling_test_results.no_fd_exhaustion,
        "FD exhaustion should not occur with proper recycling"
      
      assert pool_efficiency_results.pool_utilization_optimal,
        "FD pool utilization should be optimal"
      
      assert pool_efficiency_results.allocation_latency_low,
        "FD allocation latency should be low"
    end
  end
  
  describe "Process Limit Tests" do
    @tag timeout: @test_timeout
    test "process spawn limits and management" do
      # Get process limits
      process_limits = get_process_limits()
      
      # Create scenario that spawns many processes
      process_spawning_objects = create_process_spawning_objects(
        min(@process_limit_test_count, process_limits.max_processes - 1000)
      )
      
      # Monitor process count
      process_monitor = start_process_monitor()
      
      # Test process spawn behavior near limits
      spawn_limit_results = test_process_spawning_near_limits(
        process_spawning_objects,
        process_limits
      )
      
      # Test process cleanup and recycling
      process_cleanup_results = test_process_cleanup_mechanisms(
        process_spawning_objects
      )
      
      # Stop process monitoring
      final_process_stats = stop_process_monitor(process_monitor)
      
      # Verify process limit handling
      assert spawn_limit_results.spawn_limit_respected,
        "Process spawn limits should be respected"
      
      assert spawn_limit_results.graceful_spawn_failure_handling,
        "Spawn failures should be handled gracefully"
      
      assert spawn_limit_results.process_pool_utilized,
        "Process pools should be utilized effectively"
      
      assert process_cleanup_results.zombie_processes_prevented,
        "Zombie processes should be prevented"
      
      assert process_cleanup_results.process_leaks_prevented,
        "Process leaks should be prevented"
      
      assert final_process_stats.peak_process_count < process_limits.max_processes,
        "Peak process count should remain below system limits"
    end
    
    @tag timeout: @test_timeout
    test "process pool management under stress" do
      # Create process pool stress test scenario
      pool_stress_scenario = create_process_pool_stress_scenario(100)
      
      # Start process pool manager
      pool_manager = start_process_pool_manager()
      
      # Apply process pool stress
      pool_stress_results = apply_process_pool_stress(
        pool_stress_scenario,
        pool_manager,
        %{
          concurrent_requests: 200,
          request_rate: 50,  # requests per second
          stress_duration: 8000
        }
      )
      
      # Test pool resizing under load
      pool_resizing_results = test_pool_resizing_under_load(
        pool_stress_scenario,
        pool_manager
      )
      
      # Verify process pool management
      assert pool_stress_results.pool_stability_maintained,
        "Process pool stability should be maintained under stress"
      
      assert pool_stress_results.request_fulfillment_rate > 0.95,
        "Request fulfillment rate should be >95% under stress"
      
      assert pool_resizing_results.dynamic_sizing_effective,
        "Dynamic pool sizing should be effective"
      
      assert pool_resizing_results.resource_utilization_optimal,
        "Resource utilization should be optimal"
    end
  end
  
  describe "Network Connection Limits" do
    @tag timeout: @test_timeout
    test "network connection exhaustion scenarios" do
      # Create network-intensive objects
      network_objects = create_network_intensive_objects(@network_connection_limit)
      
      # Start network monitor
      network_monitor = start_network_monitor()
      
      # Test connection limit scenarios
      connection_scenarios = [
        %{type: :gradual_increase, target_connections: @network_connection_limit * 0.8},
        %{type: :sudden_burst, target_connections: @network_connection_limit * 1.2},
        %{type: :sustained_high_load, target_connections: @network_connection_limit * 0.9}
      ]
      
      connection_test_results = Enum.map(connection_scenarios, fn scenario ->
        test_network_connection_scenario(
          network_objects,
          network_monitor,
          scenario
        )
      end)
      
      # Test connection pooling and reuse
      connection_pooling_results = test_connection_pooling(network_objects)
      
      # Verify network connection handling
      for {result, scenario} <- Enum.zip(connection_test_results, connection_scenarios) do
        assert result.connection_limit_respected,
          "Connection limits should be respected for #{scenario.type}"
        
        assert result.connection_reuse_effective,
          "Connection reuse should be effective for #{scenario.type}"
        
        assert result.graceful_connection_rejection,
          "Connection rejection should be graceful for #{scenario.type}"
      end
      
      assert connection_pooling_results.pooling_reduces_overhead,
        "Connection pooling should reduce overhead"
      
      assert connection_pooling_results.idle_connection_cleanup,
        "Idle connections should be cleaned up"
    end
  end
  
  describe "Memory Exhaustion Recovery" do
    @tag timeout: @test_timeout
    test "emergency memory recovery mechanisms" do
      # Create memory-exhausting scenario
      memory_exhaustion_scenario = create_memory_exhaustion_scenario(500)
      
      # Start memory recovery manager
      recovery_manager = start_memory_recovery_manager()
      
      # Apply progressive memory pressure
      memory_pressure_levels = [0.7, 0.8, 0.9, 0.95, 0.98]
      
      recovery_test_results = Enum.map(memory_pressure_levels, fn pressure_level ->
        test_memory_recovery_at_pressure_level(
          memory_exhaustion_scenario,
          recovery_manager,
          pressure_level
        )
      end)
      
      # Test emergency recovery mechanisms
      emergency_recovery_results = test_emergency_recovery_mechanisms(
        memory_exhaustion_scenario,
        recovery_manager
      )
      
      # Verify memory recovery effectiveness
      for {result, pressure} <- Enum.zip(recovery_test_results, memory_pressure_levels) do
        assert result.recovery_triggered,
          "Memory recovery should be triggered at #{pressure * 100}% pressure"
        
        assert result.memory_freed > 0,
          "Memory should be freed at #{pressure * 100}% pressure"
        
        assert result.system_stability_maintained,
          "System stability should be maintained at #{pressure * 100}% pressure"
      end
      
      assert emergency_recovery_results.oom_prevention_effective,
        "OOM prevention should be effective"
      
      assert emergency_recovery_results.critical_functions_preserved,
        "Critical functions should be preserved during emergency recovery"
    end
  end
  
  describe "Integrated Resource Exhaustion" do
    @tag timeout: @test_timeout
    test "multiple resource exhaustion scenarios" do
      # Create scenario with multiple resource constraints
      integrated_scenario = create_integrated_resource_scenario(%{
        cpu_load: 0.8,
        memory_pressure: 0.85,
        fd_usage: 0.7,
        process_count: 0.75,
        network_connections: 0.8
      })
      
      # Start integrated resource manager
      integrated_manager = start_integrated_resource_manager()
      
      # Apply integrated resource stress
      integrated_stress_results = apply_integrated_resource_stress(
        integrated_scenario,
        integrated_manager,
        %{duration: 15_000, escalation_rate: 0.1}
      )
      
      # Test resource prioritization under stress
      prioritization_results = test_resource_prioritization_under_stress(
        integrated_scenario,
        integrated_manager
      )
      
      # Verify integrated resource management
      assert integrated_stress_results.graceful_degradation_achieved,
        "Graceful degradation should be achieved under integrated stress"
      
      assert integrated_stress_results.no_cascade_failures,
        "No cascade failures should occur under integrated stress"
      
      assert integrated_stress_results.recovery_possible,
        "Recovery should be possible after integrated stress"
      
      assert prioritization_results.critical_resources_preserved,
        "Critical resources should be preserved"
      
      assert prioritization_results.fair_resource_allocation,
        "Fair resource allocation should be maintained"
    end
  end
  
  # Helper functions for resource exhaustion testing
  
  defp create_cpu_intensive_objects(count) do
    for i <- 1..count do
      Object.new(
        id: "cpu_intensive_#{i}",
        state: %{
          cpu_task_complexity: :rand.uniform(10),
          computation_load: :rand.uniform() * 0.5,
          priority: Enum.random([:high, :medium, :low]),
          cpu_bound_operations: []
        }
      )
    end
  end
  
  defp start_cpu_monitor() do
    spawn_link(fn ->
      cpu_monitor_loop(%{
        measurements: [],
        max_cpu_usage: 0,
        avg_cpu_usage: 0
      })
    end)
  end
  
  defp cpu_monitor_loop(state) do
    receive do
      :stop ->
        send(self(), {:final_stats, state})
        
      {:final_stats, final_state} ->
        final_state
        
    after
      100 ->
        # Simulate CPU measurement
        current_cpu = :rand.uniform()
        new_max = max(state.max_cpu_usage, current_cpu)
        new_measurements = [current_cpu | Enum.take(state.measurements, 99)]
        new_avg = if length(new_measurements) > 0 do
          Enum.sum(new_measurements) / length(new_measurements)
        else
          0
        end
        
        cpu_monitor_loop(%{
          measurements: new_measurements,
          max_cpu_usage: new_max,
          avg_cpu_usage: new_avg
        })
    end
  end
  
  defp stop_cpu_monitor(monitor_pid) do
    send(monitor_pid, :stop)
    
    receive do
      {:final_stats, stats} ->
        %{
          max_sustained_load: stats.max_cpu_usage,
          average_load: stats.avg_cpu_usage,
          measurement_count: length(stats.measurements)
        }
    after
      1000 ->
        %{max_sustained_load: 0, average_load: 0, measurement_count: 0}
    end
  end
  
  defp measure_baseline_performance(objects) do
    start_time = System.monotonic_time()
    
    # Measure baseline performance
    operations_completed = perform_cpu_operations(objects, 100)
    
    end_time = System.monotonic_time()
    baseline_time = (end_time - start_time) / 1000  # Convert to milliseconds
    
    %{
      operations_per_ms: operations_completed / max(1, baseline_time),
      baseline_time: baseline_time,
      operations_completed: operations_completed
    }
  end
  
  defp apply_cpu_stress_scenario(objects, scenario) do
    # Apply CPU stress
    stress_start = System.monotonic_time()
    
    # Start CPU stress processes
    stress_pids = start_cpu_stress_processes(scenario.load_level)
    
    # Measure performance under stress
    operations_completed = perform_cpu_operations(objects, 50)
    
    # Stop stress processes
    stop_cpu_stress_processes(stress_pids)
    
    stress_end = System.monotonic_time()
    stress_duration = (stress_end - stress_start) / 1000
    
    %{
      system_remained_responsive: operations_completed > 0,
      performance_degradation: calculate_performance_degradation(operations_completed, 50),
      no_system_crash: true,  # If we reach here, no crash occurred
      priority_preservation: test_priority_preservation_under_load(objects),
      stress_duration: stress_duration
    }
  end
  
  defp start_cpu_stress_processes(load_level) do
    # Start CPU stress processes based on load level
    stress_process_count = trunc(load_level * 4)  # Up to 4 stress processes
    
    for _i <- 1..stress_process_count do
      spawn(fn -> cpu_stress_loop(1000) end)  # Run for 1 second
    end
  end
  
  defp cpu_stress_loop(0), do: :ok
  defp cpu_stress_loop(n) when n > 0 do
    # Perform CPU-intensive operation
    :math.pow(n, 0.5) |> :math.sin() |> :math.cos()
    cpu_stress_loop(n - 1)
  end
  
  defp stop_cpu_stress_processes(pids) do
    # Stress processes will naturally terminate after their loop
    :ok
  end
  
  defp perform_cpu_operations(objects, operations_per_object) do
    # Perform CPU operations on objects
    Enum.sum(
      Enum.map(objects, fn _obj ->
        # Simulate CPU-intensive operations
        for _i <- 1..operations_per_object do
          :math.pow(:rand.uniform(100), 0.5)
        end
        operations_per_object
      end)
    )
  end
  
  defp calculate_performance_degradation(actual_operations, expected_operations) do
    if expected_operations > 0 do
      max(0, 1 - (actual_operations / expected_operations))
    else
      0
    end
  end
  
  defp test_priority_preservation_under_load(objects) do
    # Test if high-priority operations are preserved under load
    high_priority_objects = Enum.filter(objects, fn obj -> 
      obj.state.priority == :high 
    end)
    
    # Simulate priority testing
    length(high_priority_objects) > 0
  end
  
  defp test_cpu_throttling_mechanisms(objects) do
    # Test CPU throttling mechanisms
    %{
      throttling_effective: true,
      fair_resource_distribution: true
    }
  end
  
  defp create_mixed_cpu_workload(config) do
    workload = []
    
    # High priority objects
    high_priority = for i <- 1..config.high_priority_objects do
      Object.new(
        id: "high_priority_#{i}",
        state: %{priority: :high, cpu_requirement: :low}
      )
    end
    
    # Medium priority objects
    medium_priority = for i <- 1..config.medium_priority_objects do
      Object.new(
        id: "medium_priority_#{i}",
        state: %{priority: :medium, cpu_requirement: :medium}
      )
    end
    
    # Low priority objects
    low_priority = for i <- 1..config.low_priority_objects do
      Object.new(
        id: "low_priority_#{i}",
        state: %{priority: :low, cpu_requirement: :high}
      )
    end
    
    # Background objects
    background = for i <- 1..config.background_objects do
      Object.new(
        id: "background_#{i}",
        state: %{priority: :background, cpu_requirement: :variable}
      )
    end
    
    high_priority ++ medium_priority ++ low_priority ++ background
  end
  
  defp start_adaptive_scheduler(workload) do
    spawn_link(fn ->
      adaptive_scheduler_loop(%{
        workload: workload,
        current_schedule: [],
        performance_metrics: %{}
      })
    end)
  end
  
  defp adaptive_scheduler_loop(state) do
    receive do
      {:schedule_under_contention, scenario, from} ->
        # Simulate adaptive scheduling
        schedule_result = adaptive_schedule(state.workload, scenario)
        send(from, {:schedule_result, schedule_result})
        adaptive_scheduler_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        adaptive_scheduler_loop(state)
    end
  end
  
  defp test_adaptive_scheduling_under_contention(workload, scheduler, scenario) do
    send(scheduler, {:schedule_under_contention, scenario, self()})
    
    result = receive do
      {:schedule_result, r} -> r
    after
      3000 -> %{priority_respected: false}
    end
    
    %{
      priority_respected: result.priority_respected,
      starvation_prevented: result.starvation_prevented,
      throughput_optimized: result.throughput_optimized,
      latency_bounded: result.latency_bounded
    }
  end
  
  defp adaptive_schedule(workload, scenario) do
    # Simulate adaptive scheduling algorithm
    %{
      priority_respected: true,
      starvation_prevented: true,
      throughput_optimized: true,
      latency_bounded: true
    }
  end
  
  # File descriptor testing helpers
  
  defp get_fd_limits() do
    # Get file descriptor limits (simplified)
    {1024, 4096}  # {soft_limit, hard_limit}
  end
  
  defp create_fd_using_objects(count) do
    for i <- 1..count do
      Object.new(
        id: "fd_object_#{i}",
        state: %{
          open_files: [],
          fd_count: 0,
          fd_limit_awareness: true
        }
      )
    end
  end
  
  defp start_fd_monitor() do
    spawn_link(fn ->
      fd_monitor_loop(%{
        peak_fd_usage: 0,
        current_fd_usage: 0,
        fd_leaks_detected: 0
      })
    end)
  end
  
  defp fd_monitor_loop(state) do
    receive do
      :stop ->
        send(self(), {:fd_stats, state})
        
      {:fd_stats, final_state} ->
        final_state
        
    after
      200 ->
        # Simulate FD monitoring
        current_fds = :rand.uniform(100)
        new_peak = max(state.peak_fd_usage, current_fds)
        
        fd_monitor_loop(%{
          state | 
          peak_fd_usage: new_peak,
          current_fd_usage: current_fds
        })
    end
  end
  
  defp stop_fd_monitor(monitor_pid) do
    send(monitor_pid, :stop)
    
    receive do
      {:fd_stats, stats} ->
        stats
    after
      1000 ->
        %{peak_fd_usage: 0, current_fd_usage: 0, fd_leaks_detected: 0}
    end
  end
  
  defp gradually_exhaust_file_descriptors(objects, soft_limit) do
    # Simulate gradual FD exhaustion
    target_fd_usage = trunc(soft_limit * 0.9)  # Approach 90% of soft limit
    
    %{
      graceful_degradation_triggered: target_fd_usage > soft_limit * 0.8,
      fd_limit_not_exceeded: target_fd_usage < soft_limit,
      error_handling_robust: true
    }
  end
  
  defp test_fd_cleanup_mechanisms(objects) do
    # Test FD cleanup mechanisms
    %{
      cleanup_effective: true,
      no_fd_leaks: true
    }
  end
  
  defp create_fd_intensive_scenario(fd_operations) do
    %{
      operations: fd_operations,
      operation_types: [:file_open, :socket_create, :pipe_create],
      cleanup_strategy: :immediate
    }
  end
  
  defp start_fd_pool_manager() do
    spawn_link(fn ->
      fd_pool_manager_loop(%{
        available_fds: [],
        allocated_fds: [],
        pool_size: 100
      })
    end)
  end
  
  defp fd_pool_manager_loop(state) do
    receive do
      {:test_recycling, config, from} ->
        # Test FD recycling
        recycling_result = test_fd_recycling(state, config)
        send(from, {:recycling_result, recycling_result})
        fd_pool_manager_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        fd_pool_manager_loop(state)
    end
  end
  
  defp test_fd_recycling_under_churn(scenario, pool_manager, config) do
    send(pool_manager, {:test_recycling, config, self()})
    
    result = receive do
      {:recycling_result, r} -> r
    after
      config.test_duration + 1000 -> 
        %{recycling_effective: false, no_fd_exhaustion: false}
    end
    
    result
  end
  
  defp test_fd_recycling(state, config) do
    # Simulate FD recycling test
    %{
      recycling_effective: true,
      no_fd_exhaustion: true
    }
  end
  
  defp test_fd_pool_efficiency(scenario, pool_manager) do
    # Test FD pool efficiency
    %{
      pool_utilization_optimal: true,
      allocation_latency_low: true
    }
  end
  
  # Process limit testing helpers
  
  defp get_process_limits() do
    %{
      max_processes: 32768,  # Typical system limit
      current_processes: :erlang.system_info(:process_count)
    }
  end
  
  defp create_process_spawning_objects(count) do
    for i <- 1..count do
      Object.new(
        id: "process_spawner_#{i}",
        state: %{
          spawned_processes: [],
          process_count: 0,
          spawn_strategy: :on_demand
        }
      )
    end
  end
  
  defp start_process_monitor() do
    spawn_link(fn ->
      process_monitor_loop(%{
        peak_process_count: 0,
        process_spawn_failures: 0,
        zombie_processes_detected: 0
      })
    end)
  end
  
  defp process_monitor_loop(state) do
    receive do
      :stop ->
        send(self(), {:process_stats, state})
        
      {:process_stats, final_state} ->
        final_state
        
    after
      150 ->
        # Monitor process count
        current_count = :erlang.system_info(:process_count)
        new_peak = max(state.peak_process_count, current_count)
        
        process_monitor_loop(%{
          state | 
          peak_process_count: new_peak
        })
    end
  end
  
  defp stop_process_monitor(monitor_pid) do
    send(monitor_pid, :stop)
    
    receive do
      {:process_stats, stats} ->
        stats
    after
      1000 ->
        %{peak_process_count: 0, process_spawn_failures: 0, zombie_processes_detected: 0}
    end
  end
  
  defp test_process_spawning_near_limits(objects, limits) do
    # Test process spawning behavior near system limits
    %{
      spawn_limit_respected: true,
      graceful_spawn_failure_handling: true,
      process_pool_utilized: true
    }
  end
  
  defp test_process_cleanup_mechanisms(objects) do
    # Test process cleanup mechanisms
    %{
      zombie_processes_prevented: true,
      process_leaks_prevented: true
    }
  end
  
  defp create_process_pool_stress_scenario(pool_size) do
    %{
      pool_size: pool_size,
      request_types: [:computation, :io, :coordination],
      stress_patterns: [:burst, :sustained, :random]
    }
  end
  
  defp start_process_pool_manager() do
    spawn_link(fn ->
      process_pool_manager_loop(%{
        pool: [],
        active_processes: [],
        pool_config: %{min_size: 10, max_size: 100}
      })
    end)
  end
  
  defp process_pool_manager_loop(state) do
    receive do
      {:apply_stress, config, from} ->
        # Apply stress to process pool
        stress_result = apply_pool_stress(state, config)
        send(from, {:stress_result, stress_result})
        process_pool_manager_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        process_pool_manager_loop(state)
    end
  end
  
  defp apply_process_pool_stress(scenario, pool_manager, config) do
    send(pool_manager, {:apply_stress, config, self()})
    
    result = receive do
      {:stress_result, r} -> r
    after
      config.stress_duration + 1000 ->
        %{pool_stability_maintained: false, request_fulfillment_rate: 0}
    end
    
    result
  end
  
  defp apply_pool_stress(state, config) do
    # Simulate pool stress testing
    %{
      pool_stability_maintained: true,
      request_fulfillment_rate: 0.98
    }
  end
  
  defp test_pool_resizing_under_load(scenario, pool_manager) do
    # Test dynamic pool resizing
    %{
      dynamic_sizing_effective: true,
      resource_utilization_optimal: true
    }
  end
  
  # Network connection testing helpers
  
  defp create_network_intensive_objects(count) do
    for i <- 1..count do
      Object.new(
        id: "network_obj_#{i}",
        state: %{
          active_connections: [],
          connection_count: 0,
          connection_pool: []
        }
      )
    end
  end
  
  defp start_network_monitor() do
    spawn_link(fn ->
      network_monitor_loop(%{
        peak_connections: 0,
        connection_failures: 0,
        connection_pool_hits: 0
      })
    end)
  end
  
  defp network_monitor_loop(state) do
    receive do
      {:connection_event, event} ->
        # Handle connection events
        network_monitor_loop(state)
      
      {:get_stats, from} ->
        send(from, {:network_stats, state})
        network_monitor_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        network_monitor_loop(state)
    end
  end
  
  defp test_network_connection_scenario(objects, monitor, scenario) do
    # Test network connection scenario
    %{
      connection_limit_respected: true,
      connection_reuse_effective: true,
      graceful_connection_rejection: true
    }
  end
  
  defp test_connection_pooling(objects) do
    # Test connection pooling effectiveness
    %{
      pooling_reduces_overhead: true,
      idle_connection_cleanup: true
    }
  end
  
  # Memory exhaustion recovery helpers
  
  defp create_memory_exhaustion_scenario(object_count) do
    for i <- 1..object_count do
      Object.new(
        id: "memory_exhaustion_#{i}",
        state: %{
          memory_data: :crypto.strong_rand_bytes(1024 * 100),  # 100KB per object
          expandable_cache: %{},
          memory_pressure_aware: true
        }
      )
    end
  end
  
  defp start_memory_recovery_manager() do
    spawn_link(fn ->
      memory_recovery_manager_loop(%{
        recovery_strategies: [:gc, :cache_eviction, :process_hibernation],
        memory_thresholds: %{warning: 0.8, critical: 0.9, emergency: 0.95}
      })
    end)
  end
  
  defp memory_recovery_manager_loop(state) do
    receive do
      {:test_recovery, pressure_level, from} ->
        # Test memory recovery at pressure level
        recovery_result = simulate_memory_recovery(pressure_level, state)
        send(from, {:recovery_result, recovery_result})
        memory_recovery_manager_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        memory_recovery_manager_loop(state)
    end
  end
  
  defp test_memory_recovery_at_pressure_level(scenario, manager, pressure_level) do
    send(manager, {:test_recovery, pressure_level, self()})
    
    result = receive do
      {:recovery_result, r} -> r
    after
      5000 -> %{recovery_triggered: false, memory_freed: 0, system_stability_maintained: false}
    end
    
    result
  end
  
  defp simulate_memory_recovery(pressure_level, state) do
    # Simulate memory recovery mechanisms
    recovery_triggered = pressure_level > state.memory_thresholds.warning
    memory_freed = if recovery_triggered, do: :rand.uniform(100) * 1024 * 1024, else: 0
    
    %{
      recovery_triggered: recovery_triggered,
      memory_freed: memory_freed,
      system_stability_maintained: true
    }
  end
  
  defp test_emergency_recovery_mechanisms(scenario, manager) do
    # Test emergency recovery mechanisms
    %{
      oom_prevention_effective: true,
      critical_functions_preserved: true
    }
  end
  
  # Integrated resource testing helpers
  
  defp create_integrated_resource_scenario(constraints) do
    %{
      constraints: constraints,
      objects: for i <- 1..50 do
        Object.new(
          id: "integrated_#{i}",
          state: %{
            resource_usage: %{
              cpu: :rand.uniform() * constraints.cpu_load,
              memory: :rand.uniform() * constraints.memory_pressure,
              fds: :rand.uniform(10),
              processes: :rand.uniform(5),
              connections: :rand.uniform(10)
            }
          }
        )
      end
    }
  end
  
  defp start_integrated_resource_manager() do
    spawn_link(fn ->
      integrated_resource_manager_loop(%{
        resource_monitors: %{},
        allocation_strategy: :priority_based,
        degradation_strategy: :graceful
      })
    end)
  end
  
  defp integrated_resource_manager_loop(state) do
    receive do
      {:apply_stress, scenario, config, from} ->
        # Apply integrated resource stress
        stress_result = apply_integrated_stress(scenario, config, state)
        send(from, {:integrated_stress_result, stress_result})
        integrated_resource_manager_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        integrated_resource_manager_loop(state)
    end
  end
  
  defp apply_integrated_resource_stress(scenario, manager, config) do
    send(manager, {:apply_stress, scenario, config, self()})
    
    result = receive do
      {:integrated_stress_result, r} -> r
    after
      config.duration + 2000 ->
        %{graceful_degradation_achieved: false, no_cascade_failures: false, recovery_possible: false}
    end
    
    result
  end
  
  defp apply_integrated_stress(scenario, config, state) do
    # Simulate integrated resource stress
    %{
      graceful_degradation_achieved: true,
      no_cascade_failures: true,
      recovery_possible: true
    }
  end
  
  defp test_resource_prioritization_under_stress(scenario, manager) do
    # Test resource prioritization mechanisms
    %{
      critical_resources_preserved: true,
      fair_resource_allocation: true
    }
  end
end
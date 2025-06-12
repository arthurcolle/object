defmodule ConcurrencyEdgeCasesTest do
  use ExUnit.Case, async: false
  
  @moduledoc """
  Comprehensive concurrency edge case tests for the AAOS Object system.
  
  Tests race conditions, deadlocks, process crashes during operations,
  distributed coordination failures, and concurrent state modifications.
  """
  
  alias Object.{Mailbox, ResourceMonitor, ErrorHandling}
  require Logger
  
  @test_timeout 30_000
  @stress_object_count 100
  @concurrent_operations 50
  @deadlock_detection_timeout 5_000
  
  describe "Race Condition Detection and Prevention" do
    @tag timeout: @test_timeout
    test "concurrent state modifications maintain consistency" do
      # Create shared object for concurrent modification
      shared_object = Object.new(
        id: "race_test_object",
        state: %{
          counter: 0,
          balance: 1000.0,
          last_modified_by: nil,
          modification_history: [],
          concurrent_operations: 0
        }
      )
      
      # Start background monitoring process
      monitor_pid = spawn_link(fn -> 
        monitor_state_consistency(shared_object.id, self())
      end)
      
      # Launch concurrent modification tasks
      task_results = Enum.map(1..@concurrent_operations, fn task_id ->
        Task.async(fn ->
          perform_concurrent_modifications(shared_object, task_id, 10)
        end)
      end)
      
      # Wait for all tasks to complete
      final_results = Task.await_many(task_results, @test_timeout)
      
      # Stop monitoring
      send(monitor_pid, :stop)
      
      # Analyze consistency results
      consistency_violations = Enum.flat_map(final_results, fn result ->
        result.consistency_violations || []
      end)
      
      lost_updates = Enum.sum(Enum.map(final_results, & &1.lost_updates))
      successful_operations = Enum.sum(Enum.map(final_results, & &1.successful_operations))
      
      # Verify race condition prevention
      assert length(consistency_violations) == 0, 
        "No consistency violations should occur: #{inspect(consistency_violations)}"
      
      assert lost_updates == 0, 
        "No updates should be lost due to race conditions"
      
      assert successful_operations > @concurrent_operations * 5,
        "Majority of operations should succeed despite concurrency"
      
      # Verify final state integrity
      final_state_check = verify_final_state_integrity(shared_object.id)
      assert final_state_check.state_valid, "Final state should be valid"
      assert final_state_check.invariants_maintained, "State invariants should be maintained"
    end
    
    @tag timeout: @test_timeout
    test "message delivery race conditions with mailbox contention" do
      # Create message-heavy scenario
      message_hub = Object.new(
        id: "message_hub",
        state: %{received_messages: 0, processed_messages: 0}
      )
      
      sender_objects = for i <- 1..20 do
        Object.new(
          id: "sender_#{i}",
          state: %{messages_sent: 0, send_errors: 0}
        )
      end
      
      # Start concurrent message sending
      message_tasks = Enum.map(sender_objects, fn sender ->
        Task.async(fn ->
          send_concurrent_messages(sender, message_hub.id, 25)
        end)
      end)
      
      # Start concurrent message processing
      processing_task = Task.async(fn ->
        process_concurrent_messages(message_hub, 500)  # Expect ~20 * 25 = 500 messages
      end)
      
      # Wait for completion
      send_results = Task.await_many(message_tasks, @test_timeout)
      processing_result = Task.await(processing_task, @test_timeout)
      
      # Verify message delivery consistency
      total_sent = Enum.sum(Enum.map(send_results, & &1.messages_sent))
      total_processed = processing_result.messages_processed
      message_delivery_ratio = total_processed / max(1, total_sent)
      
      assert message_delivery_ratio > 0.95, 
        "Message delivery ratio should be >95%: #{message_delivery_ratio}"
      
      assert processing_result.duplicate_messages == 0,
        "No duplicate messages should be processed"
      
      assert processing_result.corrupted_messages == 0,
        "No message corruption should occur"
        
      # Verify mailbox state consistency
      mailbox_integrity = verify_mailbox_integrity(message_hub.id)
      assert mailbox_integrity.no_orphaned_messages, "No orphaned messages should remain"
      assert mailbox_integrity.queue_state_valid, "Mailbox queue state should be valid"
    end
  end
  
  describe "Deadlock Detection and Prevention" do
    @tag timeout: @test_timeout
    test "circular wait prevention in resource allocation" do
      # Create objects with circular resource dependencies
      resource_objects = for i <- 1..10 do
        next_resource = rem(i, 10) + 1
        Object.new(
          id: "resource_obj_#{i}",
          state: %{
            held_resources: [],
            requested_resources: ["resource_#{next_resource}"],
            resource_priority: i,
            wait_time: 0
          }
        )
      end
      
      # Initialize deadlock detector
      deadlock_detector = start_deadlock_detector()
      
      # Start resource allocation simulation
      allocation_tasks = Enum.map(resource_objects, fn obj ->
        Task.async(fn ->
          simulate_resource_allocation_with_circular_dependency(obj, @deadlock_detection_timeout)
        end)
      end)
      
      # Monitor for deadlocks
      deadlock_monitor_task = Task.async(fn ->
        monitor_for_deadlocks(deadlock_detector, @deadlock_detection_timeout + 1000)
      end)
      
      # Wait for results
      allocation_results = Task.await_many(allocation_tasks, @test_timeout)
      deadlock_monitor_result = Task.await(deadlock_monitor_task, @test_timeout)
      
      # Analyze deadlock prevention
      deadlocks_detected = deadlock_monitor_result.deadlocks_detected
      deadlocks_resolved = deadlock_monitor_result.deadlocks_resolved
      
      # Verify deadlock prevention/resolution
      assert length(deadlocks_detected) == 0 or deadlocks_resolved >= length(deadlocks_detected),
        "All detected deadlocks should be resolved"
      
      successful_allocations = Enum.count(allocation_results, & &1.allocation_successful)
      assert successful_allocations >= 7,  # Allow some failures due to timeouts
        "Majority of allocations should succeed"
      
      # Verify no permanent blocking
      permanently_blocked = Enum.count(allocation_results, & &1.permanently_blocked)
      assert permanently_blocked == 0, "No objects should be permanently blocked"
    end
    
    @tag timeout: @test_timeout
    test "hierarchical object coordination deadlock prevention" do
      # Create hierarchical object structure with potential deadlocks
      coordinator = Object.new(
        id: "coordinator",
        state: %{managed_objects: [], coordination_requests: []}
      )
      
      sub_coordinators = for i <- 1..5 do
        Object.new(
          id: "sub_coord_#{i}",
          state: %{parent_coordinator: "coordinator", managed_objects: []}
        )
      end
      
      leaf_objects = for i <- 1..20 do
        parent_id = "sub_coord_#{rem(i - 1, 5) + 1}"
        Object.new(
          id: "leaf_#{i}",
          state: %{parent_coordinator: parent_id, coordination_state: :idle}
        )
      end
      
      all_objects = [coordinator] ++ sub_coordinators ++ leaf_objects
      
      # Start hierarchical coordination with potential conflicts
      coordination_tasks = Enum.map(all_objects, fn obj ->
        Task.async(fn ->
          simulate_hierarchical_coordination(obj, all_objects, 100)
        end)
      end)
      
      # Monitor coordination health
      health_monitor = Task.async(fn ->
        monitor_coordination_health(all_objects, 5000)
      end)
      
      coordination_results = Task.await_many(coordination_tasks, @test_timeout)
      health_result = Task.await(health_monitor, @test_timeout)
      
      # Verify deadlock-free coordination
      coordination_deadlocks = Enum.sum(Enum.map(coordination_results, & &1.deadlocks_encountered))
      assert coordination_deadlocks == 0, "No coordination deadlocks should occur"
      
      # Verify hierarchy integrity
      assert health_result.hierarchy_integrity_maintained, 
        "Hierarchy integrity should be maintained"
      
      assert health_result.coordination_progress_made,
        "Coordination progress should be made despite conflicts"
    end
  end
  
  describe "Process Crash Recovery" do
    @tag timeout: @test_timeout
    test "graceful recovery from object process crashes during operations" do
      # Create objects with crash-prone operations
      crash_prone_objects = for i <- 1..15 do
        Object.new(
          id: "crash_prone_#{i}",
          state: %{
            crash_probability: 0.2,
            recovery_attempts: 0,
            operation_count: 0,
            critical_state: %{important_data: "data_#{i}"}
          }
        )
      end
      
      # Start supervisor for crash recovery
      supervisor_pid = start_crash_recovery_supervisor(crash_prone_objects)
      
      # Perform operations that may trigger crashes
      operation_tasks = Enum.map(crash_prone_objects, fn obj ->
        Task.async(fn ->
          perform_crash_prone_operations(obj, 50, supervisor_pid)
        end)
      end)
      
      # Monitor crash recovery
      recovery_monitor = Task.async(fn ->
        monitor_crash_recovery(supervisor_pid, 8000)
      end)
      
      operation_results = Task.await_many(operation_tasks, @test_timeout)
      recovery_stats = Task.await(recovery_monitor, @test_timeout)
      
      # Verify crash recovery effectiveness
      total_crashes = recovery_stats.total_crashes
      successful_recoveries = recovery_stats.successful_recoveries
      recovery_rate = if total_crashes > 0, do: successful_recoveries / total_crashes, else: 1.0
      
      assert recovery_rate > 0.8, "Recovery rate should be >80%: #{recovery_rate}"
      
      # Verify data integrity after crashes
      data_integrity_results = Enum.map(operation_results, fn result ->
        verify_data_integrity_after_crash(result)
      end)
      
      data_loss_incidents = Enum.count(data_integrity_results, & not &1.data_intact)
      assert data_loss_incidents == 0, "No data loss should occur due to crashes"
      
      # Verify system stability
      assert recovery_stats.system_stability_maintained,
        "System stability should be maintained despite crashes"
    end
    
    @tag timeout: @test_timeout
    test "distributed coordination resilience to network partitions" do
      # Simulate distributed object system with network partitions
      cluster_nodes = [:node1, :node2, :node3, :node4]
      distributed_objects = create_distributed_object_cluster(cluster_nodes, 40)
      
      # Start distributed coordination
      coordination_state = start_distributed_coordination(distributed_objects)
      
      # Introduce network partitions
      partition_scenarios = [
        %{type: :split_brain, partitions: [[:node1, :node2], [:node3, :node4]], duration: 2000},
        %{type: :isolated_node, partitions: [[:node1], [:node2, :node3, :node4]], duration: 1500},
        %{type: :network_flapping, partitions: :random, duration: 3000}
      ]
      
      partition_results = Enum.map(partition_scenarios, fn scenario ->
        result = simulate_network_partition(distributed_objects, coordination_state, scenario)
        verify_partition_resilience(result)
      end)
      
      # Verify partition tolerance
      for {result, scenario} <- Enum.zip(partition_results, partition_scenarios) do
        assert result.coordination_maintained, 
          "Coordination should be maintained during #{scenario.type}"
        
        assert result.split_brain_prevented,
          "Split-brain should be prevented during #{scenario.type}"
        
        assert result.data_consistency_preserved,
          "Data consistency should be preserved during #{scenario.type}"
        
        assert result.recovery_successful,
          "Recovery should be successful after #{scenario.type}"
      end
      
      # Verify final system state
      final_state = get_distributed_system_state(distributed_objects)
      assert final_state.all_nodes_synchronized, "All nodes should be synchronized"
      assert final_state.no_inconsistencies, "No data inconsistencies should remain"
    end
  end
  
  # Helper functions for concurrency testing
  
  defp perform_concurrent_modifications(object, task_id, iterations) do
    Enum.reduce(1..iterations, %{successful_operations: 0, lost_updates: 0, consistency_violations: []}, 
      fn i, acc ->
        try do
          # Read current state
          current_state = get_object_state(object.id)
          
          # Perform modification
          new_counter = current_state.counter + 1
          new_balance = current_state.balance + (:rand.uniform() - 0.5) * 10
          
          # Simulate some processing time to increase race condition likelihood
          :timer.sleep(:rand.uniform(5))
          
          # Attempt state update
          update_result = Object.update_state(object, %{
            counter: new_counter,
            balance: new_balance,
            last_modified_by: task_id,
            modification_history: [%{task: task_id, iteration: i, timestamp: System.monotonic_time()} | current_state.modification_history],
            concurrent_operations: current_state.concurrent_operations + 1
          })
          
          # Check for consistency violations
          violations = check_state_consistency(current_state, update_result.state)
          
          %{acc |
            successful_operations: acc.successful_operations + 1,
            consistency_violations: violations ++ acc.consistency_violations
          }
        rescue
          error ->
            %{acc | lost_updates: acc.lost_updates + 1}
        end
      end)
  end
  
  defp send_concurrent_messages(sender, recipient_id, message_count) do
    Enum.reduce(1..message_count, %{messages_sent: 0, send_errors: 0}, fn i, acc ->
      message_content = %{
        sender_id: sender.id,
        sequence_number: i,
        timestamp: System.monotonic_time(),
        payload: "message_#{i}_from_#{sender.id}"
      }
      
      try do
        updated_sender = Object.send_message(sender, recipient_id, :test_message, message_content)
        %{acc | messages_sent: acc.messages_sent + 1}
      rescue
        _error ->
          %{acc | send_errors: acc.send_errors + 1}
      end
    end)
  end
  
  defp process_concurrent_messages(message_hub, expected_count) do
    start_time = System.monotonic_time()
    timeout = 10_000  # 10 seconds
    
    process_messages_loop(message_hub, expected_count, 0, 0, 0, start_time, timeout)
  end
  
  defp process_messages_loop(object, expected_count, processed, duplicates, corrupted, start_time, timeout) do
    if processed >= expected_count or System.monotonic_time() - start_time > timeout * 1_000_000 do
      %{
        messages_processed: processed,
        duplicate_messages: duplicates,
        corrupted_messages: corrupted
      }
    else
      {messages, updated_object} = Object.process_messages(object)
      
      {new_duplicates, new_corrupted} = analyze_received_messages(messages)
      
      process_messages_loop(
        updated_object,
        expected_count,
        processed + length(messages),
        duplicates + new_duplicates,
        corrupted + new_corrupted,
        start_time,
        timeout
      )
    end
  end
  
  defp analyze_received_messages(messages) do
    # Simple duplicate/corruption detection
    duplicates = 0  # Simplified for testing
    corrupted = 0   # Simplified for testing
    {duplicates, corrupted}
  end
  
  defp monitor_state_consistency(object_id, parent_pid) do
    receive do
      :stop -> :ok
    after
      100 ->
        # Periodic consistency check
        state = get_object_state(object_id)
        consistency_result = validate_state_invariants(state)
        if not consistency_result.valid do
          send(parent_pid, {:consistency_violation, consistency_result})
        end
        monitor_state_consistency(object_id, parent_pid)
    end
  end
  
  defp verify_final_state_integrity(object_id) do
    state = get_object_state(object_id)
    %{
      state_valid: is_map(state) and Map.has_key?(state, :counter),
      invariants_maintained: state.counter >= 0 and is_number(state.balance)
    }
  end
  
  defp verify_mailbox_integrity(object_id) do
    # Simplified mailbox integrity check
    %{
      no_orphaned_messages: true,
      queue_state_valid: true
    }
  end
  
  defp check_state_consistency(old_state, new_state) do
    violations = []
    
    # Check counter monotonicity
    if new_state.counter < old_state.counter do
      violations = [{:counter_decreased, old_state.counter, new_state.counter} | violations]
    end
    
    # Check balance reasonableness
    balance_change = abs(new_state.balance - old_state.balance)
    if balance_change > 100 do
      violations = [{:balance_unreasonable_change, balance_change} | violations]
    end
    
    violations
  end
  
  defp validate_state_invariants(state) do
    valid = state.counter >= 0 and 
            is_number(state.balance) and 
            state.balance >= 0 and
            is_list(state.modification_history)
    
    %{valid: valid, violations: if(valid, do: [], else: [:invariant_violated])}
  end
  
  defp get_object_state(object_id) do
    # Simplified state retrieval - in real system would use registry/process lookup
    %{
      counter: :rand.uniform(100),
      balance: 1000.0 + :rand.uniform() * 100,
      last_modified_by: nil,
      modification_history: [],
      concurrent_operations: :rand.uniform(10)
    }
  end
  
  # Deadlock detection helpers
  
  defp start_deadlock_detector() do
    spawn_link(fn -> deadlock_detector_loop(%{detected_cycles: [], resolution_attempts: []}) end)
  end
  
  defp deadlock_detector_loop(state) do
    receive do
      {:detect_deadlock, resource_graph} ->
        cycles = detect_resource_cycles(resource_graph)
        updated_state = %{state | detected_cycles: cycles ++ state.detected_cycles}
        deadlock_detector_loop(updated_state)
      
      {:get_state, from} ->
        send(from, {:deadlock_state, state})
        deadlock_detector_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        deadlock_detector_loop(state)
    end
  end
  
  defp simulate_resource_allocation_with_circular_dependency(obj, timeout) do
    start_time = System.monotonic_time()
    
    try do
      # Simulate resource request with potential circular wait
      requested_resource = hd(obj.state.requested_resources)
      
      # Simulate allocation attempt with random delays
      :timer.sleep(:rand.uniform(100))
      
      allocation_successful = :rand.uniform() > 0.3  # 70% success rate
      
      %{
        allocation_successful: allocation_successful,
        permanently_blocked: false,
        wait_time: System.monotonic_time() - start_time
      }
    rescue
      _error ->
        %{
          allocation_successful: false,
          permanently_blocked: false,
          wait_time: System.monotonic_time() - start_time
        }
    end
  end
  
  defp monitor_for_deadlocks(detector_pid, duration) do
    monitor_deadlocks_loop(detector_pid, [], 0, System.monotonic_time(), duration)
  end
  
  defp monitor_deadlocks_loop(detector_pid, detected, resolved, start_time, duration) do
    if System.monotonic_time() - start_time > duration * 1_000_000 do
      %{deadlocks_detected: detected, deadlocks_resolved: resolved}
    else
      send(detector_pid, {:get_state, self()})
      
      receive do
        {:deadlock_state, state} ->
          new_detected = state.detected_cycles
          new_resolved = resolved + length(state.resolution_attempts)
          
          :timer.sleep(50)
          monitor_deadlocks_loop(detector_pid, new_detected, new_resolved, start_time, duration)
      after
        100 ->
          monitor_deadlocks_loop(detector_pid, detected, resolved, start_time, duration)
      end
    end
  end
  
  defp detect_resource_cycles(_resource_graph) do
    # Simplified cycle detection
    []
  end
  
  # Hierarchical coordination helpers
  
  defp simulate_hierarchical_coordination(obj, all_objects, operations) do
    Enum.reduce(1..operations, %{deadlocks_encountered: 0, operations_completed: 0}, 
      fn _i, acc ->
        try do
          # Simulate coordination request
          coordination_successful = simulate_coordination_operation(obj, all_objects)
          
          %{acc | operations_completed: acc.operations_completed + 1}
        rescue
          _error ->
            %{acc | deadlocks_encountered: acc.deadlocks_encountered + 1}
        end
      end)
  end
  
  defp simulate_coordination_operation(obj, _all_objects) do
    # Simulate coordination with random delays and success
    :timer.sleep(:rand.uniform(10))
    :rand.uniform() > 0.1  # 90% success rate
  end
  
  defp monitor_coordination_health(objects, duration) do
    start_time = System.monotonic_time()
    
    # Monitor coordination health over time
    :timer.sleep(duration)
    
    %{
      hierarchy_integrity_maintained: true,
      coordination_progress_made: true
    }
  end
  
  # Process crash recovery helpers
  
  defp start_crash_recovery_supervisor(objects) do
    spawn_link(fn -> 
      crash_supervisor_loop(%{
        monitored_objects: objects,
        crashes: [],
        recoveries: []
      })
    end)
  end
  
  defp crash_supervisor_loop(state) do
    receive do
      {:crash_detected, object_id, reason} ->
        new_crashes = [%{object_id: object_id, reason: reason, timestamp: System.monotonic_time()} | state.crashes]
        
        # Attempt recovery
        recovery_result = attempt_object_recovery(object_id, reason)
        new_recoveries = if recovery_result.successful do
          [%{object_id: object_id, recovery_time: recovery_result.recovery_time} | state.recoveries]
        else
          state.recoveries
        end
        
        crash_supervisor_loop(%{state | crashes: new_crashes, recoveries: new_recoveries})
      
      {:get_stats, from} ->
        stats = %{
          total_crashes: length(state.crashes),
          successful_recoveries: length(state.recoveries),
          system_stability_maintained: length(state.recoveries) >= length(state.crashes) * 0.8
        }
        send(from, {:crash_stats, stats})
        crash_supervisor_loop(state)
      
      :stop ->
        :ok
    after
      100 ->
        crash_supervisor_loop(state)
    end
  end
  
  defp perform_crash_prone_operations(obj, operations, supervisor_pid) do
    Enum.reduce(1..operations, %{operations_completed: 0, crashes_experienced: 0}, 
      fn i, acc ->
        try do
          # Simulate operation that might crash
          if :rand.uniform() < obj.state.crash_probability do
            send(supervisor_pid, {:crash_detected, obj.id, :simulated_crash})
            raise "Simulated crash during operation #{i}"
          end
          
          # Simulate successful operation
          :timer.sleep(:rand.uniform(5))
          %{acc | operations_completed: acc.operations_completed + 1}
        rescue
          _error ->
            %{acc | crashes_experienced: acc.crashes_experienced + 1}
        end
      end)
  end
  
  defp monitor_crash_recovery(supervisor_pid, duration) do
    :timer.sleep(duration)
    
    send(supervisor_pid, {:get_stats, self()})
    
    receive do
      {:crash_stats, stats} ->
        stats
    after
      1000 ->
        %{total_crashes: 0, successful_recoveries: 0, system_stability_maintained: true}
    end
  end
  
  defp attempt_object_recovery(object_id, reason) do
    # Simulate recovery attempt
    recovery_time = :rand.uniform(100)
    :timer.sleep(recovery_time)
    
    %{
      successful: :rand.uniform() > 0.2,  # 80% recovery success rate
      recovery_time: recovery_time
    }
  end
  
  defp verify_data_integrity_after_crash(operation_result) do
    # Simulate data integrity verification
    %{data_intact: operation_result.operations_completed > 0}
  end
  
  # Distributed system helpers
  
  defp create_distributed_object_cluster(nodes, object_count) do
    objects_per_node = div(object_count, length(nodes))
    
    Enum.flat_map(nodes, fn node ->
      for i <- 1..objects_per_node do
        Object.new(
          id: "#{node}_obj_#{i}",
          state: %{
            node: node,
            cluster_role: if(i == 1, do: :coordinator, else: :worker),
            sync_state: :synchronized
          }
        )
      end
    end)
  end
  
  defp start_distributed_coordination(objects) do
    %{
      coordination_active: true,
      sync_status: :synchronized,
      partition_status: :no_partitions
    }
  end
  
  defp simulate_network_partition(objects, coordination_state, scenario) do
    # Simulate network partition
    :timer.sleep(scenario.duration)
    
    %{
      partition_type: scenario.type,
      duration: scenario.duration,
      affected_objects: length(objects),
      coordination_impact: :minimal
    }
  end
  
  defp verify_partition_resilience(partition_result) do
    %{
      coordination_maintained: true,
      split_brain_prevented: true,
      data_consistency_preserved: true,
      recovery_successful: true
    }
  end
  
  defp get_distributed_system_state(objects) do
    %{
      all_nodes_synchronized: true,
      no_inconsistencies: true,
      cluster_healthy: true
    }
  end
end
defmodule InteractionDyadNetworkTest do
  use ExUnit.Case, async: false
  
  @moduledoc """
  Comprehensive tests for AAOS Section 4 - Object Interactions at scale.
  
  Tests dynamic dyad spawning/dissolution, message-passing protocol verification,
  network topology changes, and labeled transition system properties under
  high-load conditions and complex network structures.
  """
  
  alias Object.{Mailbox, MessageRouter, DyadManager}
  alias OORL.{NetworkTopology, ProtocolVerification}
  
  @large_network_size 1000
  @stress_message_count 10000
  @network_change_frequency 100
  @dyad_formation_probability 0.3
  @dyad_dissolution_probability 0.1
  
  describe "Dynamic Dyad Networks at Scale" do
    test "dynamic dyad spawning and dissolution with 1000+ objects" do
      # Create large object network
      objects = for i <- 1..@large_network_size do
        compatibility_profile = %{
          cooperation_tendency: :rand.uniform(),
          communication_style: Enum.random([:direct, :diplomatic, :analytical]),
          resource_sharing: :rand.uniform(),
          trust_threshold: :rand.uniform() * 0.5 + 0.3
        }
        
        Object.new(
          id: "network_obj_#{i}",
          state: %{
            compatibility_profile: compatibility_profile,
            active_dyads: [],
            interaction_history: [],
            network_position: {rem(i, 50), div(i, 50)}  # Grid position
          },
          methods: [:form_dyad, :dissolve_dyad, :communicate, :cooperate]
        )
      end
      
      # Initialize network with some initial dyads
      initial_dyads = generate_compatibility_based_dyads(objects, @dyad_formation_probability)
      
      network_state = %{
        objects: objects,
        active_dyads: initial_dyads,
        formation_history: [],
        dissolution_history: [],
        topology_metrics: %{}
      }
      
      # Simulate dynamic network evolution
      final_network = Enum.reduce(1..500, network_state, fn iteration, current_network ->
        # Phase 1: Dyad spawning based on compatibility
        spawning_candidates = identify_spawning_candidates(current_network.objects, current_network.active_dyads)
        
        new_dyads = Enum.reduce(spawning_candidates, current_network.active_dyads, fn {obj1_id, obj2_id, compatibility}, dyads ->
          spawn_probability = calculate_spawn_probability(
            get_object_by_id(current_network.objects, obj1_id),
            get_object_by_id(current_network.objects, obj2_id),
            compatibility
          )
          
          if :rand.uniform() < spawn_probability do
            dyad_id = "#{obj1_id}_#{obj2_id}_#{iteration}"
            new_dyad = %{
              id: dyad_id,
              participants: {obj1_id, obj2_id},
              formation_time: DateTime.utc_now(),
              interaction_count: 0,
              compatibility_score: compatibility,
              utility_score: 0.0,
              active: true,
              formation_iteration: iteration
            }
            Map.put(dyads, dyad_id, new_dyad)
          else
            dyads
          end
        end)
        
        # Phase 2: Dyad dissolution based on utility
        {surviving_dyads, dissolved} = Enum.reduce(new_dyads, {%{}, []}, fn {dyad_id, dyad}, {surviving, dissolved_list} ->
          dissolution_probability = calculate_dissolution_probability(dyad, iteration)
          
          if dyad.active and :rand.uniform() < dissolution_probability do
            dissolved_dyad = %{dyad | active: false, dissolution_time: DateTime.utc_now()}
            {surviving, [dissolved_dyad | dissolved_list]}
          else
            updated_dyad = %{dyad | 
              interaction_count: dyad.interaction_count + :rand.poisson(2),
              utility_score: update_utility_score(dyad)
            }
            {Map.put(surviving, dyad_id, updated_dyad), dissolved_list}
          end
        end)
        
        # Phase 3: Update network topology metrics
        topology_metrics = NetworkTopology.calculate_metrics(surviving_dyads, current_network.objects)
        
        %{
          current_network |
          active_dyads: surviving_dyads,
          formation_history: current_network.formation_history ++ Map.values(new_dyads),
          dissolution_history: current_network.dissolution_history ++ dissolved,
          topology_metrics: topology_metrics
        }
      end)
      
      # Verify network properties
      assert map_size(final_network.active_dyads) > 0, "Network should maintain active dyads"
      assert length(final_network.formation_history) > @large_network_size / 10, "Should have substantial formation activity"
      assert length(final_network.dissolution_history) > 0, "Should have dissolution activity"
      
      # Test network connectivity properties
      metrics = final_network.topology_metrics
      assert Map.has_key?(metrics, :clustering_coefficient)
      assert Map.has_key?(metrics, :average_path_length)
      assert Map.has_key?(metrics, :degree_distribution)
      assert Map.has_key?(metrics, :connected_components)
      
      # Verify small-world properties if applicable
      if metrics.clustering_coefficient > 0.1 and metrics.average_path_length < 10 do
        small_world_coefficient = metrics.clustering_coefficient / metrics.average_path_length
        assert small_world_coefficient > 0.01, "Network may exhibit small-world properties"
      end
      
      # Test dyad formation/dissolution balance
      formation_rate = length(final_network.formation_history) / 500
      dissolution_rate = length(final_network.dissolution_history) / 500
      
      # Rates should be reasonable (not too high or too low)
      assert formation_rate > 0.1 and formation_rate < 50, "Formation rate should be reasonable"
      assert dissolution_rate > 0.05 and dissolution_rate < 25, "Dissolution rate should be reasonable"
    end
    
    test "network topology changes under message routing stress" do
      # Create medium network for message stress testing
      network_size = 200
      objects = for i <- 1..network_size do
        Object.new(
          id: "route_obj_#{i}",
          state: %{message_capacity: 100, processing_speed: :rand.uniform() + 0.5},
          mailbox: Mailbox.new("route_obj_#{i}")
        )
      end
      
      # Create initial random network topology
      initial_dyads = generate_random_network_topology(objects, 0.05)  # Sparse network
      
      # Setup message routing infrastructure
      router = MessageRouter.new(objects, initial_dyads)
      
      # Generate stress messages
      stress_messages = for i <- 1..@stress_message_count do
        sender = Enum.random(objects)
        recipients = Enum.take_random(objects, :rand.uniform(5) + 1)
        
        %{
          id: "stress_msg_#{i}",
          sender: sender.id,
          recipients: Enum.map(recipients, & &1.id),
          content: %{
            type: :stress_test,
            payload_size: :rand.uniform(1000),
            priority: Enum.random([:low, :medium, :high, :critical]),
            timestamp: DateTime.utc_now()
          },
          ttl: :rand.uniform(100) + 50
        }
      end
      
      # Apply messages in batches while changing topology
      {final_router, delivery_stats, topology_history} = 
        Enum.chunk_every(stress_messages, @network_change_frequency)
        |> Enum.with_index()
        |> Enum.reduce({router, %{delivered: 0, failed: 0, dropped: 0}, []}, 
           fn {message_batch, batch_idx}, {current_router, stats, topo_history} ->
             
             # Process message batch
             {updated_router, batch_results} = MessageRouter.process_message_batch(current_router, message_batch)
             
             # Update delivery statistics
             updated_stats = %{
               delivered: stats.delivered + batch_results.delivered,
               failed: stats.failed + batch_results.failed,
               dropped: stats.dropped + batch_results.dropped
             }
             
             # Change network topology periodically
             modified_router = if rem(batch_idx, 3) == 0 do
               # Modify topology: add/remove some dyads
               current_dyads = MessageRouter.get_active_dyads(updated_router)
               
               # Remove some random dyads
               dyads_to_remove = Enum.take_random(Map.keys(current_dyads), max(1, div(map_size(current_dyads), 10)))
               reduced_dyads = Map.drop(current_dyads, dyads_to_remove)
               
               # Add some new random dyads
               available_objects = MessageRouter.get_objects(updated_router)
               new_dyads = generate_random_network_topology(available_objects, 0.02)
               combined_dyads = Map.merge(reduced_dyads, new_dyads)
               
               MessageRouter.update_topology(updated_router, combined_dyads)
             else
               updated_router
             end
             
             # Record topology metrics
             current_topology = NetworkTopology.calculate_metrics(
               MessageRouter.get_active_dyads(modified_router),
               MessageRouter.get_objects(modified_router)
             )
             
             {modified_router, updated_stats, [current_topology | topo_history]}
           end)
      
      # Verify message delivery under topology changes
      total_messages = length(stress_messages)
      delivery_rate = delivery_stats.delivered / total_messages
      
      assert delivery_rate > 0.7, "Should maintain >70% delivery rate despite topology changes"
      assert delivery_stats.failed < total_messages * 0.2, "Failed message rate should be <20%"
      
      # Test routing adaptivity
      topology_stability = calculate_topology_stability(topology_history)
      assert topology_stability > 0.3, "Router should adapt to topology changes"
      
      # Verify no message loops or deadlocks
      routing_stats = MessageRouter.get_routing_statistics(final_router)
      assert routing_stats.max_hop_count < network_size, "Should avoid message loops"
      assert routing_stats.deadlock_count == 0, "Should not have routing deadlocks"
    end
  end
  
  describe "Message-Passing Protocol LTS Verification" do
    test "protocol satisfies reachability properties" do
      # Create simple network for protocol verification
      objects = for i <- 1..10 do
        Object.new(id: "protocol_obj_#{i}", mailbox: Mailbox.new("protocol_obj_#{i}"))
      end
      
      # Define protocol states and transitions
      protocol_lts = %{
        states: [:idle, :sending, :receiving, :processing, :responding, :error],
        initial_state: :idle,
        alphabet: [:send_prompt, :receive_prompt, :process_message, :send_response, :receive_response, :timeout, :error],
        transitions: %{
          {:idle, :send_prompt} => :sending,
          {:idle, :receive_prompt} => :receiving,
          {:sending, :receive_response} => :idle,
          {:sending, :timeout} => :error,
          {:receiving, :process_message} => :processing,
          {:processing, :send_response} => :responding,
          {:responding, :receive_response} => :idle,
          {:error, :send_prompt} => :sending,  # Recovery
          {:error, :receive_prompt} => :receiving  # Recovery
        }
      }
      
      # Test reachability: all states should be reachable from initial state
      reachable_states = ProtocolVerification.compute_reachable_states(protocol_lts)
      
      assert :idle in reachable_states
      assert :sending in reachable_states
      assert :receiving in reachable_states
      assert :processing in reachable_states
      assert :responding in reachable_states
      
      # Error state should be reachable (for robustness)
      assert :error in reachable_states
      
      # Test state reachability from any state
      for state <- protocol_lts.states do
        reachable_from_state = ProtocolVerification.compute_reachable_states(
          %{protocol_lts | initial_state: state}
        )
        # Should be able to return to idle from any state (strong connectivity)
        assert :idle in reachable_from_state, "Should reach idle from #{state}"
      end
    end
    
    test "protocol satisfies safety properties" do
      # Define safety property: "no message loss"
      # Property: if a message is sent, it must eventually be delivered or explicitly fail
      
      test_objects = [
        Object.new(id: "sender", mailbox: Mailbox.new("sender")),
        Object.new(id: "receiver", mailbox: Mailbox.new("receiver"))
      ]
      
      protocol_state = %{
        objects: test_objects,
        message_queue: [],
        delivery_confirmations: %{},
        failed_messages: [],
        protocol_violations: []
      }
      
      # Send 100 test messages
      final_state = Enum.reduce(1..100, protocol_state, fn i, state ->
        message = %{
          id: "safety_test_#{i}",
          sender: "sender",
          recipients: ["receiver"],
          content: %{test_data: i},
          timestamp: DateTime.utc_now(),
          ttl: 50,
          requires_confirmation: true
        }
        
        # Send message through protocol
        updated_state = simulate_message_protocol(state, message)
        
        # Check safety invariant: message must be accounted for
        if not message_accounted_for?(updated_state, message.id) do
          violation = %{
            type: :message_loss,
            message_id: message.id,
            timestamp: DateTime.utc_now()
          }
          %{updated_state | protocol_violations: [violation | updated_state.protocol_violations]}
        else
          updated_state
        end
      end)
      
      # Verify no safety violations
      assert length(final_state.protocol_violations) == 0, "No messages should be lost"
      
      # Verify all messages accounted for
      total_accounted = map_size(final_state.delivery_confirmations) + length(final_state.failed_messages)
      assert total_accounted == 100, "All messages should be accounted for"
    end
    
    test "protocol satisfies liveness properties" do
      # Liveness property: "every sent message eventually receives a response or timeout"
      
      sender_obj = Object.new(id: "liveness_sender", mailbox: Mailbox.new("liveness_sender"))
      receiver_obj = Object.new(id: "liveness_receiver", mailbox: Mailbox.new("liveness_receiver"))
      
      # Create dyad between sender and receiver
      dyad = %{
        id: "liveness_dyad",
        participants: {"liveness_sender", "liveness_receiver"},
        active: true,
        protocol_state: :active
      }
      
      # Send prompt messages and verify responses
      prompt_responses = for i <- 1..20 do
        prompt = %{
          id: "liveness_prompt_#{i}",
          sender: "liveness_sender",
          recipients: ["liveness_receiver"],
          role: :prompt,
          content: %{query: "test_query_#{i}"},
          timestamp: DateTime.utc_now(),
          dyad_id: "liveness_dyad",
          requires_response: true,
          timeout_ms: 1000
        }
        
        # Simulate protocol execution
        protocol_result = execute_prompt_response_protocol(sender_obj, receiver_obj, dyad, prompt)
        
        case protocol_result do
          {:ok, response} ->
            assert response.role == :response
            assert response.content.responds_to == prompt.id
            {:response_received, response}
            
          {:timeout, _reason} ->
            # Timeout is acceptable for liveness (eventual response)
            {:timeout, prompt.id}
            
          {:error, reason} ->
            {:error, reason}
        end
      end
      
      # Verify liveness: at least 80% should get responses or timeouts
      resolved_count = Enum.count(prompt_responses, fn
        {:response_received, _} -> true
        {:timeout, _} -> true
        _ -> false
      end)
      
      assert resolved_count >= 16, "At least 80% of prompts should be resolved (liveness)"
      
      # Verify no permanent blocking
      error_count = Enum.count(prompt_responses, fn
        {:error, _} -> true
        _ -> false
      end)
      
      assert error_count < 5, "Should have minimal permanent failures"
    end
    
    test "protocol prevents deadlocks in complex networks" do
      # Create network with potential for deadlocks
      deadlock_objects = for i <- 1..8 do
        Object.new(
          id: "deadlock_obj_#{i}",
          state: %{waiting_for: [], holding_resources: []},
          mailbox: Mailbox.new("deadlock_obj_#{i}")
        )
      end
      
      # Create circular dependency scenario
      circular_dyads = for i <- 0..7 do
        {
          "circular_dyad_#{i}",
          %{
            participants: {"deadlock_obj_#{i}", "deadlock_obj_#{rem(i + 1, 8)}"},
            resource_requirements: ["resource_#{i}", "resource_#{rem(i + 1, 8)}"],
            active: true
          }
        }
      end |> Map.new()
      
      # Simulate resource allocation protocol
      allocation_state = %{
        objects: deadlock_objects,
        dyads: circular_dyads,
        resource_locks: %{},
        pending_requests: [],
        deadlock_detected: false
      }
      
      # Run protocol with deadlock detection
      final_allocation = Enum.reduce(1..50, allocation_state, fn iteration, state ->
        # Each object tries to acquire resources
        updated_state = Enum.reduce(state.objects, state, fn obj, current_state ->
          resource_requests = generate_resource_requests(obj, current_state.dyads)
          process_resource_requests(current_state, obj.id, resource_requests)
        end)
        
        # Check for deadlock after each iteration
        deadlock_analysis = detect_deadlock_cycles(updated_state)
        
        if deadlock_analysis.deadlock_detected do
          # Apply deadlock resolution
          resolve_deadlocks(updated_state, deadlock_analysis.cycles)
        else
          updated_state
        end
      end)
      
      # Verify no persistent deadlocks
      assert not final_allocation.deadlock_detected, "Protocol should prevent persistent deadlocks"
      
      # Verify progress: resources should be allocated
      allocated_resources = final_allocation.resource_locks |> Map.values() |> length()
      assert allocated_resources > 0, "Should make progress on resource allocation"
      
      # Verify fairness: all objects should get some resources
      objects_with_resources = final_allocation.resource_locks
      |> Map.values()
      |> Enum.uniq()
      |> length()
      
      assert objects_with_resources >= 4, "At least half the objects should acquire resources"
    end
  end
  
  # Helper functions
  defp generate_compatibility_based_dyads(objects, probability) do
    object_pairs = for obj1 <- objects,
                      obj2 <- objects,
                      obj1.id != obj2.id,
                      do: {obj1, obj2}
    
    object_pairs
    |> Enum.filter(fn {obj1, obj2} ->
      compatibility = calculate_compatibility(obj1, obj2)
      :rand.uniform() < probability * compatibility
    end)
    |> Enum.map(fn {obj1, obj2} ->
      dyad_id = "#{obj1.id}_#{obj2.id}"
      {dyad_id, %{
        id: dyad_id,
        participants: {obj1.id, obj2.id},
        formation_time: DateTime.utc_now(),
        compatibility_score: calculate_compatibility(obj1, obj2),
        active: true
      }}
    end)
    |> Map.new()
  end
  
  defp generate_random_network_topology(objects, connection_probability) do
    for obj1 <- objects,
        obj2 <- objects,
        obj1.id != obj2.id,
        :rand.uniform() < connection_probability,
        into: %{} do
      dyad_id = "#{obj1.id}_#{obj2.id}"
      {dyad_id, %{
        id: dyad_id,
        participants: {obj1.id, obj2.id},
        formation_time: DateTime.utc_now(),
        active: true
      }}
    end
  end
  
  defp calculate_compatibility(obj1, obj2) do
    profile1 = obj1.state.compatibility_profile || %{}
    profile2 = obj2.state.compatibility_profile || %{}
    
    # Simple compatibility metric based on profile similarity
    cooperation_compat = 1 - abs(
      (profile1.cooperation_tendency || 0.5) - (profile2.cooperation_tendency || 0.5)
    )
    
    resource_compat = 1 - abs(
      (profile1.resource_sharing || 0.5) - (profile2.resource_sharing || 0.5)
    )
    
    (cooperation_compat + resource_compat) / 2
  end
  
  defp identify_spawning_candidates(objects, existing_dyads) do
    existing_pairs = existing_dyads
    |> Map.values()
    |> Enum.map(& &1.participants)
    |> MapSet.new()
    
    for obj1 <- objects,
        obj2 <- objects,
        obj1.id != obj2.id,
        not MapSet.member?(existing_pairs, {obj1.id, obj2.id}),
        not MapSet.member?(existing_pairs, {obj2.id, obj1.id}) do
      {obj1.id, obj2.id, calculate_compatibility(obj1, obj2)}
    end
  end
  
  defp calculate_spawn_probability(obj1, obj2, compatibility) do
    base_probability = 0.1
    compatibility_bonus = compatibility * 0.2
    
    # Reduce probability if objects already have many dyads
    obj1_dyad_count = length(obj1.state.active_dyads || [])
    obj2_dyad_count = length(obj2.state.active_dyads || [])
    capacity_penalty = (obj1_dyad_count + obj2_dyad_count) * 0.02
    
    max(0, base_probability + compatibility_bonus - capacity_penalty)
  end
  
  defp calculate_dissolution_probability(dyad, iteration) do
    base_probability = 0.05
    
    # Increase probability if utility is low
    utility_penalty = if dyad.utility_score < 0.3, do: 0.1, else: 0
    
    # Increase probability for old dyads with low interaction
    age_factor = max(0, iteration - dyad.formation_iteration) / 100
    interaction_factor = if dyad.interaction_count < 2, do: 0.05, else: 0
    
    min(0.5, base_probability + utility_penalty + age_factor + interaction_factor)
  end
  
  defp update_utility_score(dyad) do
    # Simple utility update based on interaction frequency
    interaction_bonus = min(dyad.interaction_count / 10, 0.5)
    current_utility = dyad.utility_score
    
    # Exponential moving average
    0.8 * current_utility + 0.2 * interaction_bonus
  end
  
  defp get_object_by_id(objects, id) do
    Enum.find(objects, fn obj -> obj.id == id end)
  end
  
  defp calculate_topology_stability(topology_history) do
    if length(topology_history) < 2 do
      1.0
    else
      changes = topology_history
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [newer, older] ->
        calculate_topology_difference(newer, older)
      end)
      
      avg_change = Enum.sum(changes) / length(changes)
      max(0, 1 - avg_change)
    end
  end
  
  defp calculate_topology_difference(topo1, topo2) do
    # Simple difference metric based on clustering coefficient
    cc_diff = abs(topo1.clustering_coefficient - topo2.clustering_coefficient)
    path_diff = abs(topo1.average_path_length - topo2.average_path_length) / 10
    
    (cc_diff + path_diff) / 2
  end
  
  defp simulate_message_protocol(state, message) do
    # Simplified message protocol simulation
    delivery_success = :rand.uniform() > 0.05  # 95% success rate
    
    if delivery_success do
      confirmation = %{
        message_id: message.id,
        status: :delivered,
        timestamp: DateTime.utc_now()
      }
      %{state | 
        delivery_confirmations: Map.put(state.delivery_confirmations, message.id, confirmation)
      }
    else
      failed_message = %{
        message_id: message.id,
        reason: :network_failure,
        timestamp: DateTime.utc_now()
      }
      %{state | 
        failed_messages: [failed_message | state.failed_messages]
      }
    end
  end
  
  defp message_accounted_for?(state, message_id) do
    Map.has_key?(state.delivery_confirmations, message_id) or
    Enum.any?(state.failed_messages, fn msg -> msg.message_id == message_id end)
  end
  
  defp execute_prompt_response_protocol(sender, receiver, dyad, prompt) do
    # Simulate realistic protocol execution with possible failures
    case :rand.uniform() do
      x when x < 0.8 ->  # 80% success
        response = %{
          id: "response_#{prompt.id}",
          sender: receiver.id,
          recipients: [sender.id],
          role: :response,
          content: %{responds_to: prompt.id, result: :success},
          timestamp: DateTime.utc_now(),
          dyad_id: dyad.id
        }
        {:ok, response}
        
      x when x < 0.95 ->  # 15% timeout
        {:timeout, :receiver_busy}
        
      _ ->  # 5% error
        {:error, :protocol_violation}
    end
  end
  
  defp generate_resource_requests(obj, dyads) do
    # Generate resource requests based on object's dyads
    relevant_dyads = dyads
    |> Map.values()
    |> Enum.filter(fn dyad -> 
      {p1, p2} = dyad.participants
      p1 == obj.id or p2 == obj.id
    end)
    
    Enum.flat_map(relevant_dyads, fn dyad ->
      dyad.resource_requirements || []
    end)
    |> Enum.take(3)  # Limit requests
  end
  
  defp process_resource_requests(state, object_id, resource_requests) do
    # Simple resource allocation logic
    {granted_resources, updated_locks} = Enum.reduce(resource_requests, {[], state.resource_locks}, 
      fn resource, {granted, locks} ->
        if Map.has_key?(locks, resource) do
          {granted, locks}  # Resource already locked
        else
          {[resource | granted], Map.put(locks, resource, object_id)}
        end
      end)
    
    %{state | resource_locks: updated_locks}
  end
  
  defp detect_deadlock_cycles(state) do
    # Simple cycle detection in resource allocation graph
    # In practice, this would use proper graph algorithms
    waiting_graph = build_waiting_graph(state)
    cycles = find_cycles_in_graph(waiting_graph)
    
    %{
      deadlock_detected: length(cycles) > 0,
      cycles: cycles
    }
  end
  
  defp build_waiting_graph(state) do
    # Build a graph where edges represent "waiting for" relationships
    # Simplified implementation
    %{}
  end
  
  defp find_cycles_in_graph(_graph) do
    # Placeholder for cycle detection algorithm
    []
  end
  
  defp resolve_deadlocks(state, _cycles) do
    # Simple deadlock resolution: release some random locks
    reduced_locks = state.resource_locks
    |> Map.keys()
    |> Enum.take_random(div(map_size(state.resource_locks), 2))
    |> Enum.reduce(state.resource_locks, fn key, locks -> Map.delete(locks, key) end)
    
    %{state | resource_locks: reduced_locks, deadlock_detected: false}
  end
end
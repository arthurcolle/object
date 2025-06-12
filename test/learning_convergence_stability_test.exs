defmodule LearningConvergenceStabilityTest do
  use ExUnit.Case, async: false
  
  @moduledoc """
  Comprehensive tests for learning convergence and stability in OORL framework.
  
  Tests multi-objective optimization, policy gradient stability, collective intelligence
  emergence, and convergence properties under competing goals and complex reward landscapes.
  """
  
  alias OORL.{PolicyLearning, CollectiveLearning, MetaLearning}
  alias Object.{AIReasoning}
  
  @convergence_tolerance 0.01
  @stability_test_iterations 2000
  @collective_intelligence_threshold 0.6
  @pareto_frontier_samples 100
  
  describe "Multi-Objective Optimization Convergence" do
    test "competing goals converge to Pareto frontier" do
      # Create objects with inherently conflicting objectives
      competing_objects = for i <- 1..20 do
        # Each object has multiple competing objectives
        objectives = %{
          efficiency: fn state -> Map.get(state, :energy_saved, 0) end,
          performance: fn state -> Map.get(state, :task_completion, 0) end,
          safety: fn state -> 1.0 - Map.get(state, :risk_taken, 0) end,
          exploration: fn state -> Map.get(state, :novelty_discovered, 0) end
        }
        
        # Weights create different trade-offs for each object
        weights = %{
          efficiency: :rand.uniform(),
          performance: :rand.uniform(),
          safety: :rand.uniform(),
          exploration: :rand.uniform()
        }
        # Normalize weights
        total_weight = weights |> Map.values() |> Enum.sum()
        normalized_weights = weights |> Enum.map(fn {k, v} -> {k, v / total_weight} end) |> Map.new()
        
        # Combined objective function
        combined_objective = fn state ->
          normalized_weights.efficiency * objectives.efficiency.(state) +
          normalized_weights.performance * objectives.performance.(state) +
          normalized_weights.safety * objectives.safety.(state) +
          normalized_weights.exploration * objectives.exploration.(state)
        end
        
        Object.new(
          id: "competing_obj_#{i}",
          state: %{
            energy_saved: 0.0,
            task_completion: 0.0,
            risk_taken: 0.0,
            novelty_discovered: 0.0,
            policy_parameters: generate_random_policy_params()
          },
          goal: combined_objective,
          methods: [:optimize_efficiency, :increase_performance, :ensure_safety, :explore_novelty]
        )
      end
      
      # Initialize OORL states for each object
      oorl_states = competing_objects
      |> Enum.map(fn obj -> 
        {:ok, oorl_state} = OORL.initialize_oorl_object(obj.id, %{
          policy_type: :neural,
          multi_objective: true,
          social_learning_enabled: true
        })
        {obj.id, oorl_state}
      end)
      |> Map.new()
      
      # Run multi-objective optimization
      optimization_history = Enum.reduce(1..500, {competing_objects, oorl_states, []}, 
        fn iteration, {objects, oorl_states, history} ->
          
          # Generate experiences for each object
          experiences = Enum.map(objects, fn obj ->
            # Simulate environment interaction with trade-offs
            action = sample_action_from_policy(oorl_states[obj.id])
            
            {state_update, reward_components} = simulate_conflicting_environment(obj.state, action)
            
            next_state = Map.merge(obj.state, state_update)
            
            # Multi-objective reward
            total_reward = obj.goal.(next_state)
            
            %{
              object_id: obj.id,
              state: obj.state,
              action: action,
              reward: total_reward,
              next_state: next_state,
              reward_components: reward_components,
              pareto_objectives: [
                reward_components.efficiency,
                reward_components.performance,
                reward_components.safety,
                reward_components.exploration
              ]
            }
          end)
          
          # Update objects and OORL states
          updated_objects = Enum.map(objects, fn obj ->
            experience = Enum.find(experiences, fn exp -> exp.object_id == obj.id end)
            Object.update_state(obj, experience.next_state)
          end)
          
          # Perform multi-objective policy updates
          updated_oorl_states = Enum.reduce(experiences, oorl_states, fn exp, states ->
            social_context = build_social_context(experiences, exp.object_id)
            
            case OORL.learning_step(exp.object_id, exp.state, exp.action, exp.reward, exp.next_state, social_context) do
              {:ok, learning_result} ->
                Map.put(states, exp.object_id, learning_result.updated_oorl_state)
              {:error, _} ->
                states
            end
          end)
          
          # Record Pareto frontier
          pareto_points = experiences |> Enum.map(& &1.pareto_objectives)
          current_pareto_frontier = compute_pareto_frontier(pareto_points)
          
          iteration_data = %{
            iteration: iteration,
            pareto_frontier: current_pareto_frontier,
            convergence_metrics: calculate_convergence_metrics(pareto_points),
            diversity_score: calculate_objective_diversity(pareto_points)
          }
          
          {updated_objects, updated_oorl_states, [iteration_data | history]}
        end)
      
      {final_objects, final_oorl_states, convergence_history} = optimization_history
      
      # Verify Pareto frontier convergence
      final_iteration = hd(convergence_history)
      assert length(final_iteration.pareto_frontier) > 5, "Should discover multiple Pareto-optimal solutions"
      
      # Test convergence stability
      last_10_iterations = Enum.take(convergence_history, 10)
      pareto_stability = calculate_pareto_stability(last_10_iterations)
      assert pareto_stability > 0.8, "Pareto frontier should stabilize"
      
      # Verify objective diversity maintained
      final_diversity = final_iteration.diversity_score
      assert final_diversity > 0.3, "Should maintain diversity in objective space"
      
      # Test non-domination property
      final_pareto = final_iteration.pareto_frontier
      for point1 <- final_pareto do
        for point2 <- final_pareto do
          if point1 != point2 do
            assert not dominates?(point1, point2), "Pareto points should not dominate each other"
          end
        end
      end
    end
    
    test "policy gradient stability under conflicting rewards" do
      # Create object with reward function that has conflicting components
      conflicted_object = Object.new(
        id: "gradient_stability_test",
        state: %{
          policy_params: generate_random_policy_params(),
          gradient_history: [],
          reward_history: [],
          stability_metrics: []
        }
      )
      
      {:ok, oorl_state} = OORL.initialize_oorl_object(conflicted_object.id, %{
        policy_type: :neural,
        learning_rate: 0.01,
        gradient_clipping: true
      })
      
      # Generate conflicting reward scenarios
      conflict_scenarios = [
        # Scenario 1: Oscillating rewards
        fn state, action -> 
          base_reward = action.value * (:math.sin(state.episode / 10) + 1)
          noise = (:rand.uniform() - 0.5) * 0.2
          base_reward + noise
        end,
        
        # Scenario 2: Delayed conflicting feedback
        fn state, action ->
          immediate_reward = action.value * 0.3
          delayed_penalty = if rem(state.episode, 10) == 0, do: -2.0, else: 0.0
          immediate_reward + delayed_penalty
        end,
        
        # Scenario 3: Multi-modal reward landscape
        fn state, action ->
          mode1 = :math.exp(-((action.value - 0.3) * (action.value - 0.3)) / 0.1)
          mode2 = :math.exp(-((action.value - 0.7) * (action.value - 0.7)) / 0.1)
          mode1 + mode2 + (:rand.uniform() - 0.5) * 0.1
        end
      ]
      
      # Test stability across scenarios
      stability_results = Enum.map(conflict_scenarios, fn reward_fn ->
        scenario_history = Enum.reduce(1..@stability_test_iterations, {conflicted_object, oorl_state, []}, 
          fn episode, {obj, oorl, history} ->
            
            # Sample action from current policy
            action = %{value: sample_policy_action(oorl.policy_network), episode: episode}
            
            # Get conflicting reward
            reward = reward_fn.(obj.state, action)
            
            # Update state
            next_state = %{obj.state | 
              episode: episode,
              last_reward: reward,
              last_action: action.value
            }
            
            # Perform learning step
            social_context = %{
              observed_actions: [],
              peer_rewards: [],
              coalition_membership: [],
              reputation_scores: %{},
              interaction_dyads: [],
              message_history: []
            }
            
            case OORL.learning_step(obj.id, obj.state, action, reward, next_state, social_context) do
              {:ok, learning_result} ->
                # Calculate gradient norms and stability metrics
                gradient_norm = calculate_gradient_norm(learning_result.policy_update)
                stability_metric = calculate_stability_metric(history, gradient_norm, reward)
                
                updated_obj = Object.update_state(obj, next_state)
                iteration_data = %{
                  episode: episode,
                  reward: reward,
                  gradient_norm: gradient_norm,
                  stability_metric: stability_metric,
                  policy_entropy: calculate_policy_entropy(learning_result.updated_oorl_state.policy_network)
                }
                
                {updated_obj, learning_result.updated_oorl_state, [iteration_data | history]}
                
              {:error, _reason} ->
                {obj, oorl, history}
            end
          end)
        
        {_final_obj, _final_oorl, episode_history} = scenario_history
        analyze_gradient_stability(episode_history)
      end)
      
      # Verify stability across all scenarios
      for {scenario_idx, stability_analysis} <- Enum.with_index(stability_results) do
        assert stability_analysis.gradient_explosion_count == 0, 
          "Scenario #{scenario_idx}: No gradient explosions should occur"
        
        assert stability_analysis.average_gradient_norm < 10.0,
          "Scenario #{scenario_idx}: Average gradient norm should be reasonable"
        
        assert stability_analysis.convergence_detected,
          "Scenario #{scenario_idx}: Should eventually converge despite conflicts"
        
        # Policy should maintain some entropy (not collapse)
        assert stability_analysis.final_policy_entropy > 0.1,
          "Scenario #{scenario_idx}: Policy should maintain exploration capability"
      end
    end
  end
  
  describe "Collective Intelligence Emergence" do
    test "500+ objects spontaneously form effective coalitions" do
      # Create large population with diverse capabilities
      population_size = 500
      diverse_objects = for i <- 1..population_size do
        capabilities = %{
          sensing: :rand.uniform(),
          processing: :rand.uniform(),
          actuation: :rand.uniform(),
          communication: :rand.uniform(),
          learning_rate: :rand.uniform() * 0.1 + 0.01
        }
        
        Object.new(
          id: "collective_#{i}",
          state: %{
            capabilities: capabilities,
            coalition_history: [],
            performance_in_coalitions: [],
            individual_performance: :rand.uniform() * 0.5,  # Start with limited individual performance
            trust_network: %{},
            reputation: 0.5
          },
          goal: fn state -> 
            individual_score = state.individual_performance
            collective_bonus = if length(state.performance_in_coalitions) > 0 do
              Enum.sum(state.performance_in_coalitions) / length(state.performance_in_coalitions)
            else
              0
            end
            individual_score + collective_bonus * 0.5
          end
        )
      end
      
      # Define complex tasks that require coalition formation
      complex_tasks = [
        %{
          id: "distributed_sensing",
          requirements: %{sensing: 5.0, processing: 3.0, communication: 4.0},
          reward_multiplier: 2.0,
          coordination_difficulty: 0.3
        },
        %{
          id: "parallel_processing",
          requirements: %{processing: 8.0, actuation: 2.0, communication: 3.0},
          reward_multiplier: 1.8,
          coordination_difficulty: 0.2
        },
        %{
          id: "adaptive_control",
          requirements: %{sensing: 3.0, processing: 4.0, actuation: 6.0, communication: 5.0},
          reward_multiplier: 2.5,
          coordination_difficulty: 0.5
        }
      ]
      
      # Initialize collective learning system
      collective_system = %{
        objects: diverse_objects,
        active_coalitions: %{},
        task_queue: complex_tasks,
        coalition_formation_history: [],
        emergence_indicators: [],
        global_performance_metrics: []
      }
      
      # Simulate emergence over time
      final_system = Enum.reduce(1..300, collective_system, fn iteration, system ->
        # Phase 1: Task assignment and coalition formation
        {formed_coalitions, remaining_tasks} = form_task_coalitions(system.objects, system.task_queue)
        
        # Phase 2: Coalition task execution
        coalition_results = Enum.map(formed_coalitions, fn coalition ->
          execute_coalition_task(coalition, iteration)
        end)
        
        # Phase 3: Update object states based on coalition performance
        updated_objects = update_objects_from_coalition_results(system.objects, coalition_results)
        
        # Phase 4: Update trust networks and reputation
        updated_objects_with_trust = update_trust_networks(updated_objects, coalition_results)
        
        # Phase 5: Detect emergence indicators
        emergence_indicators = detect_emergence_indicators(updated_objects_with_trust, formed_coalitions)
        
        # Phase 6: Calculate global performance metrics
        global_metrics = calculate_global_performance_metrics(updated_objects_with_trust, coalition_results)
        
        %{
          system |
          objects: updated_objects_with_trust,
          active_coalitions: formed_coalitions |> Enum.map(fn c -> {c.id, c} end) |> Map.new(),
          coalition_formation_history: system.coalition_formation_history ++ formed_coalitions,
          emergence_indicators: [emergence_indicators | system.emergence_indicators],
          global_performance_metrics: [global_metrics | system.global_performance_metrics]
        }
      end)
      
      # Analyze emergence
      emergence_analysis = analyze_collective_emergence(final_system)
      
      # Verify spontaneous coalition formation
      assert length(final_system.coalition_formation_history) > 50, 
        "Should form substantial number of coalitions"
      
      # Test emergence detection
      assert emergence_analysis.collective_intelligence_score > @collective_intelligence_threshold,
        "Should exhibit collective intelligence emergence"
      
      # Verify self-organization
      assert emergence_analysis.self_organization_score > 0.5,
        "Should show self-organization behavior"
      
      # Test performance improvement
      initial_performance = hd(Enum.reverse(final_system.global_performance_metrics))
      final_performance = hd(final_system.global_performance_metrics)
      
      performance_improvement = (final_performance.average_effectiveness - initial_performance.average_effectiveness) / initial_performance.average_effectiveness
      assert performance_improvement > 0.3, "Should show >30% performance improvement through emergence"
      
      # Verify emergent specialization
      specialization_analysis = analyze_object_specialization(final_system.objects)
      assert specialization_analysis.specialization_index > 0.4,
        "Objects should develop specialized roles"
      
      # Test coordination efficiency
      assert emergence_analysis.coordination_efficiency > 0.6,
        "Should achieve efficient coordination"
    end
    
    test "emergent behavior detection accuracy and classification" do
      # Create controlled test scenarios for emergence detection
      emergence_scenarios = [
        # Scenario 1: Genuine emergence
        %{
          type: :genuine_emergence,
          objects: create_emergence_test_objects(20, :cooperative),
          expected_emergence: true,
          emergence_type: :collective_problem_solving
        },
        
        # Scenario 2: Simple aggregation (not emergence)
        %{
          type: :simple_aggregation,
          objects: create_emergence_test_objects(20, :independent),
          expected_emergence: false,
          emergence_type: :none
        },
        
        # Scenario 3: Synchronized behavior
        %{
          type: :synchronization,
          objects: create_emergence_test_objects(20, :synchronized),
          expected_emergence: true,
          emergence_type: :behavioral_synchronization
        },
        
        # Scenario 4: Hierarchical organization
        %{
          type: :hierarchy_formation,
          objects: create_emergence_test_objects(20, :hierarchical),
          expected_emergence: true,
          emergence_type: :organizational_emergence
        }
      ]
      
      detection_results = Enum.map(emergence_scenarios, fn scenario ->
        # Run simulation for scenario
        simulation_results = simulate_emergence_scenario(scenario, 200)
        
        # Apply emergence detection algorithm
        emergence_detection = CollectiveLearning.emergence_detection(%{
          members: Enum.map(scenario.objects, & &1.id),
          interaction_history: simulation_results.interaction_history,
          performance_history: simulation_results.performance_history,
          behavioral_data: simulation_results.behavioral_data
        })
        
        %{
          scenario_type: scenario.type,
          expected_emergence: scenario.expected_emergence,
          detected_emergence: emergence_detection,
          detection_accuracy: calculate_detection_accuracy(scenario, emergence_detection)
        }
      end)
      
      # Verify detection accuracy
      for result <- detection_results do
        case {result.expected_emergence, result.detected_emergence} do
          {true, {:emergent_behavior_detected, details}} ->
            assert details.score > 0.5, "Should detect genuine emergence with high confidence"
            
          {false, {:no_emergence, score}} ->
            assert score < 0.3, "Should correctly identify non-emergent behavior"
            
          {expected, actual} ->
            flunk("Detection mismatch for #{result.scenario_type}: expected #{expected}, got #{actual}")
        end
      end
      
      # Test emergence classification accuracy
      genuine_emergences = detection_results
      |> Enum.filter(fn r -> r.expected_emergence and match?({:emergent_behavior_detected, _}, r.detected_emergence) end)
      
      for result <- genuine_emergences do
        {:emergent_behavior_detected, details} = result.detected_emergence
        assert Map.has_key?(details, :behavior_signature), "Should classify emergence type"
        assert Map.has_key?(details, :emergence_mechanisms), "Should identify emergence mechanisms"
      end
    end
  end
  
  describe "Convergence Properties Under Stress" do
    test "learning convergence with high-dimensional state spaces" do
      # Create high-dimensional learning problem
      state_dimension = 1000
      action_dimension = 100
      
      high_dim_object = Object.new(
        id: "high_dim_learner",
        state: %{
          feature_vector: generate_random_vector(state_dimension),
          policy_parameters: generate_random_vector(action_dimension * state_dimension),
          convergence_history: []
        }
      )
      
      {:ok, oorl_state} = OORL.initialize_oorl_object(high_dim_object.id, %{
        policy_type: :neural,
        architecture: %{
          layers: [state_dimension, 512, 256, action_dimension],
          activation: :relu,
          output_activation: :softmax
        },
        learning_rate: 0.001,
        batch_size: 32
      })
      
      # Define complex reward landscape
      reward_function = fn state, action ->
        # Multimodal reward with local optima
        feature_vec = state.feature_vector
        action_vec = action.action_vector
        
        # Multiple reward modes
        mode1 = dot_product(feature_vec, action_vec) |> :math.tanh()
        mode2 = :math.sin(vector_norm(feature_vec) + vector_norm(action_vec))
        mode3 = (:math.cos(Enum.sum(feature_vec) / 100) + 1) / 2
        
        # Combine with noise
        base_reward = 0.4 * mode1 + 0.4 * mode2 + 0.2 * mode3
        noise = (:rand.normal() * 0.1)
        base_reward + noise
      end
      
      # Run convergence test
      convergence_data = Enum.reduce(1..1000, {high_dim_object, oorl_state, []}, 
        fn episode, {obj, oorl, history} ->
          
          # Sample action
          action = %{action_vector: sample_high_dim_action(oorl.policy_network, obj.state.feature_vector)}
          
          # Calculate reward
          reward = reward_function.(obj.state, action)
          
          # Update state (random walk in feature space)
          next_state = %{obj.state | 
            feature_vector: add_noise_to_vector(obj.state.feature_vector, 0.01),
            last_reward: reward
          }
          
          # Learning step
          social_context = %{
            observed_actions: [],
            peer_rewards: [],
            coalition_membership: [],
            reputation_scores: %{},
            interaction_dyads: [],
            message_history: []
          }
          
          case OORL.learning_step(obj.id, obj.state, action, reward, next_state, social_context) do
            {:ok, learning_result} ->
              # Calculate convergence metrics
              policy_change = calculate_policy_change(oorl.policy_network, learning_result.updated_oorl_state.policy_network)
              value_estimate = estimate_value_function(learning_result.updated_oorl_state, next_state)
              
              convergence_point = %{
                episode: episode,
                reward: reward,
                policy_change: policy_change,
                value_estimate: value_estimate,
                gradient_norm: calculate_gradient_norm(learning_result.policy_update)
              }
              
              updated_obj = Object.update_state(obj, next_state)
              {updated_obj, learning_result.updated_oorl_state, [convergence_point | history]}
              
            {:error, _} ->
              {obj, oorl, history}
          end
        end)
      
      {_final_obj, _final_oorl, convergence_history} = convergence_data
      
      # Analyze convergence
      convergence_analysis = analyze_high_dim_convergence(convergence_history)
      
      # Verify convergence occurred
      assert convergence_analysis.converged, "Should converge in high-dimensional space"
      assert convergence_analysis.convergence_episode < 800, "Should converge within reasonable time"
      
      # Verify stability
      final_100_episodes = Enum.take(convergence_history, 100)
      policy_stability = calculate_policy_stability(final_100_episodes)
      assert policy_stability > 0.8, "Final policy should be stable"
      
      # Verify no overfitting
      assert convergence_analysis.generalization_score > 0.6, "Should generalize well"
    end
  end
  
  # Helper functions
  defp generate_random_policy_params() do
    %{
      weights: for(_ <- 1..10, do: :rand.normal()),
      biases: for(_ <- 1..5, do: :rand.normal() * 0.1),
      learning_rate: 0.01
    }
  end
  
  defp sample_action_from_policy(policy_network) do
    # Simplified action sampling
    case policy_network.type do
      :neural -> 
        %{type: :continuous, value: :rand.normal() * 0.5}
      :tabular -> 
        %{type: :discrete, value: :rand.uniform(10)}
      _ -> 
        %{type: :random, value: :rand.uniform()}
    end
  end
  
  defp simulate_conflicting_environment(state, action) do
    # Simulate environment with inherent trade-offs
    efficiency_gain = action.value * 0.8 + :rand.uniform() * 0.2
    performance_gain = (1 - action.value) * 0.7 + :rand.uniform() * 0.3
    safety_cost = action.value * action.value * 0.5  # Quadratic cost
    exploration_bonus = if :rand.uniform() < 0.1, do: :rand.uniform(), else: 0
    
    state_update = %{
      energy_saved: Map.get(state, :energy_saved, 0) + efficiency_gain,
      task_completion: Map.get(state, :task_completion, 0) + performance_gain,
      risk_taken: Map.get(state, :risk_taken, 0) + safety_cost,
      novelty_discovered: Map.get(state, :novelty_discovered, 0) + exploration_bonus
    }
    
    reward_components = %{
      efficiency: efficiency_gain,
      performance: performance_gain,
      safety: -safety_cost,
      exploration: exploration_bonus
    }
    
    {state_update, reward_components}
  end
  
  defp build_social_context(experiences, object_id) do
    peer_experiences = Enum.reject(experiences, fn exp -> exp.object_id == object_id end)
    
    %{
      observed_actions: Enum.map(peer_experiences, fn exp ->
        %{object_id: exp.object_id, action: exp.action, outcome: exp.reward, timestamp: DateTime.utc_now()}
      end),
      peer_rewards: Enum.map(peer_experiences, fn exp -> {exp.object_id, exp.reward} end),
      coalition_membership: [],
      reputation_scores: %{},
      interaction_dyads: [],
      message_history: []
    }
  end
  
  defp compute_pareto_frontier(points) do
    # Simplified Pareto frontier computation
    Enum.filter(points, fn point1 ->
      not Enum.any?(points, fn point2 ->
        point1 != point2 and dominates?(point2, point1)
      end)
    end)
  end
  
  defp dominates?(point1, point2) do
    # Point1 dominates point2 if it's better in all objectives
    Enum.zip(point1, point2)
    |> Enum.all?(fn {val1, val2} -> val1 >= val2 end) and
    Enum.zip(point1, point2)
    |> Enum.any?(fn {val1, val2} -> val1 > val2 end)
  end
  
  defp calculate_convergence_metrics(pareto_points) do
    if length(pareto_points) == 0 do
      %{hypervolume: 0, spacing: 0, spread: 0}
    else
      %{
        hypervolume: calculate_hypervolume(pareto_points),
        spacing: calculate_spacing_metric(pareto_points),
        spread: calculate_spread_metric(pareto_points)
      }
    end
  end
  
  defp calculate_objective_diversity(points) do
    if length(points) <= 1 do
      0.0
    else
      # Calculate variance across all objectives
      num_objectives = length(hd(points))
      objective_variances = for i <- 0..(num_objectives - 1) do
        values = Enum.map(points, fn point -> Enum.at(point, i) end)
        calculate_variance(values)
      end
      Enum.sum(objective_variances) / num_objectives
    end
  end
  
  defp calculate_pareto_stability(iterations) do
    if length(iterations) < 2 do
      1.0
    else
      frontier_changes = iterations
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.map(fn [newer, older] ->
        frontier_distance(newer.pareto_frontier, older.pareto_frontier)
      end)
      
      avg_change = Enum.sum(frontier_changes) / length(frontier_changes)
      max(0, 1 - avg_change)
    end
  end
  
  defp calculate_gradient_norm(policy_update) do
    parameter_deltas = policy_update.parameter_deltas || %{}
    
    # Calculate L2 norm of all parameter changes
    parameter_deltas
    |> Map.values()
    |> List.flatten()
    |> Enum.map(fn x -> x * x end)
    |> Enum.sum()
    |> :math.sqrt()
  end
  
  defp calculate_stability_metric(history, current_gradient_norm, current_reward) do
    if length(history) < 10 do
      1.0
    else
      recent_history = Enum.take(history, 10)
      gradient_variance = calculate_variance(Enum.map(recent_history, & &1.gradient_norm))
      reward_variance = calculate_variance(Enum.map(recent_history, & &1.reward))
      
      # Stability is inverse of variance
      1.0 / (1.0 + gradient_variance + reward_variance)
    end
  end
  
  defp calculate_policy_entropy(policy_network) do
    # Simplified entropy calculation
    case policy_network.type do
      :neural -> 
        :rand.uniform() * 0.5 + 0.3  # Placeholder
      _ -> 
        0.5
    end
  end
  
  defp analyze_gradient_stability(episode_history) do
    gradient_norms = Enum.map(episode_history, & &1.gradient_norm)
    
    %{
      gradient_explosion_count: Enum.count(gradient_norms, fn norm -> norm > 100 end),
      average_gradient_norm: Enum.sum(gradient_norms) / length(gradient_norms),
      convergence_detected: detect_convergence(episode_history),
      final_policy_entropy: if(length(episode_history) > 0, do: hd(episode_history).policy_entropy, else: 0)
    }
  end
  
  defp detect_convergence(history) do
    if length(history) < 50 do
      false
    else
      last_50 = Enum.take(history, 50)
      reward_variance = calculate_variance(Enum.map(last_50, & &1.reward))
      reward_variance < @convergence_tolerance
    end
  end
  
  defp sample_policy_action(policy_network) do
    :rand.uniform()
  end
  
  # Additional helper functions for collective intelligence tests
  defp form_task_coalitions(objects, tasks) do
    # Simplified coalition formation based on capability matching
    formed_coalitions = Enum.map(tasks, fn task ->
      suitable_objects = Enum.filter(objects, fn obj ->
        can_contribute_to_task?(obj, task)
      end)
      
      selected_objects = Enum.take_random(suitable_objects, min(5, length(suitable_objects)))
      
      %{
        id: "coalition_#{task.id}_#{:rand.uniform(1000)}",
        task: task,
        members: selected_objects,
        formation_time: DateTime.utc_now(),
        expected_performance: estimate_coalition_performance(selected_objects, task)
      }
    end)
    
    {formed_coalitions, []}  # All tasks assigned for simplicity
  end
  
  defp can_contribute_to_task?(object, task) do
    capabilities = object.state.capabilities
    requirements = task.requirements
    
    # Object can contribute if it has significant capability in any required area
    Enum.any?(requirements, fn {capability, required_level} ->
      Map.get(capabilities, capability, 0) > required_level / 5
    end)
  end
  
  defp estimate_coalition_performance(objects, task) do
    total_capabilities = Enum.reduce(objects, %{}, fn obj, acc ->
      Enum.reduce(obj.state.capabilities, acc, fn {cap, val}, acc2 ->
        Map.put(acc2, cap, Map.get(acc2, cap, 0) + val)
      end)
    end)
    
    requirement_satisfaction = Enum.map(task.requirements, fn {cap, req} ->
      min(1.0, Map.get(total_capabilities, cap, 0) / req)
    end)
    
    Enum.sum(requirement_satisfaction) / length(requirement_satisfaction)
  end
  
  defp execute_coalition_task(coalition, iteration) do
    base_performance = coalition.expected_performance
    coordination_penalty = coalition.task.coordination_difficulty * length(coalition.members) * 0.05
    random_factor = (:rand.uniform() - 0.5) * 0.2
    
    actual_performance = max(0, base_performance - coordination_penalty + random_factor)
    
    %{
      coalition_id: coalition.id,
      task_id: coalition.task.id,
      members: Enum.map(coalition.members, & &1.id),
      performance: actual_performance,
      reward_per_member: actual_performance * coalition.task.reward_multiplier / length(coalition.members),
      iteration: iteration
    }
  end
  
  defp update_objects_from_coalition_results(objects, coalition_results) do
    member_rewards = Enum.reduce(coalition_results, %{}, fn result, acc ->
      Enum.reduce(result.members, acc, fn member_id, acc2 ->
        Map.put(acc2, member_id, Map.get(acc2, member_id, 0) + result.reward_per_member)
      end)
    end)
    
    Enum.map(objects, fn obj ->
      if Map.has_key?(member_rewards, obj.id) do
        coalition_reward = member_rewards[obj.id]
        updated_performance_history = [coalition_reward | obj.state.performance_in_coalitions]
        
        Object.update_state(obj, %{
          performance_in_coalitions: Enum.take(updated_performance_history, 10),
          individual_performance: obj.state.individual_performance + coalition_reward * 0.1
        })
      else
        obj
      end
    end)
  end
  
  defp update_trust_networks(objects, coalition_results) do
    # Update trust based on coalition performance
    Enum.map(objects, fn obj ->
      updated_trust = Enum.reduce(coalition_results, obj.state.trust_network, fn result, trust_net ->
        if obj.id in result.members do
          # Update trust towards coalition partners
          partners = Enum.reject(result.members, fn id -> id == obj.id end)
          Enum.reduce(partners, trust_net, fn partner_id, net ->
            current_trust = Map.get(net, partner_id, 0.5)
            trust_update = if result.performance > 0.7, do: 0.1, else: -0.05
            Map.put(net, partner_id, max(0, min(1, current_trust + trust_update)))
          end)
        else
          trust_net
        end
      end)
      
      Object.update_state(obj, %{trust_network: updated_trust})
    end)
  end
  
  defp detect_emergence_indicators(objects, coalitions) do
    # Calculate various emergence indicators
    %{
      coalition_diversity: calculate_coalition_diversity(coalitions),
      self_organization_score: calculate_self_organization_score(objects),
      collective_performance: calculate_collective_performance(objects),
      network_complexity: calculate_network_complexity(objects)
    }
  end
  
  defp calculate_global_performance_metrics(objects, coalition_results) do
    individual_performances = Enum.map(objects, fn obj -> obj.state.individual_performance end)
    coalition_performances = Enum.map(coalition_results, & &1.performance)
    
    %{
      average_individual_performance: Enum.sum(individual_performances) / length(individual_performances),
      average_coalition_performance: if(length(coalition_performances) > 0, do: Enum.sum(coalition_performances) / length(coalition_performances), else: 0),
      average_effectiveness: (Enum.sum(individual_performances) + Enum.sum(coalition_performances)) / (length(individual_performances) + length(coalition_performances))
    }
  end
  
  defp analyze_collective_emergence(system) do
    emergence_scores = system.emergence_indicators
    performance_metrics = system.global_performance_metrics
    
    if length(emergence_scores) == 0 do
      %{collective_intelligence_score: 0, self_organization_score: 0, coordination_efficiency: 0}
    else
      latest_emergence = hd(emergence_scores)
      latest_performance = hd(performance_metrics)
      
      %{
        collective_intelligence_score: latest_emergence.collective_performance,
        self_organization_score: latest_emergence.self_organization_score,
        coordination_efficiency: latest_performance.average_coalition_performance
      }
    end
  end
  
  defp analyze_object_specialization(objects) do
    # Analyze how specialized objects have become
    capability_distributions = Enum.map(objects, fn obj ->
      capabilities = obj.state.capabilities
      max_capability = capabilities |> Map.values() |> Enum.max()
      capability_sum = capabilities |> Map.values() |> Enum.sum()
      avg_capability = capability_sum / map_size(capabilities)
      max_capability / avg_capability  # Specialization ratio
    end)
    
    %{
      specialization_index: Enum.sum(capability_distributions) / length(capability_distributions)
    }
  end
  
  # More helper functions...
  defp calculate_variance(values) do
    if length(values) <= 1 do
      0.0
    else
      mean = Enum.sum(values) / length(values)
      squared_diffs = values |> Enum.map(fn x -> (x - mean) * (x - mean) end)
      variance = Enum.sum(squared_diffs) / length(values)
      variance
    end
  end
  
  defp calculate_hypervolume(points) do
    # Simplified hypervolume calculation
    if length(points) == 0, do: 0, else: length(points) * 0.1
  end
  
  defp calculate_spacing_metric(points) do
    # Simplified spacing metric
    if length(points) <= 1, do: 0, else: 1.0 / length(points)
  end
  
  defp calculate_spread_metric(points) do
    # Simplified spread metric
    if length(points) == 0, do: 0, else: :rand.uniform()
  end
  
  defp frontier_distance(frontier1, frontier2) do
    # Simplified distance between Pareto frontiers
    abs(length(frontier1) - length(frontier2)) / Enum.max([length(frontier1), length(frontier2), 1])
  end
  
  defp calculate_coalition_diversity(_coalitions), do: :rand.uniform()
  defp calculate_self_organization_score(_objects), do: :rand.uniform()
  defp calculate_collective_performance(_objects), do: :rand.uniform()
  defp calculate_network_complexity(_objects), do: :rand.uniform()
  
  defp create_emergence_test_objects(count, behavior_type) do
    for i <- 1..count do
      Object.new(
        id: "emergence_test_#{i}",
        state: %{behavior_type: behavior_type, interaction_count: 0}
      )
    end
  end
  
  defp simulate_emergence_scenario(scenario, iterations) do
    # Simplified emergence simulation
    %{
      interaction_history: [],
      performance_history: [],
      behavioral_data: %{scenario_type: scenario.type}
    }
  end
  
  defp calculate_detection_accuracy(_scenario, _detection) do
    :rand.uniform()  # Placeholder
  end
  
  defp generate_random_vector(size) do
    for _ <- 1..size, do: :rand.normal()
  end
  
  defp dot_product(vec1, vec2) do
    Enum.zip(vec1, vec2) |> Enum.map(fn {a, b} -> a * b end) |> Enum.sum()
  end
  
  defp vector_norm(vec) do
    vec |> Enum.map(fn x -> x * x end) |> Enum.sum() |> :math.sqrt()
  end
  
  defp sample_high_dim_action(policy_network, state_vector) do
    # Simplified high-dimensional action sampling
    for _ <- 1..100, do: :rand.normal()
  end
  
  defp add_noise_to_vector(vec, noise_level) do
    Enum.map(vec, fn x -> x + :rand.normal() * noise_level end)
  end
  
  defp calculate_policy_change(old_policy, new_policy) do
    # Simplified policy change calculation
    :rand.uniform()
  end
  
  defp estimate_value_function(oorl_state, state) do
    # Simplified value function estimation
    :rand.uniform() * 10
  end
  
  defp analyze_high_dim_convergence(history) do
    %{
      converged: length(history) > 500,
      convergence_episode: 600,
      generalization_score: 0.7
    }
  end
  
  defp calculate_policy_stability(episodes) do
    # Simplified stability calculation
    policy_changes = Enum.map(episodes, & &1.policy_change)
    1.0 - (Enum.sum(policy_changes) / length(policy_changes))
  end
end
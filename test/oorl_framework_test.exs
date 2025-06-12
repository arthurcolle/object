defmodule OORLFrameworkTest do
  use ExUnit.Case, async: true
  
  alias OORL.{PolicyLearning, CollectiveLearning, MetaLearning}
  
  describe "OORL initialization" do
    test "initializes OORL object with default configuration" do
      {:ok, oorl_state} = OORL.initialize_oorl_object("test_object_1")
      
      assert oorl_state.policy_network.type == :neural
      assert oorl_state.social_learning_graph.center == "test_object_1"
      assert is_list(oorl_state.experience_buffer)
      assert Map.has_key?(oorl_state.meta_learning_state, :learning_history)
    end
    
    test "initializes OORL object with custom configuration" do
      config = %{
        policy_type: :tabular,
        social_learning_enabled: false,
        curiosity_driven: false
      }
      
      {:ok, oorl_state} = OORL.initialize_oorl_object("test_object_2", config)
      
      assert oorl_state.policy_network.type == :tabular
      assert oorl_state.exploration_strategy.type == :epsilon_greedy
    end
  end
  
  describe "PolicyLearning module" do
    test "updates policy with experiences and social context" do
      object_id = "policy_test_object"
      experiences = [
        %{
          state: %{position: [0, 0]},
          action: :move_right,
          reward: 1.0,
          next_state: %{position: [1, 0]},
          social_context: %{observed_actions: [], peer_rewards: []},
          meta_features: %{},
          timestamp: DateTime.utc_now(),
          interaction_dyad: nil,
          learning_signal: 1.0
        }
      ]
      
      social_context = %{
        observed_actions: [],
        peer_rewards: [],
        coalition_membership: [],
        reputation_scores: %{},
        interaction_dyads: [],
        message_history: []
      }
      
      {:ok, policy_updates} = PolicyLearning.update_policy(object_id, experiences, social_context)
      
      assert Map.has_key?(policy_updates, :parameter_deltas)
      assert Map.has_key?(policy_updates, :learning_rate_adjustment)
    end
    
    test "performs social imitation learning" do
      object_id = "social_learner"
      peer_policies = %{"peer_1" => %{type: :neural}, "peer_2" => %{type: :tabular}}
      performance_rankings = [{"peer_1", 0.8}, {"peer_2", 0.6}]
      
      imitation_weights = PolicyLearning.social_imitation_learning(object_id, peer_policies, performance_rankings)
      
      assert is_map(imitation_weights)
      # Should include weights for compatible peers
      assert Map.has_key?(imitation_weights, "peer_1") or Map.has_key?(imitation_weights, "peer_2")
    end
    
    test "interaction dyad learning processes dyad experiences" do
      object_id = "dyad_learner"
      dyad_experiences = [
        %{
          interaction_dyad: "dyad_1",
          reward: 1.2,
          learning_signal: 0.8,
          state: %{},
          action: :collaborate,
          next_state: %{},
          social_context: %{},
          meta_features: %{},
          timestamp: DateTime.utc_now()
        }
      ]
      
      learning_updates = PolicyLearning.interaction_dyad_learning(object_id, dyad_experiences)
      
      assert Map.has_key?(learning_updates, :total_dyad_improvement)
      assert Map.has_key?(learning_updates, :active_dyads)
      assert learning_updates.active_dyads == 1
    end
  end
  
  describe "CollectiveLearning module" do
    test "forms learning coalition with compatible objects" do
      objects = ["obj_1", "obj_2", "obj_3"]
      task_requirements = %{collaboration: true, coordination: false}
      
      case CollectiveLearning.form_learning_coalition(objects, task_requirements) do
        {:ok, coalition} ->
          assert Map.has_key?(coalition, :members)
          assert Map.has_key?(coalition, :trust_weights)
          assert Map.has_key?(coalition, :collective_goals)
          
        {:error, reason} ->
          assert is_binary(reason)
      end
    end
    
    test "performs distributed policy optimization" do
      coalition = %{
        members: ["member_1", "member_2"],
        trust_weights: %{"member_1" => 1.0, "member_2" => 0.8},
        shared_experience_buffer: [],
        collective_goals: [:maximize_collective_reward],
        coordination_protocol: :consensus_based
      }
      
      {:ok, global_update} = CollectiveLearning.distributed_policy_optimization(coalition)
      
      assert Map.has_key?(global_update, :global_gradient)
    end
    
    test "detects emergent behaviors in coalition" do
      coalition = %{
        members: ["emerg_1", "emerg_2", "emerg_3"],
        trust_weights: %{"emerg_1" => 1.0, "emerg_2" => 0.9, "emerg_3" => 0.8}
      }
      
      case CollectiveLearning.emergence_detection(coalition) do
        {:emergent_behavior_detected, details} ->
          assert Map.has_key?(details, :score)
          assert Map.has_key?(details, :behavior_signature)
          assert details.score > 0.2
          
        {:no_emergence, score} ->
          assert is_float(score)
      end
    end
  end
  
  describe "MetaLearning module" do
    test "evolves learning strategy based on performance" do
      object_id = "meta_learner"
      performance_history = [0.3, 0.4, 0.35, 0.6, 0.7]
      environmental_context = %{complexity: :medium, dynamics: :stable}
      
      case MetaLearning.evolve_learning_strategy(object_id, performance_history, environmental_context) do
        {:ok, new_strategy} ->
          assert Map.has_key?(new_strategy, :exploration_rate)
          assert Map.has_key?(new_strategy, :learning_rate_schedule)
          assert Map.has_key?(new_strategy, :social_learning_weight)
          
        {:error, reason} ->
          assert String.contains?(reason, "Meta-learning adaptation failed")
      end
    end
    
    test "performs reward function evolution" do
      object_id = "reward_evolver"
      goal_satisfaction_history = [0.2, 0.1, 0.15, 0.05, 0.03]  # Declining satisfaction
      
      case MetaLearning.reward_function_evolution(object_id, goal_satisfaction_history) do
        {:reward_evolution_needed, new_components} ->
          assert Map.has_key?(new_components, :intrinsic_motivation_boost)
          assert Map.has_key?(new_components, :novelty_seeking_reward)
          
        {:no_evolution_needed, alignment_score} ->
          assert is_float(alignment_score)
      end
    end
    
    test "implements curiosity-driven exploration" do
      object_id = "curious_learner"
      state_visitation_history = [:state_a, :state_b, :state_a, :state_c, :state_a]
      
      {:ok, exploration_result} = MetaLearning.curiosity_driven_exploration(object_id, state_visitation_history)
      
      assert exploration_result.exploration_policy == :curiosity_driven
      assert is_list(exploration_result.target_states)
      assert is_float(exploration_result.expected_information_gain)
    end
  end
  
  describe "OORL learning step integration" do
    test "performs complete learning step with all components" do
      object_id = "integration_test_object"
      state = %{position: [2, 3], energy: 0.8}
      action = :explore_north
      reward = 0.5
      next_state = %{position: [2, 4], energy: 0.7}
      
      social_context = %{
        observed_actions: [%{object_id: "peer_1", action: :explore_east, outcome: :success, timestamp: DateTime.utc_now()}],
        peer_rewards: [{"peer_1", 0.7}],
        coalition_membership: [],
        reputation_scores: %{"peer_1" => 0.8},
        interaction_dyads: [],
        message_history: []
      }
      
      case OORL.learning_step(object_id, state, action, reward, next_state, social_context) do
        {:ok, learning_result} ->
          assert Map.has_key?(learning_result, :policy_update)
          assert Map.has_key?(learning_result, :social_updates)
          assert Map.has_key?(learning_result, :meta_updates)
          assert Map.has_key?(learning_result, :total_learning_signal)
          assert is_float(learning_result.total_learning_signal)
          
        {:error, reason} ->
          flunk("Learning step failed: #{reason}")
      end
    end
  end
  
  describe "OORL framework edge cases" do
    test "handles empty experience buffer gracefully" do
      {:ok, oorl_state} = OORL.initialize_oorl_object("edge_case_object")
      
      assert oorl_state.experience_buffer == []
      assert is_map(oorl_state.policy_network)
    end
    
    test "handles malformed social context" do
      object_id = "malformed_test"
      experiences = []
      malformed_context = %{invalid_key: "invalid_value"}
      
      # Should not crash, might return error or default behavior
      result = PolicyLearning.update_policy(object_id, experiences, malformed_context)
      
      assert is_tuple(result)
      case result do
        {:ok, _} -> :ok
        {:error, _} -> :ok
      end
    end
    
    test "handles zero-member coalition formation" do
      objects = []
      task_requirements = %{}
      
      case CollectiveLearning.form_learning_coalition(objects, task_requirements) do
        {:error, reason} ->
          assert is_binary(reason)
        _ ->
          flunk("Should return error for empty object list")
      end
    end
  end
end
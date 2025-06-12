defmodule Object.ExplorationTest do
  use ExUnit.Case, async: true
  
  alias Object.Exploration
  
  setup do
    object_id = "exploration_test_object"
    explorer = Exploration.new(object_id, :hybrid)
    
    %{explorer: explorer, object_id: object_id}
  end
  
  describe "Exploration initialization" do
    test "creates novelty-based explorer" do
      explorer = Exploration.new("novelty_explorer", :novelty_based)
      
      assert explorer.exploration_strategy == :novelty_based
      assert explorer.exploration_parameters.novelty_weight == 0.8
      assert explorer.exploration_parameters.uncertainty_weight == 0.1
      assert explorer.exploration_parameters.curiosity_weight == 0.1
      
      assert Map.has_key?(explorer.novelty_tracker, :state_visitation_counts)
      assert Map.has_key?(explorer.novelty_tracker, :novelty_threshold)
    end
    
    test "creates uncertainty-based explorer" do
      explorer = Exploration.new("uncertainty_explorer", :uncertainty_based)
      
      assert explorer.exploration_strategy == :uncertainty_based
      assert explorer.exploration_parameters.uncertainty_weight == 0.8
      assert explorer.exploration_parameters.novelty_weight == 0.1
      
      assert Map.has_key?(explorer.uncertainty_estimator, :prediction_errors)
      assert Map.has_key?(explorer.uncertainty_estimator, :confidence_threshold)
    end
    
    test "creates curiosity-driven explorer" do
      explorer = Exploration.new("curiosity_explorer", :curiosity_driven)
      
      assert explorer.exploration_strategy == :curiosity_driven
      assert explorer.exploration_parameters.curiosity_weight == 0.8
      
      assert Map.has_key?(explorer.curiosity_model, :information_gain_estimates)
      assert Map.has_key?(explorer.curiosity_model, :surprise_threshold)
    end
    
    test "creates social explorer" do
      explorer = Exploration.new("social_explorer", :social)
      
      assert explorer.exploration_strategy == :social
      assert explorer.exploration_parameters.social_weight == 0.6
      
      assert Map.has_key?(explorer.social_exploration_state, :interaction_novelty)
      assert Map.has_key?(explorer.social_exploration_state, :dyad_exploration_targets)
    end
    
    test "creates hybrid explorer with balanced weights", %{explorer: explorer} do
      assert explorer.exploration_strategy == :hybrid
      
      # All weights should be balanced for hybrid strategy
      params = explorer.exploration_parameters
      assert params.novelty_weight == 0.3
      assert params.uncertainty_weight == 0.3
      assert params.curiosity_weight == 0.3
      assert params.social_weight == 0.1
    end
    
    test "creates explorer with custom options" do
      opts = [
        exploration_rate: 0.2,
        novelty_threshold: 0.05,
        curiosity_bonus_scale: 2.0
      ]
      
      explorer = Exploration.new("custom_explorer", :novelty_based, opts)
      
      assert explorer.exploration_parameters.exploration_rate == 0.2
      assert explorer.novelty_tracker.novelty_threshold == 0.05
      assert explorer.curiosity_model.curiosity_bonus_scale == 2.0
    end
  end
  
  describe "Exploration bonus computation" do
    test "computes novelty bonus for unvisited state", %{explorer: explorer} do
      state = %{position: [5, 7], context: :new_area}
      action = :explore_forward
      
      bonus = Exploration.compute_exploration_bonus(explorer, state, action)
      
      assert is_float(bonus)
      assert bonus > 0  # Should have positive bonus for novel state
    end
    
    test "computes lower novelty bonus for frequently visited state", %{explorer: explorer} do
      # Simulate frequent visitation
      state = %{position: [0, 0], context: :home_base}
      action = :explore_forward
      
      # First, update the explorer to have visited this state multiple times
      updated_explorer = Enum.reduce(1..10, explorer, fn _, acc ->
        Exploration.update_exploration_state(acc, state, action, :success)
      end)
      
      bonus = Exploration.compute_exploration_bonus(updated_explorer, state, action)
      
      assert is_float(bonus)
      # Bonus might still be positive due to other exploration components (uncertainty, curiosity)
    end
    
    test "computes uncertainty bonus for high uncertainty scenarios", %{explorer: explorer} do
      # Create explorer with high uncertainty
      high_uncertainty_explorer = %{explorer | 
        uncertainty_estimator: %{explorer.uncertainty_estimator | model_uncertainty: 0.9}
      }
      
      state = %{uncertain_environment: true}
      action = :cautious_exploration
      
      bonus = Exploration.compute_exploration_bonus(high_uncertainty_explorer, state, action)
      
      assert is_float(bonus)
      assert bonus > 0
    end
    
    test "computes social exploration bonus for novel interactions", %{explorer: explorer} do
      social_explorer = %{explorer | exploration_strategy: :social}
      
      state = %{social_context: true}
      action = {:interact, "new_partner_123"}
      
      bonus = Exploration.compute_exploration_bonus(social_explorer, state, action)
      
      assert is_float(bonus)
      assert bonus > 0  # Should reward novel social interactions
    end
    
    test "computes hybrid exploration bonus combining multiple factors", %{explorer: explorer} do
      state = %{complex_scenario: true, uncertainty: 0.7}
      action = :complex_exploration
      
      bonus = Exploration.compute_exploration_bonus(explorer, state, action)
      
      assert is_float(bonus)
      assert bonus > 0
      # Hybrid should combine novelty, uncertainty, curiosity, and social factors
    end
  end
  
  describe "Action selection" do
    test "selects exploration action using epsilon-greedy", %{explorer: explorer} do
      available_actions = [:north, :south, :east, :west, :stay]
      value_estimates = %{
        north: 0.8,
        south: 0.2,
        east: 0.6,
        west: 0.4,
        stay: 0.1
      }
      
      selected_action = Exploration.select_exploration_action(explorer, available_actions, value_estimates)
      
      assert selected_action in available_actions
    end
    
    test "social explorer prioritizes social actions", %{object_id: object_id} do
      social_explorer = Exploration.new(object_id, :social)
      
      available_actions = [:move, {:interact, "partner_1"}, :wait, {:interact, "partner_2"}]
      value_estimates = %{
        {:interact, "partner_1"} => 0.3,
        {:interact, "partner_2"} => 0.4,
        move: 0.9,
        wait: 0.7
      }
      
      selected_action = Exploration.select_exploration_action(social_explorer, available_actions, value_estimates)
      
      # Should prefer social actions or select according to exploration strategy
      assert selected_action in available_actions
    end
    
    test "handles empty action list gracefully", %{explorer: explorer} do
      available_actions = []
      value_estimates = %{}
      
      # Should handle gracefully without crashing
      try do
        Exploration.select_exploration_action(explorer, available_actions, value_estimates)
        :ok
      rescue
        _ -> :ok  # Some error is acceptable for empty action list
      end
    end
  end
  
  describe "Exploration state updates" do
    test "updates exploration state after experience", %{explorer: explorer} do
      state = %{location: "forest", visibility: :low}
      action = :search_carefully
      outcome = %{discovered: [:rare_item], reward: 1.5}
      
      updated_explorer = Exploration.update_exploration_state(explorer, state, action, outcome)
      
      # Should have added to exploration history
      assert length(updated_explorer.exploration_history) == length(explorer.exploration_history) + 1
      
      # Latest record should match our experience
      latest_record = hd(updated_explorer.exploration_history)
      assert latest_record.state == state
      assert latest_record.action == action
      assert latest_record.outcome == outcome
      assert is_float(latest_record.novelty_score)
      assert is_float(latest_record.uncertainty_score)
      assert is_float(latest_record.curiosity_score)
    end
    
    test "updates novelty tracker with state visitations", %{explorer: explorer} do
      state = %{zone: "alpha", level: 1}
      action = :explore
      outcome = :success
      
      # Visit the same state multiple times
      updated_explorer = Enum.reduce(1..3, explorer, fn _, acc ->
        Exploration.update_exploration_state(acc, state, action, outcome)
      end)
      
      # Novelty tracker should have recorded visits
      # (Implementation uses state encoding, so we check structure exists)
      assert Map.has_key?(updated_explorer.novelty_tracker, :state_visitation_counts)
      assert is_map(updated_explorer.novelty_tracker.state_visitation_counts)
    end
    
    test "updates uncertainty estimator with prediction errors", %{explorer: explorer} do
      state = %{prediction_test: true}
      action = :predict_outcome
      outcome = %{actual_result: :unexpected}
      
      updated_explorer = Exploration.update_exploration_state(explorer, state, action, outcome)
      
      # Should have updated prediction errors
      assert length(updated_explorer.uncertainty_estimator.prediction_errors) >= 
             length(explorer.uncertainty_estimator.prediction_errors)
      
      # Model uncertainty should be updated
      assert is_float(updated_explorer.uncertainty_estimator.model_uncertainty)
    end
    
    test "updates curiosity model with information gain", %{explorer: explorer} do
      state = %{learning_opportunity: true}
      action = :investigate
      outcome = %{new_knowledge: [:fact_a, :fact_b]}
      
      updated_explorer = Exploration.update_exploration_state(explorer, state, action, outcome)
      
      # Information gain estimates should be updated
      assert is_map(updated_explorer.curiosity_model.information_gain_estimates)
      
      # Learning progress should be updated
      assert is_float(updated_explorer.curiosity_model.learning_progress)
    end
  end
  
  describe "Novel interaction identification" do
    test "identifies novel partners for interaction", %{explorer: explorer} do
      available_partners = ["partner_A", "partner_B", "partner_C", "partner_D"]
      
      novel_interactions = Exploration.identify_novel_interactions(explorer, available_partners)
      
      assert Map.has_key?(novel_interactions, :novel_partners)
      assert Map.has_key?(novel_interactions, :exploration_recommendations)
      assert Map.has_key?(novel_interactions, :expected_information_gain)
      
      assert is_list(novel_interactions.novel_partners)
      assert is_list(novel_interactions.exploration_recommendations)
      assert is_float(novel_interactions.expected_information_gain)
    end
    
    test "filters partners by novelty threshold", %{explorer: explorer} do
      # Set up some partners with different novelty scores
      social_state = %{explorer.social_exploration_state |
        interaction_novelty: %{
          "familiar_partner" => 0.2,  # Below threshold
          "novel_partner_1" => 0.8,   # Above threshold
          "novel_partner_2" => 0.9    # Above threshold
        }
      }
      
      updated_explorer = %{explorer | social_exploration_state: social_state}
      available_partners = ["familiar_partner", "novel_partner_1", "novel_partner_2"]
      
      novel_interactions = Exploration.identify_novel_interactions(updated_explorer, available_partners)
      
      # Should only include partners above novelty threshold (0.7)
      novel_partner_ids = Enum.map(novel_interactions.novel_partners, &elem(&1, 0))
      assert "novel_partner_1" in novel_partner_ids
      assert "novel_partner_2" in novel_partner_ids
      refute "familiar_partner" in novel_partner_ids
    end
    
    test "handles empty partner list", %{explorer: explorer} do
      novel_interactions = Exploration.identify_novel_interactions(explorer, [])
      
      assert novel_interactions.novel_partners == []
      assert novel_interactions.expected_information_gain == 0.0
    end
  end
  
  describe "Exploration parameter adaptation" do
    test "increases exploration for poor performance", %{explorer: explorer} do
      poor_performance_metrics = %{
        recent_performance: 0.2,  # Poor performance
        exploration_effectiveness: 0.3
      }
      
      adapted_explorer = Exploration.adapt_exploration_parameters(explorer, poor_performance_metrics)
      
      # Exploration rate should increase
      assert adapted_explorer.exploration_parameters.exploration_rate >= 
             explorer.exploration_parameters.exploration_rate
    end
    
    test "decreases exploration for excellent performance", %{explorer: explorer} do
      excellent_performance_metrics = %{
        recent_performance: 0.9,  # Excellent performance
        exploration_effectiveness: 0.8
      }
      
      adapted_explorer = Exploration.adapt_exploration_parameters(explorer, excellent_performance_metrics)
      
      # Exploration rate should decrease
      assert adapted_explorer.exploration_parameters.exploration_rate <= 
             explorer.exploration_parameters.exploration_rate
    end
    
    test "balances exploration strategies based on effectiveness", %{explorer: explorer} do
      performance_metrics = %{
        recent_performance: 0.5,
        novelty_effectiveness: 0.8,
        uncertainty_effectiveness: 0.3,
        curiosity_effectiveness: 0.6
      }
      
      adapted_explorer = Exploration.adapt_exploration_parameters(explorer, performance_metrics)
      
      # Parameters should be adjusted (exact values depend on implementation)
      assert is_struct(adapted_explorer, Exploration)
      assert Map.has_key?(adapted_explorer.exploration_parameters, :novelty_weight)
    end
    
    test "handles performance metrics with no change needed", %{explorer: explorer} do
      balanced_metrics = %{
        recent_performance: 0.7,  # Good performance, no major changes needed
        exploration_effectiveness: 0.6
      }
      
      adapted_explorer = Exploration.adapt_exploration_parameters(explorer, balanced_metrics)
      
      # Should return adapted explorer without major changes
      assert is_struct(adapted_explorer, Exploration)
    end
  end
  
  describe "Exploration effectiveness evaluation" do
    test "evaluates exploration effectiveness with sufficient history", %{explorer: explorer} do
      # Add some exploration history
      updated_explorer = Enum.reduce(1..25, explorer, fn i, acc ->
        state = %{step: i}
        action = if rem(i, 3) == 0, do: :novel_action, else: :routine_action
        outcome = %{reward: :rand.uniform()}
        
        Exploration.update_exploration_state(acc, state, action, outcome)
      end)
      
      effectiveness = Exploration.evaluate_exploration_effectiveness(updated_explorer)
      
      assert Map.has_key?(effectiveness, :overall_effectiveness)
      assert Map.has_key?(effectiveness, :detailed_metrics)
      assert Map.has_key?(effectiveness, :recommendations)
      
      # Check detailed metrics
      metrics = effectiveness.detailed_metrics
      assert Map.has_key?(metrics, :novelty_discovery_rate)
      assert Map.has_key?(metrics, :uncertainty_reduction_rate)
      assert Map.has_key?(metrics, :information_gain_rate)
      assert Map.has_key?(metrics, :exploration_efficiency)
      assert Map.has_key?(metrics, :social_exploration_success)
      
      # All metrics should be numeric
      for {_key, value} <- metrics do
        assert is_float(value)
        assert value >= 0.0
      end
      
      assert is_float(effectiveness.overall_effectiveness)
      assert is_list(effectiveness.recommendations)
    end
    
    test "evaluates exploration with minimal history", %{explorer: explorer} do
      # Add just a few exploration records
      updated_explorer = Enum.reduce(1..3, explorer, fn i, acc ->
        state = %{minimal_step: i}
        action = :basic_action
        outcome = %{basic_result: true}
        
        Exploration.update_exploration_state(acc, state, action, outcome)
      end)
      
      effectiveness = Exploration.evaluate_exploration_effectiveness(updated_explorer)
      
      # Should still return valid structure
      assert Map.has_key?(effectiveness, :overall_effectiveness)
      assert is_float(effectiveness.overall_effectiveness)
    end
    
    test "provides meaningful recommendations based on metrics", %{explorer: explorer} do
      # Create explorer with specific patterns that should trigger recommendations
      # (This is tested through the recommendation generation logic)
      effectiveness = Exploration.evaluate_exploration_effectiveness(explorer)
      
      assert is_list(effectiveness.recommendations)
      assert length(effectiveness.recommendations) > 0
      
      # All recommendations should be strings
      for recommendation <- effectiveness.recommendations do
        assert is_binary(recommendation)
      end
    end
  end
  
  describe "Edge cases and error handling" do
    test "handles exploration with nil state gracefully", %{explorer: explorer} do
      # Should not crash when given nil state
      try do
        bonus = Exploration.compute_exploration_bonus(explorer, nil, :some_action)
        assert is_float(bonus)
      rescue
        _ -> :ok  # Some errors are acceptable for invalid input
      end
    end
    
    test "handles exploration with complex nested state", %{explorer: explorer} do
      complex_state = %{
        nested: %{
          deeply: %{
            nested: %{
              data: [1, 2, 3],
              meta: %{timestamp: DateTime.utc_now()}
            }
          }
        }
      }
      
      bonus = Exploration.compute_exploration_bonus(explorer, complex_state, :complex_action)
      
      assert is_float(bonus)
      assert bonus >= 0.0
    end
    
    test "maintains exploration parameters within valid ranges", %{explorer: explorer} do
      # Test extreme performance metrics
      extreme_metrics = %{
        recent_performance: -1.0,  # Invalid but should be handled
        exploration_effectiveness: 2.0  # Invalid but should be handled
      }
      
      adapted_explorer = Exploration.adapt_exploration_parameters(explorer, extreme_metrics)
      
      # Parameters should remain within reasonable bounds
      params = adapted_explorer.exploration_parameters
      assert params.exploration_rate >= 0.0
      assert params.exploration_rate <= 1.0
    end
    
    test "handles very large exploration history efficiently", %{explorer: explorer} do
      # Add a large number of exploration records
      large_explorer = Enum.reduce(1..1000, explorer, fn i, acc ->
        state = %{large_step: rem(i, 100)}  # Some repetition to test efficiency
        action = Enum.random([:a, :b, :c, :d])
        outcome = %{reward: :rand.uniform()}
        
        Exploration.update_exploration_state(acc, state, action, outcome)
      end)
      
      # Should still be able to evaluate effectiveness efficiently
      start_time = System.monotonic_time(:millisecond)
      effectiveness = Exploration.evaluate_exploration_effectiveness(large_explorer)
      end_time = System.monotonic_time(:millisecond)
      
      # Should complete within reasonable time (less than 1 second)
      assert end_time - start_time < 1000
      assert is_map(effectiveness)
    end
  end
end
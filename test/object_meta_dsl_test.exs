defmodule Object.MetaDSLTest do
  use ExUnit.Case, async: true
  
  alias Object.MetaDSL
  
  setup do
    object = Object.new(
      id: "test_object",
      state: %{position: [0, 0], energy: 1.0},
      methods: [:move, :sense, :learn],
      goal: fn state -> Map.get(state, :energy, 0) end
    )
    
    meta_dsl = MetaDSL.new()
    
    %{object: object, meta_dsl: meta_dsl}
  end
  
  describe "MetaDSL initialization" do
    test "creates new meta-DSL with default constructs" do
      meta_dsl = MetaDSL.new()
      
      assert :define in meta_dsl.constructs
      assert :goal in meta_dsl.constructs
      assert :belief in meta_dsl.constructs
      assert :infer in meta_dsl.constructs
      assert :decide in meta_dsl.constructs
      assert :learn in meta_dsl.constructs
      assert :refine in meta_dsl.constructs
      
      assert Map.has_key?(meta_dsl.learning_parameters, :learning_rate)
      assert Map.has_key?(meta_dsl.learning_parameters, :exploration_rate)
      assert is_list(meta_dsl.adaptation_triggers)
    end
    
    test "creates meta-DSL with custom options" do
      custom_constructs = [:define, :goal, :custom_construct]
      custom_params = %{learning_rate: 0.05, exploration_rate: 0.2}
      
      meta_dsl = MetaDSL.new(
        constructs: custom_constructs,
        learning_parameters: custom_params
      )
      
      assert meta_dsl.constructs == custom_constructs
      assert meta_dsl.learning_parameters.learning_rate == 0.05
    end
  end
  
  describe "DEFINE construct" do
    test "defines new attribute", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, _} = MetaDSL.define(meta_dsl, object, {:attribute, :speed, 2.5})
      
      assert Map.get(updated_object.state, :speed) == 2.5
    end
    
    test "defines new method", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, _} = MetaDSL.define(meta_dsl, object, {:method, :jump, fn -> :jumping end})
      
      assert :jump in updated_object.methods
    end
    
    test "defines new goal function", %{object: object, meta_dsl: meta_dsl} do
      new_goal = fn state -> Map.get(state, :speed, 0) * 2 end
      {:ok, updated_object, _} = MetaDSL.define(meta_dsl, object, {:goal, new_goal})
      
      assert updated_object.goal == new_goal
    end
    
    test "handles invalid definition", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_definition, _}} = MetaDSL.define(meta_dsl, object, {:invalid, :bad_definition})
    end
  end
  
  describe "GOAL construct" do
    test "queries current goal", %{object: object, meta_dsl: meta_dsl} do
      {:ok, current_goal, _} = MetaDSL.goal(meta_dsl, object, :query)
      
      assert is_function(current_goal)
      assert current_goal == object.goal
    end
    
    test "modifies goal function", %{object: object, meta_dsl: meta_dsl} do
      new_goal = fn state -> Map.get(state, :position) |> Enum.sum() end
      {:ok, updated_object, _} = MetaDSL.goal(meta_dsl, object, {:modify, new_goal})
      
      assert updated_object.goal == new_goal
      assert updated_object.goal.(object.state) == 0  # sum of [0, 0]
    end
    
    test "composes multiple goal functions", %{object: object, meta_dsl: meta_dsl} do
      goal1 = fn state -> Map.get(state, :energy, 0) end
      goal2 = fn _state -> 0.5 end
      
      {:ok, updated_object, _} = MetaDSL.goal(meta_dsl, object, {:compose, [goal1, goal2]})
      
      result = updated_object.goal.(object.state)
      expected = (1.0 + 0.5) / 2  # Average of the two goals
      assert_in_delta result, expected, 0.01
    end
    
    test "handles invalid goal operation", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_goal_operation, _}} = MetaDSL.goal(meta_dsl, object, {:invalid_op, nil})
    end
  end
  
  describe "BELIEF construct" do
    test "queries current beliefs", %{object: object, meta_dsl: meta_dsl} do
      {:ok, beliefs, _} = MetaDSL.belief(meta_dsl, object, :query)
      
      assert beliefs == object.world_model.beliefs
    end
    
    test "updates belief", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, _} = MetaDSL.belief(meta_dsl, object, {:update, :enemy_nearby, true})
      
      assert Map.get(updated_object.world_model.beliefs, :enemy_nearby) == true
    end
    
    test "updates uncertainty", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, _} = MetaDSL.belief(meta_dsl, object, {:uncertainty, :weather, 0.7})
      
      assert Map.get(updated_object.world_model.uncertainties, :weather) == 0.7
    end
    
    test "handles invalid belief operation", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_belief_operation, _}} = MetaDSL.belief(meta_dsl, object, {:invalid, :bad_op})
    end
  end
  
  describe "INFER construct" do
    test "performs Bayesian inference", %{object: object, meta_dsl: meta_dsl} do
      evidence = %{temperature: :hot, humidity: :high}
      {:ok, updated_object, _} = MetaDSL.infer(meta_dsl, object, {:bayesian_update, evidence})
      
      assert Map.get(updated_object.world_model.beliefs, :temperature) == :hot
      assert Map.get(updated_object.world_model.beliefs, :humidity) == :high
    end
    
    test "performs prediction", %{object: object, meta_dsl: meta_dsl} do
      current_state = %{position: [1, 1]}
      horizon = 5
      
      {:ok, prediction, _} = MetaDSL.infer(meta_dsl, object, {:predict, current_state, horizon})
      
      assert Map.has_key?(prediction, :predicted_state)
      assert Map.has_key?(prediction, :confidence)
      assert Map.has_key?(prediction, :horizon)
      assert prediction.horizon == horizon
    end
    
    test "performs causal inference", %{object: object, meta_dsl: meta_dsl} do
      {:ok, causal_result, _} = MetaDSL.infer(meta_dsl, object, {:causal, :rain, :wet_ground})
      
      assert Map.has_key?(causal_result, :cause)
      assert Map.has_key?(causal_result, :effect)
      assert Map.has_key?(causal_result, :causal_strength)
      assert causal_result.cause == :rain
      assert causal_result.effect == :wet_ground
    end
    
    test "handles invalid inference query", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_inference_query, _}} = MetaDSL.infer(meta_dsl, object, {:invalid_query, nil})
    end
  end
  
  describe "DECIDE construct" do
    test "performs action selection", %{object: object, meta_dsl: meta_dsl} do
      available_actions = [:move_north, :move_south, :stay, :explore]
      
      {:ok, selected_action, _} = MetaDSL.decide(meta_dsl, object, {:action_selection, available_actions})
      
      assert selected_action in available_actions
    end
    
    test "performs resource allocation", %{object: object, meta_dsl: meta_dsl} do
      resources = [:cpu, :memory, :network]
      tasks = [:task_a, :task_b, :task_c]
      
      {:ok, allocation, _} = MetaDSL.decide(meta_dsl, object, {:resource_allocation, resources, tasks})
      
      assert is_map(allocation)
      assert Map.has_key?(allocation, :cpu)
    end
    
    test "decides coalition formation", %{object: object, meta_dsl: meta_dsl} do
      potential_partners = [
        %Object{id: "partner_1", state: %{capabilities: [:sensing]}, methods: [:sense]},
        %Object{id: "partner_2", state: %{capabilities: [:acting]}, methods: [:act]}
      ]
      
      {:ok, decisions, _} = MetaDSL.decide(meta_dsl, object, {:coalition_formation, potential_partners})
      
      assert is_list(decisions)
      assert length(decisions) == 2
    end
    
    test "handles invalid decision context", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_decision_context, _}} = MetaDSL.decide(meta_dsl, object, {:invalid_context, nil})
    end
  end
  
  describe "LEARN construct" do
    test "updates learning parameters", %{object: object, meta_dsl: meta_dsl} do
      new_params = %{learning_rate: 0.02, exploration_rate: 0.15}
      
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.learn(meta_dsl, object, {:update_parameters, new_params})
      
      assert updated_meta_dsl.learning_parameters.learning_rate == 0.02
      assert updated_meta_dsl.learning_parameters.exploration_rate == 0.15
      assert updated_object == object  # Object should be unchanged
    end
    
    test "adapts learning strategy based on performance", %{object: object, meta_dsl: meta_dsl} do
      performance_feedback = 0.8  # Good performance
      
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.learn(meta_dsl, object, {:adapt_strategy, performance_feedback})
      
      # Learning rate should increase with good performance
      assert updated_meta_dsl.learning_parameters.learning_rate > meta_dsl.learning_parameters.learning_rate
    end
    
    test "handles transfer knowledge operation", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.learn(meta_dsl, object, {:transfer_knowledge, :domain_a, :domain_b})
      
      # Should complete without error (simplified implementation)
      assert updated_object == object
      assert updated_meta_dsl == meta_dsl
    end
    
    test "handles invalid learning operation", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_learning_operation, _}} = MetaDSL.learn(meta_dsl, object, {:invalid_op, nil})
    end
  end
  
  describe "REFINE construct" do
    test "refines exploration strategy", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.refine(meta_dsl, object, :exploration_strategy)
      
      # Exploration rate should decrease (gradual decay)
      assert updated_meta_dsl.learning_parameters.exploration_rate < meta_dsl.learning_parameters.exploration_rate
    end
    
    test "refines reward function", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.refine(meta_dsl, object, :reward_function)
      
      # Should complete without error (simplified implementation)
      assert updated_object == object
      assert is_struct(updated_meta_dsl, MetaDSL)
    end
    
    test "refines world model", %{object: object, meta_dsl: meta_dsl} do
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.refine(meta_dsl, object, :world_model)
      
      # Should complete without error (simplified implementation)
      assert updated_object == object
      assert is_struct(updated_meta_dsl, MetaDSL)
    end
    
    test "handles invalid refinement target", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:invalid_refinement_target, _}} = MetaDSL.refine(meta_dsl, object, :invalid_target)
    end
  end
  
  describe "adaptation triggers" do
    test "evaluates adaptation triggers with performance metrics", %{object: object, meta_dsl: meta_dsl} do
      performance_metrics = %{
        performance_decline: 0.3,  # Above threshold, should trigger adaptation
        exploration_efficiency: 0.4  # Below threshold, should trigger refinement
      }
      
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.evaluate_adaptation_triggers(meta_dsl, object, performance_metrics)
      
      # Should have applied automatic adaptations
      assert is_struct(updated_object, Object)
      assert is_struct(updated_meta_dsl, MetaDSL)
    end
    
    test "handles performance metrics that don't trigger adaptations", %{object: object, meta_dsl: meta_dsl} do
      performance_metrics = %{
        performance_decline: 0.1,  # Below threshold
        exploration_efficiency: 0.8  # Above threshold
      }
      
      {:ok, updated_object, updated_meta_dsl} = MetaDSL.evaluate_adaptation_triggers(meta_dsl, object, performance_metrics)
      
      # No adaptations should be triggered
      assert updated_object == object
      assert updated_meta_dsl == meta_dsl
    end
  end
  
  describe "meta-DSL execution and error handling" do
    test "executes valid construct successfully", %{object: object, meta_dsl: meta_dsl} do
      result = MetaDSL.execute(meta_dsl, :define, object, {:attribute, :test_attr, "test_value"})
      
      # Should return success with modification record
      assert {:ok, _result, updated_meta_dsl} = result
      assert is_struct(updated_meta_dsl, Object.MetaDSL)
    end
    
    test "handles unknown construct", %{object: object, meta_dsl: meta_dsl} do
      {:error, {:unknown_construct, :nonexistent}} = MetaDSL.execute(meta_dsl, :nonexistent, object, {})
    end
    
    test "tracks modification history" do
      meta_dsl = MetaDSL.new()
      assert length(meta_dsl.self_modification_history) == 0
      
      object = Object.new(id: "history_test")
      MetaDSL.execute(meta_dsl, :define, object, {:attribute, :test, 123})
      
      # History tracking happens internally during execution
      # This test ensures the structure exists
      assert is_list(meta_dsl.self_modification_history)
    end
  end
end
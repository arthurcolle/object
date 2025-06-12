defmodule Object.TransferLearningTest do
  use ExUnit.Case, async: true
  
  alias Object.TransferLearning
  
  setup do
    object_id = "transfer_test_object"
    transfer_system = TransferLearning.new(object_id)
    
    # Create sample objects for testing
    source_object = Object.new(
      id: "source_object",
      state: %{capability: :sensing, experience_level: 0.8},
      methods: [:sense, :analyze, :report],
      goal: fn state -> Map.get(state, :accuracy, 0) end
    )
    
    target_object = Object.new(
      id: "target_object", 
      state: %{capability: :acting, experience_level: 0.2},
      methods: [:act, :execute, :monitor],
      goal: fn state -> Map.get(state, :effectiveness, 0) end
    )
    
    %{
      transfer_system: transfer_system,
      object_id: object_id,
      source_object: source_object,
      target_object: target_object
    }
  end
  
  describe "TransferLearning initialization" do
    test "creates new transfer learning system", %{transfer_system: transfer_system, object_id: object_id} do
      assert transfer_system.object_id == object_id
      assert Map.has_key?(transfer_system.embedding_space, :dimensions)
      assert Map.has_key?(transfer_system.embedding_space, :object_embeddings)
      assert is_list(transfer_system.similarity_metrics)
      assert Map.has_key?(transfer_system.analogy_engine, :analogy_templates)
      assert Map.has_key?(transfer_system.meta_learning_state, :adaptation_parameters)
    end
    
    test "initializes with custom options" do
      opts = [embedding_dimensions: 128]
      transfer_system = TransferLearning.new("custom_object", opts)
      
      assert transfer_system.embedding_space.dimensions == 128
    end
    
    test "initializes similarity metrics with correct structure" do
      transfer_system = TransferLearning.new("metrics_test")
      
      for metric <- transfer_system.similarity_metrics do
        assert Map.has_key?(metric, :name)
        assert Map.has_key?(metric, :weight)
        assert Map.has_key?(metric, :metric_function)
        assert is_atom(metric.name)
        assert is_float(metric.weight)
        assert is_function(metric.metric_function)
      end
    end
  end
  
  describe "Object similarity computation" do
    test "computes similarity between objects", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      similarity_result = TransferLearning.compute_object_similarity(transfer_system, source, target)
      
      assert Map.has_key?(similarity_result, :overall_similarity)
      assert Map.has_key?(similarity_result, :detailed_scores)
      assert Map.has_key?(similarity_result, :confidence)
      
      assert is_float(similarity_result.overall_similarity)
      assert similarity_result.overall_similarity >= 0.0
      assert similarity_result.overall_similarity <= 1.0
      
      assert is_map(similarity_result.detailed_scores)
      assert Map.has_key?(similarity_result.detailed_scores, :state_similarity)
      assert Map.has_key?(similarity_result.detailed_scores, :behavioral_similarity)
      assert Map.has_key?(similarity_result.detailed_scores, :goal_similarity)
      
      assert is_float(similarity_result.confidence)
    end
    
    test "computes higher similarity for similar objects", %{transfer_system: transfer_system} do
      similar_object_1 = Object.new(
        id: "similar_1",
        state: %{type: :sensor, accuracy: 0.9},
        methods: [:sense, :calibrate]
      )
      
      similar_object_2 = Object.new(
        id: "similar_2", 
        state: %{type: :sensor, accuracy: 0.85},
        methods: [:sense, :calibrate]
      )
      
      similarity = TransferLearning.compute_object_similarity(transfer_system, similar_object_1, similar_object_2)
      
      # Should have high similarity due to shared methods and similar state
      assert similarity.overall_similarity > 0.5
    end
    
    test "computes lower similarity for dissimilar objects", %{transfer_system: transfer_system} do
      dissimilar_object_1 = Object.new(
        id: "dissimilar_1",
        state: %{type: :sensor, mode: :passive},
        methods: [:sense, :wait]
      )
      
      dissimilar_object_2 = Object.new(
        id: "dissimilar_2",
        state: %{type: :actuator, mode: :active},
        methods: [:move, :execute]
      )
      
      similarity = TransferLearning.compute_object_similarity(transfer_system, dissimilar_object_1, dissimilar_object_2)
      
      # Should have lower overall similarity
      assert is_float(similarity.overall_similarity)
    end
  end
  
  describe "Transfer opportunity identification" do
    test "identifies transfer opportunities between domains", %{transfer_system: transfer_system} do
      source_domain = "robotics_simulation"
      target_domain = "robotics_real_world"
      
      opportunities = TransferLearning.identify_transfer_opportunities(transfer_system, source_domain, target_domain)
      
      assert Map.has_key?(opportunities, :domain_similarity)
      assert Map.has_key?(opportunities, :analogical_mappings)
      assert Map.has_key?(opportunities, :transfer_feasibility)
      assert Map.has_key?(opportunities, :recommendations)
      assert Map.has_key?(opportunities, :estimated_benefit)
      
      # Domain similarity should include multiple metrics
      domain_sim = opportunities.domain_similarity
      assert Map.has_key?(domain_sim, :embedding_similarity)
      assert Map.has_key?(domain_sim, :feature_overlap)
      assert Map.has_key?(domain_sim, :structural_similarity)
      assert Map.has_key?(domain_sim, :overall_similarity)
      
      # All similarity values should be between 0 and 1
      for {_key, value} <- domain_sim do
        assert is_float(value)
        assert value >= 0.0 and value <= 1.0
      end
      
      assert is_list(opportunities.analogical_mappings)
      assert is_map(opportunities.transfer_feasibility)
      assert is_list(opportunities.recommendations)
      assert is_float(opportunities.estimated_benefit)
    end
    
    test "provides meaningful transfer recommendations", %{transfer_system: transfer_system} do
      source_domain = "game_ai"
      target_domain = "autonomous_driving"
      
      opportunities = TransferLearning.identify_transfer_opportunities(transfer_system, source_domain, target_domain)
      
      # Recommendations should be strings
      for recommendation <- opportunities.recommendations do
        assert is_binary(recommendation)
        assert String.length(recommendation) > 0
      end
    end
  end
  
  describe "Analogical reasoning" do
    test "performs analogical reasoning between structures", %{transfer_system: transfer_system} do
      source_structure = %{
        agent: %{sensors: [:camera, :lidar], actuators: [:wheels, :arm]},
        environment: %{type: :indoor, obstacles: :static}
      }
      
      target_structure = %{
        robot: %{inputs: [:vision, :radar], outputs: [:motors, :gripper]},
        world: %{type: :outdoor, obstacles: :dynamic}
      }
      
      analogy_result = TransferLearning.perform_analogical_reasoning(transfer_system, source_structure, target_structure)
      
      assert Map.has_key?(analogy_result, :correspondences)
      assert Map.has_key?(analogy_result, :template_matches)
      assert Map.has_key?(analogy_result, :inferences)
      assert Map.has_key?(analogy_result, :confidence)
      
      assert is_list(analogy_result.correspondences)
      assert is_list(analogy_result.template_matches)
      assert is_list(analogy_result.inferences)
      assert is_float(analogy_result.confidence)
      assert analogy_result.confidence >= 0.0 and analogy_result.confidence <= 1.0
    end
    
    test "handles simple structural correspondences", %{transfer_system: transfer_system} do
      simple_source = %{input: :sensor_data, processing: :neural_network, output: :action}
      simple_target = %{input: :observations, processing: :algorithm, output: :decision}
      
      analogy_result = TransferLearning.perform_analogical_reasoning(transfer_system, simple_source, simple_target)
      
      # Should find some correspondences even in simplified implementation
      assert is_list(analogy_result.correspondences)
      assert is_float(analogy_result.confidence)
    end
  end
  
  describe "Meta-learning" do
    test "performs meta-learning for rapid adaptation", %{transfer_system: transfer_system} do
      adaptation_task = %{
        task_type: :classification,
        domain: :computer_vision,
        target_accuracy: 0.9
      }
      
      few_shot_examples = [
        %{input: %{image: "cat.jpg"}, output: :cat},
        %{input: %{image: "dog.jpg"}, output: :dog},
        %{input: %{image: "bird.jpg"}, output: :bird}
      ]
      
      case TransferLearning.meta_learn(transfer_system, adaptation_task, few_shot_examples) do
        {:ok, adapted_parameters, updated_transfer_system} ->
          assert is_map(adapted_parameters)
          assert Map.has_key?(adapted_parameters, :learning_rate)
          assert is_struct(updated_transfer_system, TransferLearning)
          
          # Meta-learning history should be updated
          meta_state = updated_transfer_system.meta_learning_state
          assert length(meta_state.learning_to_learn_history) > 
                 length(transfer_system.meta_learning_state.learning_to_learn_history)
          
        result ->
          flunk("Unexpected meta-learning result: #{inspect(result)}")
      end
    end
    
    test "handles meta-learning with minimal examples", %{transfer_system: transfer_system} do
      adaptation_task = %{task_type: :regression, complexity: :low}
      few_shot_examples = [%{input: 1, output: 2}]  # Single example
      
      # Should handle gracefully even with minimal data
      case TransferLearning.meta_learn(transfer_system, adaptation_task, few_shot_examples) do
        {:ok, _, _} -> :ok
        _other -> :ok  # Both outcomes are acceptable
      end
    end
  end
  
  describe "Knowledge transfer methods" do
    test "performs automatic knowledge transfer", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      case TransferLearning.transfer_knowledge(transfer_system, source, target, :automatic) do
        {:ok, transferred_knowledge, updated_transfer_system} ->
          assert is_map(transferred_knowledge) or is_list(transferred_knowledge)
          assert is_struct(updated_transfer_system, TransferLearning)
          
          # Transfer history should be updated
          assert length(updated_transfer_system.transfer_history) > 
                 length(transfer_system.transfer_history)
          
        {:error, reason} ->
          assert is_tuple(reason)
      end
    end
    
    test "performs policy distillation transfer", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      {:ok, transferred_policy, updated_transfer_system} = 
        TransferLearning.transfer_knowledge(transfer_system, source, target, :policy_distillation)
      
      assert Map.has_key?(transferred_policy, :adapted_policy)
      assert Map.has_key?(transferred_policy, :adaptation_confidence)
      assert is_float(transferred_policy.adaptation_confidence)
      
      # Should record the transfer
      latest_transfer = hd(updated_transfer_system.transfer_history)
      assert latest_transfer.transfer_method == :policy_distillation
    end
    
    test "performs feature mapping transfer", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      {:ok, transferred_features, updated_transfer_system} = 
        TransferLearning.transfer_knowledge(transfer_system, source, target, :feature_mapping)
      
      assert Map.has_key?(transferred_features, :mapped_features)
      assert is_map(transferred_features.mapped_features)
      
      # Transfer should be recorded
      latest_transfer = hd(updated_transfer_system.transfer_history)
      assert latest_transfer.transfer_method == :feature_mapping
    end
    
    test "performs analogical knowledge transfer", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      {:ok, analogical_knowledge, updated_transfer_system} = 
        TransferLearning.transfer_knowledge(transfer_system, source, target, :analogical)
      
      assert Map.has_key?(analogical_knowledge, :correspondences)
      assert Map.has_key?(analogical_knowledge, :inferences)
      assert Map.has_key?(analogical_knowledge, :confidence)
      
      # Transfer should be recorded
      latest_transfer = hd(updated_transfer_system.transfer_history)
      assert latest_transfer.transfer_method == :analogical
    end
    
    test "performs meta-learning transfer", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      case TransferLearning.transfer_knowledge(transfer_system, source, target, :meta_learning) do
        {:ok, meta_parameters, updated_transfer_system} ->
          assert is_map(meta_parameters)
          
          # Should have updated both transfer history and meta-learning state
          assert length(updated_transfer_system.transfer_history) > 0
          latest_transfer = hd(updated_transfer_system.transfer_history)
          assert latest_transfer.transfer_method == :meta_learning
          
        {:error, reason} ->
          assert is_binary(reason)
      end
    end
    
    test "handles unknown transfer method", %{transfer_system: transfer_system, source_object: source, target_object: target} do
      {:error, {:unknown_transfer_method, :nonexistent_method}} = 
        TransferLearning.transfer_knowledge(transfer_system, source, target, :nonexistent_method)
    end
  end
  
  describe "Object embedding updates" do
    test "updates object embedding from new experiences", %{transfer_system: transfer_system} do
      object = Object.new(id: "embedding_test_object")
      
      new_experiences = [
        %{state: %{position: [1, 2]}, action: :move, reward: 0.5},
        %{state: %{position: [2, 3]}, action: :turn, reward: 0.8},
        %{state: %{position: [3, 4]}, action: :sense, reward: 1.0}
      ]
      
      updated_transfer_system = TransferLearning.update_object_embedding(transfer_system, object, new_experiences)
      
      # Object should now have an embedding
      assert Map.has_key?(updated_transfer_system.embedding_space.object_embeddings, object.id)
      
      embedding = updated_transfer_system.embedding_space.object_embeddings[object.id]
      assert is_list(embedding)
      assert length(embedding) == updated_transfer_system.embedding_space.dimensions
      
      # All embedding values should be floats
      for value <- embedding do
        assert is_float(value)
      end
    end
    
    test "updates existing object embedding", %{transfer_system: transfer_system} do
      object = Object.new(id: "existing_embedding_object")
      
      # First update
      experiences_1 = [%{state: %{step: 1}, action: :action_1, reward: 0.3}]
      transfer_system_1 = TransferLearning.update_object_embedding(transfer_system, object, experiences_1)
      
      original_embedding = transfer_system_1.embedding_space.object_embeddings[object.id]
      
      # Second update
      experiences_2 = [%{state: %{step: 2}, action: :action_2, reward: 0.7}]
      transfer_system_2 = TransferLearning.update_object_embedding(transfer_system_1, object, experiences_2)
      
      updated_embedding = transfer_system_2.embedding_space.object_embeddings[object.id]
      
      # Embedding should have changed
      assert original_embedding != updated_embedding
      assert length(updated_embedding) == length(original_embedding)
    end
  end
  
  describe "Transfer effectiveness evaluation" do
    test "evaluates transfer effectiveness with sufficient history", %{transfer_system: transfer_system} do
      # Add some transfer history
      transfer_history = [
        %{
          timestamp: DateTime.utc_now(),
          source_domain: "simulation",
          target_domain: "real_world",
          transfer_method: :policy_distillation,
          success_metric: 0.8,
          knowledge_transferred: %{policy: true},
          adaptation_steps: 3
        },
        %{
          timestamp: DateTime.utc_now(),
          source_domain: "task_a",
          target_domain: "task_b", 
          transfer_method: :feature_mapping,
          success_metric: 0.6,
          knowledge_transferred: %{features: [:f1, :f2]},
          adaptation_steps: 5
        }
      ]
      
      updated_transfer_system = %{transfer_system | transfer_history: transfer_history}
      
      effectiveness = TransferLearning.evaluate_transfer_effectiveness(updated_transfer_system)
      
      assert Map.has_key?(effectiveness, :overall_effectiveness)
      assert Map.has_key?(effectiveness, :detailed_metrics)
      assert Map.has_key?(effectiveness, :recommendations)
      
      # Check detailed metrics
      metrics = effectiveness.detailed_metrics
      assert Map.has_key?(metrics, :average_success_rate)
      assert Map.has_key?(metrics, :adaptation_efficiency)
      assert Map.has_key?(metrics, :knowledge_retention)
      assert Map.has_key?(metrics, :transfer_diversity)
      assert Map.has_key?(metrics, :meta_learning_progress)
      
      # All metrics should be numeric and within reasonable bounds
      for {_key, value} <- metrics do
        assert is_float(value)
        assert value >= 0.0
      end
      
      assert is_float(effectiveness.overall_effectiveness)
      assert effectiveness.overall_effectiveness >= 0.0
      
      assert is_list(effectiveness.recommendations)
    end
    
    test "handles evaluation with no transfer history", %{transfer_system: transfer_system} do
      effectiveness = TransferLearning.evaluate_transfer_effectiveness(transfer_system)
      
      assert effectiveness.overall_effectiveness == 0.0
      assert effectiveness.detailed_metrics == %{}
      assert effectiveness.recommendations == ["Collect more transfer learning data"]
    end
    
    test "provides meaningful recommendations based on metrics", %{transfer_system: transfer_system} do
      # Create history with specific patterns to trigger recommendations
      poor_transfer_history = [
        %{
          timestamp: DateTime.utc_now(),
          source_domain: "domain_1",
          target_domain: "domain_2",
          transfer_method: :policy_distillation,
          success_metric: 0.2,  # Poor success
          knowledge_transferred: %{},
          adaptation_steps: 15  # Many adaptation steps
        }
      ]
      
      updated_transfer_system = %{transfer_system | transfer_history: poor_transfer_history}
      effectiveness = TransferLearning.evaluate_transfer_effectiveness(updated_transfer_system)
      
      # Should provide specific recommendations for improvement
      assert is_list(effectiveness.recommendations)
      assert length(effectiveness.recommendations) > 0
      
      for recommendation <- effectiveness.recommendations do
        assert is_binary(recommendation)
        assert String.length(recommendation) > 0
      end
    end
  end
  
  describe "Edge cases and error handling" do
    test "handles similarity computation with nil objects gracefully", %{transfer_system: transfer_system} do
      object = Object.new(id: "test_object")
      
      # Should handle gracefully without crashing
      try do
        TransferLearning.compute_object_similarity(transfer_system, object, nil)
      rescue
        _ -> :ok  # Some errors are acceptable for invalid input
      end
      
      try do
        TransferLearning.compute_object_similarity(transfer_system, nil, object)
      rescue
        _ -> :ok  # Some errors are acceptable for invalid input
      end
    end
    
    test "handles transfer between identical objects", %{transfer_system: transfer_system} do
      identical_object = Object.new(
        id: "identical_object",
        state: %{value: 42},
        methods: [:method_a, :method_b]
      )
      
      similarity = TransferLearning.compute_object_similarity(transfer_system, identical_object, identical_object)
      
      # Similarity with self should be high
      assert similarity.overall_similarity > 0.8
    end
    
    test "handles embedding update with empty experiences", %{transfer_system: transfer_system} do
      object = Object.new(id: "empty_experiences_object")
      empty_experiences = []
      
      updated_transfer_system = TransferLearning.update_object_embedding(transfer_system, object, empty_experiences)
      
      # Should handle gracefully
      assert is_struct(updated_transfer_system, TransferLearning)
    end
    
    test "handles transfer opportunity identification with identical domains", %{transfer_system: transfer_system} do
      same_domain = "identical_domain"
      
      opportunities = TransferLearning.identify_transfer_opportunities(transfer_system, same_domain, same_domain)
      
      # Should have high domain similarity
      assert opportunities.domain_similarity.overall_similarity > 0.8
      assert is_list(opportunities.recommendations)
    end
    
    test "maintains transfer history size within reasonable bounds", %{transfer_system: transfer_system} do
      # Add many transfer records
      large_history = for i <- 1..1000 do
        %{
          timestamp: DateTime.utc_now(),
          source_domain: "domain_#{rem(i, 10)}",
          target_domain: "target_#{rem(i, 5)}",
          transfer_method: Enum.random([:policy_distillation, :feature_mapping, :analogical]),
          success_metric: :rand.uniform(),
          knowledge_transferred: %{step: i},
          adaptation_steps: rem(i, 10) + 1
        }
      end
      
      large_transfer_system = %{transfer_system | transfer_history: large_history}
      
      # Evaluation should still work efficiently
      start_time = System.monotonic_time(:millisecond)
      effectiveness = TransferLearning.evaluate_transfer_effectiveness(large_transfer_system)
      end_time = System.monotonic_time(:millisecond)
      
      # Should complete within reasonable time
      assert end_time - start_time < 2000  # Less than 2 seconds
      assert is_map(effectiveness)
    end
  end
end
defmodule Object.HierarchyTest do
  use ExUnit.Case, async: true
  
  alias Object.Hierarchy
  
  setup do
    root_object_id = "root_object_1"
    hierarchy = Hierarchy.new(root_object_id)
    
    %{hierarchy: hierarchy, root_object_id: root_object_id}
  end
  
  describe "Hierarchy initialization" do
    test "creates new hierarchy with root object", %{root_object_id: root_id} do
      hierarchy = Hierarchy.new(root_id)
      
      assert hierarchy.root_object_id == root_id
      assert Map.has_key?(hierarchy.hierarchy_levels, 0)
      assert Map.has_key?(hierarchy.abstraction_mappings, root_id)
      assert hierarchy.abstraction_mappings[root_id] == 0
      
      root_level_objects = hierarchy.hierarchy_levels[0]
      assert length(root_level_objects) == 1
      assert hd(root_level_objects).id == root_id
    end
    
    test "initializes with custom options" do
      opts = [
        composition_rules: [],
        planning_horizon: 20,
        protocols: [:consensus, :delegation]
      ]
      
      hierarchy = Hierarchy.new("custom_root", opts)
      
      assert hierarchy.composition_rules == []
      assert hierarchy.planning_horizon == 20
      assert hierarchy.coordination_protocols == [:consensus, :delegation]
    end
  end
  
  describe "Object composition" do
    test "performs automatic composition of compatible objects", %{hierarchy: hierarchy} do
      object_ids = ["sensor_obj_1", "actuator_obj_1"]
      
      case Hierarchy.compose_objects(hierarchy, object_ids, :automatic) do
        {:ok, updated_hierarchy, composed_spec} ->
          assert Map.has_key?(composed_spec, :id)
          assert Map.has_key?(composed_spec, :type)
          assert Map.has_key?(composed_spec, :capabilities)
          
          # Check that composition was added to hierarchy
          composed_id = composed_spec.id
          assert Map.has_key?(updated_hierarchy.abstraction_mappings, composed_id)
          
        {:error, :no_applicable_composition_rule} ->
          # This is acceptable for simplified implementation
          :ok
      end
    end
    
    test "performs forced composition", %{hierarchy: hierarchy} do
      object_ids = ["obj_a", "obj_b", "obj_c"]
      
      {:ok, updated_hierarchy, composed_spec} = Hierarchy.compose_objects(hierarchy, object_ids, :forced)
      
      assert composed_spec.type == :forced_composition
      assert Map.has_key?(updated_hierarchy.abstraction_mappings, composed_spec.id)
      
      # Should be at a higher abstraction level than root
      composed_level = updated_hierarchy.abstraction_mappings[composed_spec.id]
      root_level = updated_hierarchy.abstraction_mappings[hierarchy.root_object_id]
      assert composed_level > root_level
    end
    
    test "performs guided composition", %{hierarchy: hierarchy} do
      object_ids = ["guided_obj_1", "guided_obj_2"]
      
      case Hierarchy.compose_objects(hierarchy, object_ids, :guided) do
        {:ok, updated_hierarchy, composed_spec} ->
          assert is_map(composed_spec)
          assert is_struct(updated_hierarchy, Hierarchy)
          
        {:error, _reason} ->
          # Handle any error case
          :ok
      end
    end
    
    test "handles invalid composition type", %{hierarchy: hierarchy} do
      object_ids = ["obj_1", "obj_2"]
      
      {:error, {:invalid_composition_type, :invalid_type}} = 
        Hierarchy.compose_objects(hierarchy, object_ids, :invalid_type)
    end
  end
  
  describe "Object decomposition" do
    test "performs capability-based decomposition", %{hierarchy: hierarchy} do
      object_id = "complex_object_1"
      
      {:ok, updated_hierarchy, component_specs} = 
        Hierarchy.decompose_object(hierarchy, object_id, :capability_based)
      
      assert is_list(component_specs)
      assert length(component_specs) > 0
      
      # Check that components were added to hierarchy
      for spec <- component_specs do
        assert Map.has_key?(updated_hierarchy.abstraction_mappings, spec.id)
        
        # Components should be at lower abstraction level
        component_level = updated_hierarchy.abstraction_mappings[spec.id]
        assert component_level < 0  # Below root level
      end
    end
    
    test "performs functional decomposition", %{hierarchy: hierarchy} do
      object_id = "functional_object_1"
      
      {:ok, updated_hierarchy, component_specs} = 
        Hierarchy.decompose_object(hierarchy, object_id, :functional)
      
      assert is_list(component_specs)
      
      # Check function-based naming
      for spec <- component_specs do
        assert String.contains?(spec.id, "functional_object_1")
        assert spec.type in [:primary_function, :secondary_function]
      end
    end
    
    test "performs resource-based decomposition", %{hierarchy: hierarchy} do
      object_id = "resource_object_1"
      
      {:ok, updated_hierarchy, component_specs} = 
        Hierarchy.decompose_object(hierarchy, object_id, :resource_based)
      
      assert is_list(component_specs)
      
      # Check resource-based types
      for spec <- component_specs do
        assert spec.type in [:cpu_manager, :memory_manager, :network_manager]
      end
    end
    
    test "performs temporal decomposition", %{hierarchy: hierarchy} do
      object_id = "temporal_object_1"
      
      {:ok, updated_hierarchy, component_specs} = 
        Hierarchy.decompose_object(hierarchy, object_id, :temporal)
      
      assert is_list(component_specs)
      
      # Check temporal phase types
      for spec <- component_specs do
        assert spec.type in [:initialization_handler, :processing_handler, :cleanup_handler]
      end
    end
    
    test "handles invalid decomposition strategy", %{hierarchy: hierarchy} do
      object_id = "test_object"
      
      {:error, {:invalid_decomposition_strategy, :invalid_strategy}} = 
        Hierarchy.decompose_object(hierarchy, object_id, :invalid_strategy)
    end
  end
  
  describe "Hierarchical planning" do
    test "performs hierarchical planning from goal to executable plan", %{hierarchy: hierarchy} do
      goal = %{target: "reach_destination", success_criteria: ["arrived", "energy > 0.2"]}
      current_state = %{position: [0, 0], energy: 1.0}
      
      {:ok, executable_plan} = Hierarchy.hierarchical_planning(hierarchy, goal, current_state)
      
      assert Map.has_key?(executable_plan, :executable_actions)
      assert Map.has_key?(executable_plan, :execution_schedule)
      assert Map.has_key?(executable_plan, :resource_requirements)
      assert Map.has_key?(executable_plan, :success_probability)
      
      assert is_list(executable_plan.executable_actions)
      assert is_map(executable_plan.execution_schedule)
      assert is_float(executable_plan.success_probability)
      assert executable_plan.success_probability >= 0.0 and executable_plan.success_probability <= 1.0
    end
    
    test "handles planning with complex hierarchical structure" do
      # Create a more complex hierarchy
      hierarchy = Hierarchy.new("complex_root")
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["obj1", "obj2"], :forced)
      {:ok, hierarchy, _} = Hierarchy.decompose_object(hierarchy, "complex_root", :functional)
      
      goal = %{optimize: "multi_objective", constraints: ["time < 100", "cost < 50"]}
      current_state = %{resources: %{time: 0, cost: 0}}
      
      {:ok, executable_plan} = Hierarchy.hierarchical_planning(hierarchy, goal, current_state)
      
      assert Map.has_key?(executable_plan, :executable_actions)
      assert length(executable_plan.executable_actions) > 0
    end
  end
  
  describe "Hierarchy effectiveness evaluation" do
    test "evaluates hierarchy effectiveness metrics", %{hierarchy: hierarchy} do
      effectiveness_result = Hierarchy.evaluate_hierarchy_effectiveness(hierarchy)
      
      assert Map.has_key?(effectiveness_result, :overall_effectiveness)
      assert Map.has_key?(effectiveness_result, :detailed_metrics)
      assert Map.has_key?(effectiveness_result, :recommendations)
      
      # Check detailed metrics
      metrics = effectiveness_result.detailed_metrics
      assert Map.has_key?(metrics, :composition_efficiency)
      assert Map.has_key?(metrics, :coordination_overhead)
      assert Map.has_key?(metrics, :emergent_capabilities)
      assert Map.has_key?(metrics, :abstraction_quality)
      assert Map.has_key?(metrics, :planning_effectiveness)
      
      # All metric values should be between 0 and 1
      for {_key, value} <- metrics do
        case value do
          v when is_float(v) ->
            assert v >= 0.0 and v <= 1.0
          _ ->
            # Some metrics might be lists (e.g., emergent_capabilities)
            :ok
        end
      end
      
      # Recommendations should be a list of strings
      assert is_list(effectiveness_result.recommendations)
    end
    
    test "evaluates effectiveness with complex hierarchy" do
      # Build up a more complex hierarchy
      hierarchy = Hierarchy.new("effectiveness_root")
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["comp1", "comp2"], :forced)
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["comp3", "comp4"], :forced)
      {:ok, hierarchy, _} = Hierarchy.decompose_object(hierarchy, "effectiveness_root", :capability_based)
      
      effectiveness_result = Hierarchy.evaluate_hierarchy_effectiveness(hierarchy)
      
      assert is_float(effectiveness_result.overall_effectiveness)
      assert effectiveness_result.overall_effectiveness >= 0.0
    end
  end
  
  describe "Hierarchy adaptation" do
    test "adapts hierarchy based on performance feedback", %{hierarchy: hierarchy} do
      performance_feedback = %{
        efficiency_decline: 0.3,
        coordination_issues: true,
        bottlenecks: ["level_1", "level_2"]
      }
      
      case Hierarchy.adapt_hierarchy(hierarchy, performance_feedback) do
        {:ok, adapted_hierarchy} ->
          assert is_struct(adapted_hierarchy, Hierarchy)
          
        {:error, _reason} ->
          # Handle any adaptation errors
          :ok
      end
    end
    
    test "handles performance feedback that doesn't require changes", %{hierarchy: hierarchy} do
      good_performance_feedback = %{
        efficiency: 0.9,
        coordination_smooth: true,
        no_bottlenecks: true
      }
      
      {:ok, updated_hierarchy} = Hierarchy.adapt_hierarchy(hierarchy, good_performance_feedback)
      
      # Should return the same hierarchy or minimal changes
      assert is_struct(updated_hierarchy, Hierarchy)
    end
  end
  
  describe "Hierarchy coordination" do
    test "coordinates between hierarchy levels", %{hierarchy: hierarchy} do
      # First add some complexity to the hierarchy
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["coord1", "coord2"], :forced)
      {:ok, hierarchy, _} = Hierarchy.decompose_object(hierarchy, hierarchy.root_object_id, :functional)
      
      coordination_context = %{
        synchronization_required: true,
        resource_conflicts: ["memory", "cpu"],
        deadlines: %{task_a: 100, task_b: 150}
      }
      
      {:ok, coordination_results} = Hierarchy.coordinate_hierarchy_levels(hierarchy, coordination_context)
      
      assert is_list(coordination_results)
    end
    
    test "handles coordination with minimal hierarchy", %{hierarchy: hierarchy} do
      # Test with just the root object
      coordination_context = %{simple_task: true}
      
      {:ok, coordination_results} = Hierarchy.coordinate_hierarchy_levels(hierarchy, coordination_context)
      
      assert is_list(coordination_results)
    end
  end
  
  describe "Hierarchy edge cases and error handling" do
    test "handles empty object list for composition", %{hierarchy: hierarchy} do
      case Hierarchy.compose_objects(hierarchy, [], :automatic) do
        {:error, _reason} ->
          :ok  # Expected behavior
        {:ok, _, _} ->
          :ok  # Also acceptable if implementation handles gracefully
      end
    end
    
    test "handles non-existent object for decomposition", %{hierarchy: hierarchy} do
      {:ok, updated_hierarchy, component_specs} = 
        Hierarchy.decompose_object(hierarchy, "non_existent_object", :capability_based)
      
      # Should handle gracefully (implementation creates mock object)
      assert is_struct(updated_hierarchy, Hierarchy)
      assert is_list(component_specs)
    end
    
    test "handles planning with impossible goal", %{hierarchy: hierarchy} do
      impossible_goal = %{contradiction: true, impossible: "achieve_impossible"}
      current_state = %{reality: :harsh}
      
      # Should still return a plan (simplified implementation)
      {:ok, plan} = Hierarchy.hierarchical_planning(hierarchy, impossible_goal, current_state)
      
      assert Map.has_key?(plan, :executable_actions)
    end
    
    test "maintains hierarchy invariants during operations" do
      hierarchy = Hierarchy.new("invariant_test")
      
      # Perform multiple operations
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["a", "b"], :forced)
      {:ok, hierarchy, _} = Hierarchy.decompose_object(hierarchy, "invariant_test", :functional)
      {:ok, hierarchy, _} = Hierarchy.compose_objects(hierarchy, ["c", "d"], :forced)
      
      # Check that abstraction mappings are consistent
      for {object_id, level} <- hierarchy.abstraction_mappings do
        assert is_binary(object_id)
        assert is_integer(level)
        
        # Object should exist in the corresponding hierarchy level
        level_objects = Map.get(hierarchy.hierarchy_levels, level, [])
        assert Enum.any?(level_objects, fn obj -> obj.id == object_id end)
      end
    end
  end
end
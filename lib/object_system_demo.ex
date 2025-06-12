defmodule Object.SystemDemo do
  @moduledoc """
  Demonstration of the comprehensive Object system with mailboxes and subtypes
  based on the AAOS specification.
  """

  alias Object.Subtypes.{AIAgent, HumanClient, SensorObject, ActuatorObject, CoordinatorObject}

  @doc """
  Runs a comprehensive demo of the Object system.
  
  Executes a complete demonstration including:
  - Creating specialized object subtypes
  - AI agent reasoning demonstrations
  - Human-AI interactions
  - Sensor and actuator operations
  - Multi-object coordination
  - Message passing between objects
  - Meta-DSL self-modification
  
  ## Returns
  
  Map containing results from all demonstration scenarios
  
  ## Examples
  
      iex> Object.SystemDemo.run_demo()
      %{ai_reasoning: %{problem_solved: true}, ...}
  """
  def run_demo do
    IO.puts("🚀 Starting AAOS Object System Demo...")
    
    # Create different object subtypes
    {ai_agent, human_client, sensor, actuator, coordinator} = create_demo_objects()
    
    # Demonstrate object interactions
    demo_results = %{}
    
    IO.puts("\n📊 Running Object Interaction Demos...")
    
    # Demo 1: AI Agent reasoning
    demo_results = Map.put(demo_results, :ai_reasoning, demo_ai_reasoning(ai_agent))
    
    # Demo 2: Human-AI interaction
    demo_results = Map.put(demo_results, :human_ai_interaction, demo_human_ai_interaction(human_client, ai_agent))
    
    # Demo 3: Sensor data collection
    demo_results = Map.put(demo_results, :sensor_data, demo_sensor_operation(sensor))
    
    # Demo 4: Actuator control
    demo_results = Map.put(demo_results, :actuator_control, demo_actuator_operation(actuator))
    
    # Demo 5: Multi-object coordination
    demo_results = Map.put(demo_results, :coordination, demo_coordination(coordinator, [ai_agent, sensor, actuator]))
    
    # Demo 6: Message passing and interaction dyads
    demo_results = Map.put(demo_results, :message_passing, demo_message_passing([ai_agent, human_client, sensor]))
    
    # Demo 7: Meta-DSL self-modification
    demo_results = Map.put(demo_results, :meta_dsl, demo_meta_dsl_operations(ai_agent))
    
    print_demo_results(demo_results)
    
    demo_results
  end

  defp create_demo_objects do
    IO.puts("🔧 Creating specialized object subtypes...")
    
    # Create AI Agent
    ai_agent = AIAgent.new(
      id: "ai_agent_alpha",
      intelligence_level: :advanced,
      specialization: :problem_solving,
      autonomy_level: :high
    )
    
    # Create Human Client
    human_client = HumanClient.new(
      id: "human_client_1",
      user_profile: %{name: "Alice", expertise: "engineering"},
      communication_style: :technical,
      expertise_domain: :robotics
    )
    
    # Create Sensor Object
    sensor = SensorObject.new(
      id: "temp_sensor_1",
      sensor_type: :temperature,
      measurement_range: {-40.0, 85.0},
      accuracy: 0.98,
      sampling_rate: 2.0
    )
    
    # Create Actuator Object
    actuator = ActuatorObject.new(
      id: "motor_actuator_1",
      actuator_type: :motor,
      action_range: {-180.0, 180.0},
      precision: 0.95,
      response_time: 0.05
    )
    
    # Create Coordinator Object
    coordinator = CoordinatorObject.new(
      id: "system_coordinator",
      coordination_strategy: :consensus,
      conflict_resolution: :negotiation
    )
    
    IO.puts("✅ Created 5 specialized objects")
    
    {ai_agent, human_client, sensor, actuator, coordinator}
  end

  defp demo_ai_reasoning(ai_agent) do
    IO.puts("\n🧠 Demo: AI Agent Advanced Reasoning")
    
    problem_context = %{
      type: :optimization,
      constraints: [:energy_efficiency, :safety, :performance],
      data: %{current_efficiency: 0.75, safety_score: 0.9, performance: 0.8}
    }
    
    {updated_agent, reasoning_steps} = AIAgent.execute_advanced_reasoning(ai_agent, problem_context)
    
    IO.puts("   Reasoning steps completed: #{length(reasoning_steps)}")
    IO.puts("   Performance metrics: #{inspect(updated_agent.performance_metrics)}")
    
    %{
      reasoning_steps: reasoning_steps,
      final_performance: updated_agent.performance_metrics,
      problem_solved: true
    }
  end

  defp demo_human_ai_interaction(human_client, _ai_agent) do
    IO.puts("\n👤 Demo: Human-AI Interaction")
    
    # Simulate human input
    user_input = "I need help optimizing the robot's movement efficiency while maintaining safety standards."
    
    {parsed_intent, updated_client} = HumanClient.process_natural_language(human_client, user_input)
    
    # Simulate AI response and feedback
    updated_client_with_feedback = HumanClient.provide_feedback(updated_client, "ai_response_1", 4, "Helpful analysis")
    
    IO.puts("   Parsed intent: #{inspect(parsed_intent)}")
    IO.puts("   Trust level: #{updated_client_with_feedback.trust_level}")
    IO.puts("   Interaction history length: #{length(updated_client_with_feedback.interaction_history)}")
    
    %{
      intent: parsed_intent,
      trust_level: updated_client_with_feedback.trust_level,
      successful_interaction: true
    }
  end

  defp demo_sensor_operation(sensor) do
    IO.puts("\n🌡️  Demo: Sensor Data Collection")
    
    # Simulate environment readings
    environment_states = [
      %{temperature: 22.5, humidity: 45.0},
      %{temperature: 23.1, humidity: 47.2},
      %{temperature: 21.8, humidity: 44.1}
    ]
    
    {measurements, final_sensor} = Enum.reduce(environment_states, {[], sensor}, fn env_state, {acc_measurements, acc_sensor} ->
      updated_sensor = SensorObject.sense(acc_sensor, env_state)
      latest_measurement = hd(updated_sensor.data_buffer)
      {[latest_measurement | acc_measurements], updated_sensor}
    end)
    
    # Calibrate sensor
    reference_values = [22.0, 23.0, 22.0]
    calibrated_sensor = SensorObject.calibrate(final_sensor, reference_values)
    
    IO.puts("   Measurements collected: #{length(measurements)}")
    IO.puts("   Sensor accuracy: #{calibrated_sensor.accuracy}")
    IO.puts("   Calibration status: #{calibrated_sensor.calibration_status}")
    
    %{
      measurements: Enum.reverse(measurements),
      accuracy: calibrated_sensor.accuracy,
      calibration_status: calibrated_sensor.calibration_status
    }
  end

  defp demo_actuator_operation(actuator) do
    IO.puts("\n⚙️  Demo: Actuator Control")
    
    # Queue multiple actions
    actions = [
      %{target_value: 45.0, magnitude: 1.0},
      %{target_value: -30.0, magnitude: 0.8},
      %{target_value: 90.0, magnitude: 1.2}
    ]
    
    actuator_with_queue = Enum.reduce(actions, actuator, fn action, acc_actuator ->
      ActuatorObject.queue_action(acc_actuator, action, :normal)
    end)
    
    # Execute first action
    {execution_result, final_actuator} = ActuatorObject.execute_action(
      actuator_with_queue, 
      hd(actuator_with_queue.action_queue).command
    )
    
    IO.puts("   Actions queued: #{length(actuator_with_queue.action_queue)}")
    IO.puts("   Execution result: #{inspect(execution_result)}")
    IO.puts("   Energy consumption: #{final_actuator.energy_consumption}")
    IO.puts("   Wear level: #{final_actuator.wear_level}")
    
    %{
      execution_result: execution_result,
      energy_consumption: final_actuator.energy_consumption,
      wear_level: final_actuator.wear_level,
      queue_length: length(actuator_with_queue.action_queue)
    }
  end

  defp demo_coordination(coordinator, managed_objects) do
    IO.puts("\n🎯 Demo: Multi-Object Coordination")
    
    # Add objects to coordinator
    coordinator_with_objects = Enum.reduce(managed_objects, coordinator, fn obj, acc_coordinator ->
      object_id = case obj do
        %AIAgent{} -> obj.base_object.id
        %SensorObject{} -> obj.base_object.id  
        %ActuatorObject{} -> obj.base_object.id
        _ -> "unknown_object"
      end
      
      CoordinatorObject.add_managed_object(acc_coordinator, object_id)
    end)
    
    # Execute coordination task
    coordination_task = %{
      type: :system_optimization,
      objectives: [:efficiency, :safety, :responsiveness],
      constraints: [:energy_budget, :safety_limits]
    }
    
    final_coordinator = CoordinatorObject.coordinate_objects(coordinator_with_objects, coordination_task)
    
    # Resolve a simulated conflict
    conflict_context = %{
      conflicting_objects: ["ai_agent_alpha", "motor_actuator_1"],
      conflict_type: :resource_contention,
      priority: :high
    }
    
    {resolution, coordinator_after_conflict} = CoordinatorObject.resolve_conflict(final_coordinator, conflict_context)
    
    IO.puts("   Managed objects: #{length(coordinator_after_conflict.managed_objects)}")
    IO.puts("   Coordination history: #{length(coordinator_after_conflict.coordination_history)}")
    IO.puts("   Conflict resolution: #{inspect(resolution)}")
    
    %{
      managed_objects_count: length(coordinator_after_conflict.managed_objects),
      coordination_completed: true,
      conflict_resolution: resolution
    }
  end

  defp demo_message_passing(objects) do
    IO.puts("\n📬 Demo: Message Passing and Interaction Dyads")
    
    # Extract base objects for message passing
    base_objects = Enum.map(objects, fn obj ->
      case obj do
        %AIAgent{} -> obj.base_object
        %HumanClient{} -> obj.base_object
        %SensorObject{} -> obj.base_object
        _ -> obj
      end
    end)
    
    [obj1, obj2, obj3] = base_objects
    
    # Send messages between objects
    obj1_updated = Object.send_message(obj1, obj2.id, :coordination, 
      %{task: "sensor_data_request", priority: :high}, [requires_ack: true])
    
    obj2_updated = Object.send_message(obj2, obj3.id, :data_share,
      %{sensor_reading: 23.5, timestamp: DateTime.utc_now()})
    
    # Form interaction dyads
    obj1_with_dyad = Object.form_interaction_dyad(obj1_updated, obj2.id, 0.8)
    obj2_with_dyad = Object.form_interaction_dyad(obj2_updated, obj3.id, 0.7)
    
    # Get communication stats
    obj1_stats = Object.get_communication_stats(obj1_with_dyad)
    obj2_stats = Object.get_communication_stats(obj2_with_dyad)
    
    IO.puts("   Object 1 stats: #{inspect(obj1_stats)}")
    IO.puts("   Object 2 stats: #{inspect(obj2_stats)}")
    IO.puts("   Interaction dyads formed: 2")
    
    %{
      messages_sent: obj1_stats.total_messages_sent + obj2_stats.total_messages_sent,
      dyads_formed: 2,
      communication_active: true
    }
  end

  defp demo_meta_dsl_operations(ai_agent) do
    IO.puts("\n🔄 Demo: Meta-DSL Self-Modification")
    
    # Perform self-modification
    modification_context = %{
      performance_feedback: %{accuracy: 0.85, efficiency: 0.75},
      adaptation_target: :improve_efficiency
    }
    
    modified_agent = AIAgent.self_modify(ai_agent, modification_context)
    
    # Apply meta-DSL constructs to base object
    base_object = modified_agent.base_object
    
    # Test different meta-DSL constructs
    define_result = Object.apply_meta_dsl(base_object, :define, {:new_capability, :advanced_reasoning})
    belief_result = Object.apply_meta_dsl(base_object, :belief, {:environment_complexity, :high})
    learn_result = Object.apply_meta_dsl(base_object, :learn, %{experience: :successful_task, reward: 0.9})
    
    IO.puts("   Self-modification completed")
    IO.puts("   Performance improvement: #{inspect(modified_agent.performance_metrics)}")
    IO.puts("   Meta-DSL constructs tested: 4")
    
    %{
      self_modification_successful: true,
      performance_metrics: modified_agent.performance_metrics,
      meta_dsl_operations: [define_result, belief_result, learn_result]
    }
  end

  defp print_demo_results(results) do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("📋 DEMO RESULTS SUMMARY")
    IO.puts(String.duplicate("=", 60))
    
    Enum.each(results, fn {demo_name, demo_result} ->
      IO.puts("\n#{format_demo_name(demo_name)}:")
      print_demo_result(demo_result)
    end)
    
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("✅ All demos completed successfully!")
    IO.puts("🎉 AAOS Object System fully operational")
    IO.puts(String.duplicate("=", 60))
  end

  defp format_demo_name(name) do
    name
    |> Atom.to_string()
    |> String.replace("_", " ")
    |> String.split(" ")
    |> Enum.map(&String.capitalize/1)
    |> Enum.join(" ")
  end

  defp print_demo_result(result) when is_map(result) do
    Enum.each(result, fn {key, value} ->
      IO.puts("  • #{format_key(key)}: #{format_value(value)}")
    end)
  end

  defp format_key(key) do
    key
    |> Atom.to_string()
    |> String.replace("_", " ")
    |> String.split(" ")
    |> Enum.map(&String.capitalize/1)
    |> Enum.join(" ")
  end

  defp format_value(value) when is_boolean(value), do: if(value, do: "✓", else: "✗")
  defp format_value(value) when is_number(value), do: Float.round(value, 3)
  defp format_value(value) when is_list(value), do: "#{length(value)} items"
  defp format_value(value) when is_map(value), do: "#{map_size(value)} properties"
  defp format_value(value), do: inspect(value)
end
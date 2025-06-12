defmodule Object.SelfOrganizationDemo do
  @moduledoc """
  Demonstration of the complete self-organizing Object system.
  
  This module showcases how Objects can:
  1. Discover each other and form networks
  2. Use LLM reasoning to coordinate activities
  3. Self-organize into optimal configurations
  4. Adapt to changing conditions
  5. Execute complex workflows through function calling
  
  Run this demo to see the full system in action.
  """
  
  alias Object.{
    SystemOrchestrator,
    InteractionPatterns,
    FunctionCalling,
    LLMIntegration,
    CoordinationService
  }
  
  @doc """
  Runs a complete self-organization demonstration.
  """
  def run_full_demo do
    IO.puts("🚀 Starting Object Self-Organization Demo")
    IO.puts("=" |> String.duplicate(50))
    
    # Start the system services
    {:ok, _orchestrator} = SystemOrchestrator.start_link()
    {:ok, _coordination} = CoordinationService.start_link([])
    
    # Initialize function calling system
    function_system = FunctionCalling.new()
    
    IO.puts("✅ System services started")
    
    # Create a diverse set of Objects
    objects = create_demo_objects()
    IO.puts("✅ Created #{length(objects)} demo objects")
    
    # Register objects in the system
    updated_function_system = register_all_objects(function_system, objects)
    IO.puts("✅ Registered all objects in function calling system")
    
    # Demonstrate self-organization scenarios
    demo_scenarios = [
      :network_formation,
      :load_balancing,
      :collaborative_problem_solving,
      :adaptive_reconfiguration,
      :emergent_workflows
    ]
    
    Enum.each(demo_scenarios, fn scenario ->
      IO.puts("\n🎯 Running scenario: #{scenario}")
      run_scenario(scenario, objects, updated_function_system)
    end)
    
    # Show final system state
    show_final_system_state()
    
    IO.puts("\n🎉 Self-Organization Demo Complete!")
  end
  
  @doc """
  Creates a set of diverse Objects for demonstration.
  """
  def create_demo_objects do
    [
      # AI Agents for reasoning and coordination
      Object.create_subtype(:ai_agent, [
        id: "reasoning_agent_1",
        state: %{role: :strategic_planner, expertise: [:planning, :optimization]},
        methods: [:analyze_situation, :create_plan, :coordinate_execution]
      ]),
      
      Object.create_subtype(:ai_agent, [
        id: "reasoning_agent_2", 
        state: %{role: :problem_solver, expertise: [:analysis, :synthesis]},
        methods: [:solve_problem, :generate_insights, :evaluate_solutions]
      ]),
      
      # Coordinator Objects for system management
      Object.create_subtype(:coordinator_object, [
        id: "load_balancer",
        state: %{role: :load_management, capacity: 1000},
        methods: [:balance_load, :monitor_performance, :redistribute_tasks]
      ]),
      
      Object.create_subtype(:coordinator_object, [
        id: "resource_manager",
        state: %{role: :resource_allocation, resources: %{cpu: 100, memory: 1024}},
        methods: [:allocate_resources, :optimize_usage, :scale_capacity]
      ]),
      
      # Sensor Objects for data collection
      Object.create_subtype(:sensor_object, [
        id: "performance_sensor",
        state: %{sensor_type: :performance, readings: []},
        methods: [:collect_metrics, :analyze_trends, :detect_anomalies]
      ]),
      
      Object.create_subtype(:sensor_object, [
        id: "network_sensor",
        state: %{sensor_type: :network, connectivity_map: %{}},
        methods: [:scan_network, :measure_latency, :detect_failures]
      ]),
      
      # Actuator Objects for system actions
      Object.create_subtype(:actuator_object, [
        id: "configuration_actuator",
        state: %{actuator_type: :configuration, active_configs: []},
        methods: [:apply_configuration, :rollback_changes, :validate_config]
      ]),
      
      Object.create_subtype(:actuator_object, [
        id: "scaling_actuator", 
        state: %{actuator_type: :scaling, scale_history: []},
        methods: [:scale_up, :scale_down, :auto_scale]
      ]),
      
      # Human Client Objects for interface
      Object.create_subtype(:human_client, [
        id: "admin_interface",
        state: %{user_type: :administrator, permissions: [:all]},
        methods: [:receive_updates, :send_commands, :monitor_system]
      ])
    ]
  end
  
  defp register_all_objects(function_system, objects) do
    Enum.reduce(objects, function_system, fn object, acc_system ->
      updated_system = FunctionCalling.register_object(acc_system, object)
      
      # Also register with system orchestrator
      SystemOrchestrator.register_object(object)
      
      updated_system
    end)
  end
  
  defp run_scenario(:network_formation, objects, _function_system) do
    IO.puts("  📡 Objects discovering each other and forming networks...")
    
    # Objects use gossip protocol to discover peers
    [initiator | targets] = objects
    
    case InteractionPatterns.initiate_pattern(
      :gossip_propagation,
      initiator,
      targets,
      %{message: "Network discovery", metadata: %{discovery_round: 1}}
    ) do
      {:ok, propagation_result} ->
        IO.puts("    ✅ Network formed: #{propagation_result.total_nodes_reached} nodes connected")
        IO.puts("    📊 Coverage: #{trunc(propagation_result.coverage_percentage * 100)}%")
        
      {:error, reason} ->
        IO.puts("    ❌ Network formation failed: #{reason}")
    end
  end
  
  defp run_scenario(:load_balancing, objects, function_system) do
    IO.puts("  ⚖️  System automatically balancing load across objects...")
    
    # Load balancer uses LLM reasoning to optimize distribution
    load_balancer = Enum.find(objects, &(&1.id == "load_balancer"))
    target_objects = Enum.reject(objects, &(&1.id == "load_balancer"))
    
    case FunctionCalling.execute_llm_function_call(
      function_system,
      load_balancer,
      :balance_load,
      "Optimize system load distribution for maximum efficiency",
      %{current_load: simulate_system_load(), target_objects: target_objects}
    ) do
      {:ok, result, _updated_system, _adaptations} ->
        IO.puts("    ✅ Load balancing completed")
        IO.puts("    📈 Efficiency improvement: #{inspect(result)}")
        
      {:error, reason} ->
        IO.puts("    ❌ Load balancing failed: #{reason}")
    end
  end
  
  defp run_scenario(:collaborative_problem_solving, objects, _function_system) do
    IO.puts("  🤝 Objects collaborating to solve complex problems...")
    
    # Multiple AI agents collaborate using consensus
    ai_agents = Enum.filter(objects, &(&1.subtype == :ai_agent))
    problem = "Optimize system architecture for 10x scale increase"
    
    case InteractionPatterns.initiate_pattern(
      :swarm_consensus,
      hd(ai_agents),
      tl(ai_agents),
      %{problem: problem, threshold: 0.8}
    ) do
      {:ok, consensus_result} ->
        IO.puts("    ✅ Collaborative solution found")
        IO.puts("    🎯 Consensus score: #{consensus_result.consensus_score}")
        IO.puts("    💡 Solution: #{consensus_result.agreed_decision}")
        
      {:error, reason} ->
        IO.puts("    ❌ Collaboration failed: #{reason}")
    end
  end
  
  defp run_scenario(:adaptive_reconfiguration, _objects, _function_system) do
    IO.puts("  🔄 System adapting to simulated performance degradation...")
    
    # Trigger system self-organization
    case SystemOrchestrator.self_organize(:performance_degradation) do
      {:ok, optimization_result} ->
        IO.puts("    ✅ System reconfigured successfully")
        IO.puts("    🔧 Changes made: #{length(optimization_result.changes)}")
        IO.puts("    📊 Optimization success: #{optimization_result.success}")
        
      {:error, reason} ->
        IO.puts("    ❌ Reconfiguration failed: #{reason}")
    end
  end
  
  defp run_scenario(:emergent_workflows, objects, function_system) do
    IO.puts("  🌟 Emergent workflow execution through function composition...")
    
    # Strategic planner discovers and executes workflow
    planner = Enum.find(objects, &(&1.state[:role] == :strategic_planner))
    
    case FunctionCalling.discover_function_composition(
      function_system,
      planner,
      "Create comprehensive system health report",
      [:available_sensors, :performance_constraints]
    ) do
      {:ok, composition, _updated_planner} ->
        IO.puts("    ✅ Workflow discovered: #{composition.id}")
        IO.puts("    🔗 Steps: #{length(composition.steps)}")
        IO.puts("    📊 Confidence: #{trunc(composition.confidence * 100)}%")
        
        # Execute the composed workflow
        {:ok, workflow_result, _final_system} = FunctionCalling.execute_function_composition(
          function_system,
          planner,
          composition
        )
        IO.puts("    ✅ Workflow executed successfully")
        IO.puts("    ⏱️  Execution time: #{workflow_result.execution_time}ms")
        IO.puts("    📈 Success rate: #{trunc(workflow_result.success_rate * 100)}%")
        
      {:error, reason} ->
        IO.puts("    ❌ Workflow discovery failed: #{reason}")
    end
  end
  
  defp show_final_system_state do
    IO.puts("\n📊 Final System State")
    IO.puts("-" |> String.duplicate(30))
    
    case SystemOrchestrator.get_system_status() do
      status when is_map(status) ->
        IO.puts("🏥 Orchestrator Health: #{status.orchestrator_health}")
        IO.puts("📱 Managed Objects: #{status.managed_objects_count}")
        IO.puts("🕸️  Topology: #{inspect(status.topology)}")
        
        if status.last_adaptation do
          IO.puts("🔄 Last Adaptation: #{status.last_adaptation.timestamp}")
        end
        
      error ->
        IO.puts("❌ Could not retrieve system status: #{inspect(error)}")
    end
    
    # Show coordination service metrics
    case CoordinationService.get_metrics() do
      metrics when is_map(metrics) ->
        IO.puts("⚡ Active Sessions: #{metrics.active_sessions}")
        IO.puts("⏰ Uptime: #{metrics.uptime_seconds}s")
        
      error ->
        IO.puts("❌ Could not retrieve coordination metrics: #{inspect(error)}")
    end
  end
  
  @doc """
  Runs a simple demonstration of object interaction.
  """
  def simple_interaction_demo do
    IO.puts("🔹 Simple Object Interaction Demo")
    
    # Create two objects
    agent1 = Object.create_subtype(:ai_agent, [
      id: "agent_alpha",
      state: %{role: :communicator}
    ])
    
    agent2 = Object.create_subtype(:ai_agent, [
      id: "agent_beta", 
      state: %{role: :responder}
    ])
    
    # Agent1 sends a message to Agent2
    message = %{
      content: "Hello, would you like to collaborate on a task?",
      sender: agent1.id,
      timestamp: DateTime.utc_now()
    }
    
    # Generate LLM response
    {:ok, response, _updated_agent2} = LLMIntegration.generate_response(agent2, message)
    IO.puts("Agent #{agent1.id}: #{message.content}")
    IO.puts("Agent #{agent2.id}: #{response.content}")
    IO.puts("✅ Interaction successful!")
  end
  
  @doc """
  Demonstrates meta-learning and adaptation.
  """
  def meta_learning_demo do
    IO.puts("🧠 Meta-Learning Demo")
    
    # Create a learning agent
    learner = Object.create_subtype(:ai_agent, [
      id: "meta_learner",
      state: %{learning_performance: 0.6, adaptation_count: 0}
    ])
    
    # Simulate performance feedback
    _performance_metrics = %{
      success_rate: 0.4,
      efficiency: 0.3,
      adaptation_needed: true
    }
    
    # Use meta-DSL to adapt learning strategy
    case Object.MetaDSL.execute(
      learner.meta_dsl,
      :refine,
      learner,
      :exploration_strategy
    ) do
      {:ok, refinement_result, updated_meta_dsl} ->
        IO.puts("🔄 Learning strategy refined")
        IO.puts("📈 Adaptation applied: #{inspect(refinement_result)}")
        
        updated_learner = %{learner | meta_dsl: updated_meta_dsl}
        IO.puts("✅ Meta-learning successful!")
        {:ok, updated_learner}
        
      {:error, reason} ->
        IO.puts("❌ Meta-learning failed: #{reason}")
        {:error, reason}
    end
  end
  
  # Helper functions
  
  defp simulate_system_load do
    %{
      cpu_usage: :rand.uniform() * 0.8,
      memory_usage: :rand.uniform() * 0.9,
      network_latency: :rand.uniform() * 100,
      active_tasks: :rand.uniform(50)
    }
  end
end
defmodule Object.Application do
  @moduledoc """
  Main OTP Application for the AAOS Object system.
  
  This module implements the main supervision tree and system initialization
  for the Autonomous Agent Object Specification (AAOS) framework. It manages:
  
  - Core object supervision and lifecycle management
  - Message routing infrastructure
  - System-wide coordination services
  - Performance monitoring and telemetry
  - Demonstration system creation
  - Health monitoring and alerts
  
  The application follows OTP principles with a hierarchical supervision
  strategy that ensures fault tolerance and system resilience.
  """
  
  use Application
  require Logger
  
  @doc """
  Starts the AAOS Object System application.
  
  Initializes the complete supervision tree including:
  - Core object supervisor
  - Message routing subsystem
  - Telemetry configuration
  - Post-startup system initialization
  
  ## Parameters
  
  - `_type` - Application start type (typically `:normal`)
  - `_args` - Application arguments (unused)
  
  ## Returns
  
  - `{:ok, pid}` - Success with supervisor PID
  - `{:error, reason}` - Failure with error reason
  
  ## Examples
  
      iex> Object.Application.start(:normal, [])
      {:ok, #PID<0.123.0>}
  """
  def start(_type, _args) do
    Logger.info("Starting AAOS Object System...")
    
    # Configure telemetry
    configure_telemetry()
    
    children = [
      # Core supervisor containing all subsystems
      Object.Supervisor
    ]
    
    opts = [strategy: :one_for_one, name: Object.ApplicationSupervisor]
    
    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        Logger.info("AAOS Object System started successfully")
        post_startup_initialization()
        {:ok, pid}
      
      error ->
        Logger.error("Failed to start AAOS Object System: #{inspect(error)}")
        error
    end
  end
  
  @doc """
  Stops the AAOS Object System application.
  
  Performs graceful shutdown including cleanup of system resources
  and orderly termination of all objects.
  
  ## Parameters
  
  - `_state` - Application state (unused)
  
  ## Returns
  
  - `:ok` - Always returns ok after cleanup
  """
  def stop(_state) do
    Logger.info("Stopping AAOS Object System...")
    cleanup_system_resources()
    :ok
  end
  
  # Public API for system management
  
  @doc """
  Creates a new object in the system.
  
  Dynamically starts a new object server under the dynamic supervisor
  with the provided specification.
  
  ## Parameters
  
  - `object_spec` - Object specification struct with id, state, methods, etc.
  
  ## Returns
  
  - `{:ok, pid}` - Success with the object's process PID
  - `{:error, reason}` - Failure reason
  
  ## Examples
  
      iex> spec = %Object{id: "test_obj", state: %{value: 1}}
      iex> Object.Application.create_object(spec)
      {:ok, #PID<0.456.0>}
  """
  def create_object(object_spec) do
    DynamicSupervisor.start_child(
      Object.DynamicSupervisor,
      {Object.Server, object_spec}
    )
  end
  
  @doc """
  Stops an object by its ID.
  
  Looks up the object in the registry and terminates its process
  under the dynamic supervisor.
  
  ## Parameters
  
  - `object_id` - String identifier of the object to stop
  
  ## Returns
  
  - `:ok` - Object stopped successfully
  - `{:error, :object_not_found}` - Object not found in registry
  - `{:error, reason}` - Other termination error
  """
  def stop_object(object_id) do
    case Registry.lookup(Object.Registry, object_id) do
      [{pid, _}] ->
        DynamicSupervisor.terminate_child(Object.DynamicSupervisor, pid)
      
      [] ->
        {:error, :object_not_found}
    end
  end
  
  @doc """
  Lists all currently active objects in the system.
  
  ## Returns
  
  - List of object IDs currently registered in the system
  
  ## Examples
  
      iex> Object.Application.list_objects()
      ["ai_agent_alpha", "sensor_1", "coordinator_main"]
  """
  def list_objects() do
    Registry.select(Object.Registry, [{{:"$1", :"$2", :"$3"}, [], [:"$1"]}])
  end
  
  @doc """
  Gets comprehensive system status information.
  
  Returns detailed metrics about system health, performance,
  and operational status across all subsystems.
  
  ## Returns
  
  Map containing:
  - `:objects` - Object count and type information
  - `:performance` - System performance metrics
  - `:coordination` - Coordination service metrics
  - `:evolution` - Schema evolution metrics
  - `:message_routing` - Message routing statistics
  - `:uptime` - System uptime in milliseconds
  
  ## Examples
  
      iex> Object.Application.get_system_status()
      %{
        objects: %{total: 5, by_type: %{ai_agent: 2, sensor: 3}},
        performance: %{avg_response_time: 12.5},
        uptime: 3600000
      }
  """
  def get_system_status() do
    base_status = %{
      objects: %{
        total: count_objects(),
        by_type: count_objects_by_type()
      },
      uptime: get_system_uptime()
    }
    
    # Add optional metrics if services are available
    optional_metrics = %{}
    
    optional_metrics = try do
      Map.put(optional_metrics, :performance, Object.PerformanceMonitor.get_system_metrics())
    rescue
      _ -> optional_metrics
    end
    
    optional_metrics = try do
      Map.put(optional_metrics, :coordination, Object.CoordinationService.get_metrics())
    rescue
      _ -> optional_metrics
    end
    
    optional_metrics = try do
      Map.put(optional_metrics, :evolution, Object.SchemaEvolutionManager.get_evolution_metrics())
    rescue
      _ -> optional_metrics
    end
    
    optional_metrics = try do
      Map.put(optional_metrics, :message_routing, Object.MessageRouter.get_routing_stats())
    rescue
      _ -> optional_metrics
    end
    
    Map.merge(base_status, optional_metrics)
  end
  
  @doc """
  Creates a complete demonstration system with various object types.
  
  Sets up a realistic demonstration environment including:
  - AI agents with different capabilities
  - Human client objects
  - Sensor and actuator objects
  - Coordinator objects
  - Interaction dyads between related objects
  - Coordination tasks and scenarios
  
  This is useful for testing, demos, and understanding system behavior.
  
  ## Returns
  
  Map containing:
  - `:objects_created` - Number of successfully created objects
  - `:total_objects` - Total objects attempted
  - `:results` - Detailed results for each object creation attempt
  
  ## Examples
  
      iex> Object.Application.create_demonstration_system()
      %{
        objects_created: 7,
        total_objects: 7,
        results: [{:ok, "ai_agent_alpha", #PID<0.789.0>}, ...]
      }
  """
  def create_demonstration_system() do
    Logger.info("Creating demonstration AAOS system...")
    
    # Create different types of objects for demonstration
    demonstration_objects = [
      # AI Agents
      %Object{
        id: "ai_agent_alpha",
        subtype: :ai_agent,
        state: %{intelligence_level: :advanced, specialization: :problem_solving},
        methods: [:learn, :reason, :plan, :execute, :adapt, :self_modify]
      },
      
      %Object{
        id: "ai_agent_beta", 
        subtype: :ai_agent,
        state: %{intelligence_level: :intermediate, specialization: :coordination},
        methods: [:learn, :coordinate, :communicate, :adapt]
      },
      
      # Human Clients
      %Object{
        id: "human_client_alice",
        subtype: :human_client,
        state: %{expertise: :engineering, trust_level: 0.8},
        methods: [:communicate, :provide_feedback, :request_service]
      },
      
      # Sensor Objects
      %Object{
        id: "temperature_sensor_1",
        subtype: :sensor_object,
        state: %{sensor_type: :temperature, accuracy: 0.98, sampling_rate: 2.0},
        methods: [:sense, :calibrate, :filter_noise, :transmit_data]
      },
      
      %Object{
        id: "pressure_sensor_1",
        subtype: :sensor_object,
        state: %{sensor_type: :pressure, accuracy: 0.95, sampling_rate: 1.0},
        methods: [:sense, :calibrate, :filter_noise, :transmit_data]
      },
      
      # Actuator Objects
      %Object{
        id: "motor_actuator_1",
        subtype: :actuator_object,
        state: %{actuator_type: :motor, precision: 0.95, response_time: 0.05},
        methods: [:execute_action, :queue_action, :calibrate_motion, :monitor_wear]
      },
      
      # Coordinator Object
      %Object{
        id: "system_coordinator",
        subtype: :coordinator_object,
        state: %{coordination_strategy: :consensus, managed_objects: []},
        methods: [:coordinate, :resolve_conflicts, :allocate_resources, :monitor_performance]
      }
    ]
    
    # Create all demonstration objects
    results = Enum.map(demonstration_objects, fn object_spec ->
      case create_object(object_spec) do
        {:ok, pid} ->
          Logger.info("Created object #{object_spec.id} (#{object_spec.subtype})")
          {:ok, object_spec.id, pid}
        
        error ->
          Logger.error("Failed to create object #{object_spec.id}: #{inspect(error)}")
          {:error, object_spec.id, error}
      end
    end)
    
    # Form interaction dyads between related objects
    form_demonstration_dyads()
    
    # Start demonstration coordination tasks
    start_demonstration_coordination()
    
    success_count = Enum.count(results, fn {status, _, _} -> status == :ok end)
    
    Logger.info("Demonstration system created: #{success_count}/#{length(demonstration_objects)} objects started")
    
    %{
      objects_created: success_count,
      total_objects: length(demonstration_objects),
      results: results
    }
  end
  
  # Private functions
  
  defp configure_telemetry do
    # Set up telemetry for performance monitoring
    :telemetry.attach_many(
      "aaos-system-metrics",
      [
        [:object, :server, :start],
        [:object, :server, :stop],
        [:object, :message, :route],
        [:coordination, :session, :complete],
        [:evolution, :proposal, :vote]
      ],
      &__MODULE__.handle_system_telemetry/4,
      %{}
    )
  end
  
  def handle_system_telemetry(event, measurements, metadata, _config) do
    # Forward to performance monitor
    case event do
      [:object, :server, :start] ->
        Object.PerformanceMonitor.record_metric("system", :objects_started, 1, metadata)
      
      [:object, :server, :stop] ->
        Object.PerformanceMonitor.record_metric("system", :objects_stopped, 1, metadata)
      
      [:object, :message, :route] ->
        Object.PerformanceMonitor.record_metric("system", :messages_routed, 1, %{
          route_time: measurements[:duration]
        })
      
      _ ->
        :ok
    end
  end
  
  defp post_startup_initialization do
    # Perform any post-startup initialization
    Logger.info("Performing post-startup initialization...")
    
    # Initialize default performance thresholds if available
    try do
      Object.PerformanceMonitor.set_performance_threshold(
        :method_execution_time,
        %{warning: 1000, critical: 5000, low_warning: 10}
      )
      
      Object.PerformanceMonitor.set_performance_threshold(
        :goal_evaluation_score,
        %{warning: 0.3, critical: 0.1, low_warning: 0.0}
      )
    rescue
      _ -> 
        Logger.debug("Performance Monitor not available, skipping threshold setup")
    catch
      :exit, _ -> 
        Logger.debug("Performance Monitor not available, skipping threshold setup")
    end
    
    # Start system health monitoring
    spawn_link(fn -> system_health_monitor() end)
    
    Logger.info("Post-startup initialization complete")
  end
  
  defp cleanup_system_resources do
    # Clean up any system resources before shutdown
    Logger.info("Cleaning up system resources...")
    
    # Stop all objects gracefully
    Object.Registry
    |> Registry.select([{{:"$1", :"$2", :"$3"}, [], [:"$2"]}])
    |> Enum.each(fn pid ->
      try do
        GenServer.stop(pid, :normal, 5000)
      catch
        :exit, _reason -> :ok
      end
    end)
    
    Logger.info("System cleanup complete")
  end
  
  defp count_objects do
    Registry.count(Object.Registry)
  end
  
  defp count_objects_by_type do
    try do
      Object.SchemaRegistry.list_objects()
      |> Enum.group_by(fn {_id, schema} -> schema.subtype end)
      |> Enum.into(%{}, fn {type, objects} -> {type, length(objects)} end)
    rescue
      _ -> %{} # Schema registry not available
    end
  end
  
  defp get_system_uptime do
    # Simple uptime calculation (would be more sophisticated in production)
    {uptime_ms, _} = :erlang.statistics(:wall_clock)
    uptime_ms
  end
  
  defp form_demonstration_dyads do
    # Form dyads between related objects for demonstration
    dyad_pairs = [
      {"ai_agent_alpha", "system_coordinator"},
      {"ai_agent_beta", "human_client_alice"},
      {"temperature_sensor_1", "ai_agent_alpha"},
      {"pressure_sensor_1", "ai_agent_beta"},
      {"motor_actuator_1", "system_coordinator"}
    ]
    
    Enum.each(dyad_pairs, fn {obj1_id, obj2_id} ->
      case Registry.lookup(Object.Registry, obj1_id) do
        [{_pid, _}] ->
          Object.Server.form_dyad(obj1_id, obj2_id, 0.8)
          Logger.debug("Formed dyad between #{obj1_id} and #{obj2_id}")
        
        [] ->
          Logger.warning("Could not form dyad: object #{obj1_id} not found")
      end
    end)
  end
  
  defp start_demonstration_coordination do
    # Start a demonstration coordination task
    spawn_link(fn ->
      :timer.sleep(5000)  # Wait for objects to initialize
      
      coordination_task = %{
        type: :system_optimization,
        objectives: [:efficiency, :responsiveness],
        constraints: [:energy_budget, :safety_limits]
      }
      
      participating_objects = [
        "ai_agent_alpha",
        "ai_agent_beta", 
        "system_coordinator"
      ]
      
      try do
        case Object.CoordinationService.coordinate_objects(participating_objects, coordination_task) do
          {:ok, session_id} ->
            Logger.info("Started demonstration coordination session: #{session_id}")
          
          error ->
            Logger.warning("Failed to start demonstration coordination: #{inspect(error)}")
        end
      rescue
        _ ->
          Logger.debug("Coordination service not available, skipping demonstration coordination")
      end
    end)
  end
  
  defp system_health_monitor do
    # Continuous system health monitoring
    :timer.sleep(30000)  # Wait 30 seconds before first check
    
    monitor_loop()
  end
  
  defp monitor_loop do
    try do
      # Check system health metrics if available
      try do
        _system_metrics = Object.PerformanceMonitor.get_system_metrics()
        alerts = Object.PerformanceMonitor.get_performance_alerts()
        
        # Log any critical issues
        critical_alerts = Enum.filter(alerts, fn {_id, alert} ->
          alert.alert_level == :critical and not alert.acknowledged
        end)
        
        if length(critical_alerts) > 0 do
          Logger.warning("System has #{length(critical_alerts)} critical performance alerts")
        end
      rescue
        _ -> :ok # Performance monitor not available
      end
      
      # Check object health
      object_count = count_objects()
      
      if object_count == 0 do
        Logger.debug("No objects currently running in the system")
      end
      
      # Schedule next check
      :timer.sleep(60000)  # Check every minute
      monitor_loop()
      
    rescue
      error ->
        Logger.error("System health monitor error: #{inspect(error)}")
        :timer.sleep(60000)
        monitor_loop()
    end
  end
end
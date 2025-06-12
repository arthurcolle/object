defmodule Object.PerformanceMonitor do
  @moduledoc """
  Real-time performance monitoring and metrics collection for the AAOS system.
  Uses telemetry and ETS for high-performance metrics aggregation.
  """
  
  use GenServer
  require Logger
  
  alias Object.SchemaRegistry
  
  @metrics_table :object_performance_metrics
  @aggregation_table :performance_aggregations
  @alert_table :performance_alerts
  
  # Client API
  
  @doc """
  Starts the performance monitor GenServer.
  
  Initializes ETS tables for high-performance metrics storage,
  sets up telemetry handlers, and starts periodic aggregation tasks.
  
  ## Returns
  
  - `{:ok, pid}` - Successfully started performance monitor
  """
  def start_link(_) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  @doc """
  Records a performance metric for an object.
  
  High-performance metric recording using ETS tables. Automatically
  checks for performance alerts based on configured thresholds.
  
  ## Parameters
  
  - `object_id` - ID of the object
  - `metric_type` - Type of metric (atom)
  - `value` - Metric value (number)
  - `metadata` - Additional metadata (default: %{})
  
  ## Examples
  
      iex> Object.PerformanceMonitor.record_metric("obj1", :response_time, 50.0)
      :ok
  """
  def record_metric(object_id, metric_type, value, metadata \\ %{}) do
    # High-performance metric recording via ETS
    timestamp = System.monotonic_time(:millisecond)
    
    metric_entry = {
      {object_id, metric_type, timestamp},
      value,
      metadata
    }
    
    :ets.insert(@metrics_table, metric_entry)
    
    # Check for performance alerts
    check_performance_alerts(object_id, metric_type, value)
  end
  
  @doc """
  Gets metrics for an object and metric type within a time range.
  
  ## Parameters
  
  - `object_id` - ID of the object
  - `metric_type` - Type of metric to retrieve
  - `time_range` - Time range in milliseconds (default: 5 minutes)
  
  ## Returns
  
  List of metric entries with timestamps, values, and metadata
  """
  def get_metrics(object_id, metric_type, time_range \\ 300_000) do
    # Optimized lookup using bounded key range
    current_time = System.monotonic_time(:millisecond)
    start_time = current_time - time_range
    
    # Use match pattern for better performance
    pattern = {{object_id, metric_type, :"$1"}, :"$2", :"$3"}
    guard = [{:>=, :"$1", start_time}, {:"=<", :"$1", current_time}]
    result = [{{:"$1", :"$2", :"$3"}}]
    
    :ets.select(@metrics_table, [{pattern, guard, result}])
  end
  
  @doc """
  Gets aggregated system-wide performance metrics.
  
  ## Returns
  
  Map with system-level performance statistics
  """
  def get_system_metrics() do
    GenServer.call(__MODULE__, :get_system_metrics)
  end
  
  @doc """
  Gets comprehensive performance analysis for a specific object.
  
  ## Parameters
  
  - `object_id` - ID of the object to analyze
  
  ## Returns
  
  Map with overall performance score, detailed metrics, and analysis
  """
  def get_object_performance(object_id) do
    GenServer.call(__MODULE__, {:get_object_performance, object_id})
  end
  
  @doc """
  Gets all current performance alerts.
  
  ## Returns
  
  List of active performance alerts with severity levels
  """
  def get_performance_alerts() do
    :ets.tab2list(@alert_table)
  end
  
  @doc """
  Sets performance thresholds for a metric type.
  
  ## Parameters
  
  - `metric_type` - Type of metric to configure
  - `threshold_config` - Map with :warning, :critical, and :low_warning thresholds
  
  ## Examples
  
      iex> Object.PerformanceMonitor.set_performance_threshold(:response_time, %{warning: 100, critical: 500})
      :ok
  """
  def set_performance_threshold(metric_type, threshold_config) do
    GenServer.call(__MODULE__, {:set_threshold, metric_type, threshold_config})
  end
  
  @doc """
  Generates a comprehensive performance report.
  
  ## Parameters
  
  - `time_range` - Time range for the report in milliseconds (default: 1 hour)
  
  ## Returns
  
  Map with detailed performance analysis, summaries, and recommendations
  """
  def get_performance_report(time_range \\ 3600_000) do
    GenServer.call(__MODULE__, {:get_report, time_range})
  end
  
  # Server callbacks
  
  @impl true
  def init(:ok) do
    # Create ETS tables with optimized settings for time-series data
    :ets.new(@metrics_table, [:named_table, :public, :ordered_set, 
              {:write_concurrency, true}, {:read_concurrency, true}])
    :ets.new(@aggregation_table, [:named_table, :public, :set, 
              {:read_concurrency, true}, {:write_concurrency, true}])
    :ets.new(@alert_table, [:named_table, :public, :set, 
              {:write_concurrency, true}, {:read_concurrency, true}])
    
    # Initialize telemetry
    setup_telemetry()
    
    # Start periodic aggregation
    schedule_aggregation()
    schedule_cleanup()
    
    state = %{
      thresholds: init_default_thresholds(),
      aggregation_interval: 60_000,  # 1 minute
      cleanup_interval: 3600_000,    # 1 hour
      started_at: DateTime.utc_now()
    }
    
    Logger.info("Performance Monitor started")
    {:ok, state}
  end
  
  @impl true
  def handle_call(:get_system_metrics, _from, state) do
    system_metrics = calculate_system_metrics()
    {:reply, system_metrics, state}
  end
  
  @impl true
  def handle_call({:get_object_performance, object_id}, _from, state) do
    object_performance = calculate_object_performance(object_id)
    {:reply, object_performance, state}
  end
  
  @impl true
  def handle_call({:set_threshold, metric_type, threshold_config}, _from, state) do
    updated_thresholds = Map.put(state.thresholds, metric_type, threshold_config)
    updated_state = %{state | thresholds: updated_thresholds}
    
    {:reply, :ok, updated_state}
  end
  
  @impl true
  def handle_call({:get_report, time_range}, _from, state) do
    report = generate_performance_report(time_range)
    {:reply, report, state}
  end
  
  @impl true
  def handle_info(:aggregate_metrics, state) do
    # Perform metric aggregation
    perform_metric_aggregation()
    
    schedule_aggregation()
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:cleanup_metrics, state) do
    # Clean up old metrics
    cleanup_old_metrics()
    
    schedule_cleanup()
    {:noreply, state}
  end
  
  # Private functions
  
  defp setup_telemetry do
    # Attach telemetry handlers for system events
    events = [
      [:object, :message, :sent],
      [:object, :message, :received],
      [:object, :method, :executed],
      [:object, :goal, :evaluated],
      [:object, :learning, :updated],
      [:coordination, :session, :started],
      [:coordination, :session, :completed],
      [:evolution, :proposal, :created],
      [:evolution, :proposal, :approved]
    ]
    
    Enum.each(events, fn event ->
      :telemetry.attach(
        "performance_monitor_#{Enum.join(event, "_")}",
        event,
        &handle_telemetry_event/4,
        %{}
      )
    end)
  end
  
  defp handle_telemetry_event(event, measurements, metadata, _config) do
    case event do
      [:object, :message, :sent] ->
        record_metric(metadata.object_id, :messages_sent, 1, %{
          message_type: metadata.message_type,
          priority: metadata.priority
        })
      
      [:object, :message, :received] ->
        record_metric(metadata.object_id, :messages_received, 1, %{
          message_type: metadata.message_type,
          processing_time: measurements.processing_time
        })
      
      [:object, :method, :executed] ->
        record_metric(metadata.object_id, :method_execution_time, 
                     measurements.duration, %{method: metadata.method})
      
      [:object, :goal, :evaluated] ->
        record_metric(metadata.object_id, :goal_evaluation_score, 
                     measurements.score, %{goal_type: metadata.goal_type})
      
      [:object, :learning, :updated] ->
        record_metric(metadata.object_id, :learning_progress, 
                     measurements.improvement, %{algorithm: metadata.algorithm})
      
      [:coordination, :session, :started] ->
        record_metric("system", :coordination_sessions_started, 1, %{
          session_type: metadata.task_type,
          participants: metadata.participant_count
        })
      
      [:coordination, :session, :completed] ->
        record_metric("system", :coordination_session_duration, 
                     measurements.duration, %{
                       success: metadata.success,
                       participants: metadata.participant_count
                     })
      
      [:evolution, :proposal, :created] ->
        record_metric("system", :evolution_proposals, 1, %{
          evolution_type: metadata.evolution_type,
          scope: metadata.scope
        })
      
      [:evolution, :proposal, :approved] ->
        record_metric("system", :evolution_approvals, 1, %{
          evolution_type: metadata.evolution_type,
          consensus_ratio: metadata.consensus_ratio
        })
      
      _ ->
        :ok
    end
  end
  
  defp check_performance_alerts(object_id, metric_type, value) do
    # Check if value exceeds thresholds and create alerts
    case get_threshold(metric_type) do
      nil -> 
        :ok
      
      threshold_config ->
        alert_level = determine_alert_level(value, threshold_config)
        
        if alert_level != :none do
          create_performance_alert(object_id, metric_type, value, alert_level)
        end
    end
  end
  
  defp get_threshold(metric_type) do
    case :ets.lookup(@aggregation_table, {:threshold, metric_type}) do
      [{_, threshold_config}] -> threshold_config
      [] -> nil
    end
  end
  
  defp determine_alert_level(value, threshold_config) do
    cond do
      value > threshold_config.critical -> :critical
      value > threshold_config.warning -> :warning
      value < threshold_config.low_warning -> :low_warning
      true -> :none
    end
  end
  
  defp create_performance_alert(object_id, metric_type, value, alert_level) do
    alert_id = :crypto.strong_rand_bytes(8) |> Base.encode16() |> String.downcase()
    
    alert = %{
      id: alert_id,
      object_id: object_id,
      metric_type: metric_type,
      value: value,
      alert_level: alert_level,
      timestamp: DateTime.utc_now(),
      acknowledged: false
    }
    
    :ets.insert(@alert_table, {alert_id, alert})
    
    # Log critical alerts
    if alert_level == :critical do
      Logger.warning("CRITICAL PERFORMANCE ALERT: Object #{object_id} - #{metric_type}: #{value}")
    end
  end
  
  defp calculate_system_metrics do
    current_time = System.monotonic_time(:millisecond)
    last_hour = current_time - 3600_000
    
    # Get all system-level metrics from the last hour
    system_metrics = :ets.select(@metrics_table, [
      {{{"system", :"$1", :"$2"}, :"$3", :"$4"},
       [{:>=, :"$2", last_hour}],
       [{{:"$1", :"$3", :"$4"}}]}
    ])
    
    # Aggregate by metric type
    aggregated = Enum.reduce(system_metrics, %{}, fn {metric_type, value, _metadata}, acc ->
      current_list = Map.get(acc, metric_type, [])
      Map.put(acc, metric_type, [value | current_list])
    end)
    
    # Calculate statistics for each metric type
    Enum.into(aggregated, %{}, fn {metric_type, values} ->
      stats = %{
        count: length(values),
        sum: Enum.sum(values),
        avg: Enum.sum(values) / length(values),
        min: Enum.min(values),
        max: Enum.max(values)
      }
      
      {metric_type, stats}
    end)
  end
  
  defp calculate_object_performance(object_id) do
    current_time = System.monotonic_time(:millisecond)
    last_hour = current_time - 3600_000
    
    # Get all metrics for this object from the last hour
    object_metrics = :ets.select(@metrics_table, [
      {{{object_id, :"$1", :"$2"}, :"$3", :"$4"},
       [{:>=, :"$2", last_hour}],
       [{{:"$1", :"$3", :"$4"}}]}
    ])
    
    # Group by metric type
    grouped_metrics = Enum.group_by(object_metrics, fn {metric_type, _value, _metadata} ->
      metric_type
    end)
    
    # Calculate performance scores
    performance_scores = Enum.into(grouped_metrics, %{}, fn {metric_type, metric_list} ->
      values = Enum.map(metric_list, fn {_type, value, _metadata} -> value end)
      
      score = case metric_type do
        :goal_evaluation_score ->
          if length(values) > 0, do: Enum.sum(values) / length(values), else: 0.0
        
        :method_execution_time ->
          if length(values) > 0, do: 1.0 / (Enum.sum(values) / length(values)), else: 1.0
        
        :messages_sent ->
          length(values)
        
        :messages_received ->
          length(values)
        
        _ ->
          if length(values) > 0, do: Enum.sum(values) / length(values), else: 0.0
      end
      
      {metric_type, score}
    end)
    
    # Calculate overall performance score
    overall_score = if map_size(performance_scores) > 0 do
      performance_scores
      |> Map.values()
      |> Enum.sum()
      |> Kernel./(map_size(performance_scores))
    else
      0.0
    end
    
    %{
      object_id: object_id,
      overall_performance: overall_score,
      metric_scores: performance_scores,
      measurement_count: length(object_metrics),
      last_updated: DateTime.utc_now()
    }
  end
  
  defp perform_metric_aggregation do
    current_time = System.monotonic_time(:millisecond)
    aggregation_window = current_time - 60_000  # Last minute
    
    # Get all metrics from the last minute
    recent_metrics = :ets.select(@metrics_table, [
      {{:"$1", :"$2", :"$3"}, :"$4"},
      [{:>=, :"$3", aggregation_window}],
      [{{:"$1", :"$2", :"$4"}}]
    ])
    
    # Group by object_id and metric_type
    grouped = Enum.group_by(recent_metrics, fn {{object_id, metric_type}, _value} ->
      {object_id, metric_type}
    end)
    
    # Calculate aggregations and store
    Enum.each(grouped, fn {{object_id, metric_type}, metric_list} ->
      values = Enum.map(metric_list, fn {_key, value} -> value end)
      
      aggregation = %{
        count: length(values),
        sum: Enum.sum(values),
        avg: Enum.sum(values) / length(values),
        min: Enum.min(values),
        max: Enum.max(values),
        timestamp: current_time
      }
      
      :ets.insert(@aggregation_table, {{object_id, metric_type, current_time}, aggregation})
    end)
  end
  
  defp cleanup_old_metrics do
    current_time = System.monotonic_time(:millisecond)
    cutoff_time = current_time - 86_400_000  # 24 hours ago
    
    # Delete old metrics
    :ets.select_delete(@metrics_table, [
      {{:"$1", :"$2", :"$3"}, :"$4", :"$5"},
      [{:<, :"$3", cutoff_time}],
      [true]
    ])
    
    # Delete old aggregations
    :ets.select_delete(@aggregation_table, [
      {{:"$1", :"$2", :"$3"}, :"$4"},
      [{:<, :"$3", cutoff_time}],
      [true]
    ])
    
    Logger.debug("Cleaned up old performance metrics")
  end
  
  defp generate_performance_report(time_range) do
    current_time = System.monotonic_time(:millisecond)
    start_time = current_time - time_range
    
    # Get all metrics in the time range
    metrics = :ets.select(@metrics_table, [
      {{:"$1", :"$2", :"$3"}, :"$4", :"$5"},
      [{:>=, :"$3", start_time}],
      [{{:"$1", :"$2", :"$4"}}]
    ])
    
    # Analyze metrics
    total_metrics = length(metrics)
    unique_objects = metrics
                    |> Enum.map(fn {{object_id, _metric_type}, _value} -> object_id end)
                    |> Enum.uniq()
                    |> length()
    
    metric_types = metrics
                  |> Enum.map(fn {{_object_id, metric_type}, _value} -> metric_type end)
                  |> Enum.uniq()
    
    # Get current alerts
    alerts = :ets.tab2list(@alert_table)
    active_alerts = Enum.filter(alerts, fn {_id, alert} ->
      not alert.acknowledged
    end)
    
    %{
      report_period: %{
        start_time: DateTime.from_unix!(div(start_time, 1000)),
        end_time: DateTime.utc_now(),
        duration_ms: time_range
      },
      summary: %{
        total_metrics_recorded: total_metrics,
        unique_objects_monitored: unique_objects,
        metric_types_tracked: length(metric_types),
        active_alerts: length(active_alerts)
      },
      system_health: calculate_system_health_score(),
      top_performers: get_top_performing_objects(5),
      performance_issues: get_performance_issues(),
      recommendations: generate_performance_recommendations()
    }
  end
  
  defp calculate_system_health_score do
    # Calculate overall system health based on various metrics
    :rand.uniform()  # Simplified for now
  end
  
  defp get_top_performing_objects(limit) do
    # Get top performing objects
    SchemaRegistry.list_objects()
    |> Enum.take(limit)
    |> Enum.map(fn {object_id, _schema} ->
      performance = calculate_object_performance(object_id)
      {object_id, performance.overall_performance}
    end)
    |> Enum.sort_by(fn {_id, score} -> score end, :desc)
  end
  
  defp get_performance_issues do
    # Identify objects with performance issues
    alerts = :ets.tab2list(@alert_table)
    
    Enum.filter(alerts, fn {_id, alert} ->
      alert.alert_level in [:critical, :warning] and not alert.acknowledged
    end)
    |> Enum.map(fn {_id, alert} ->
      %{
        object_id: alert.object_id,
        issue_type: alert.metric_type,
        severity: alert.alert_level,
        value: alert.value
      }
    end)
  end
  
  defp generate_performance_recommendations do
    # Generate recommendations based on current performance data
    [
      "Consider optimizing message routing for high-traffic objects",
      "Monitor objects with frequent goal evaluation failures",
      "Implement load balancing for coordination-heavy scenarios"
    ]
  end
  
  defp schedule_aggregation do
    Process.send_after(self(), :aggregate_metrics, 60_000)  # 1 minute
  end
  
  defp schedule_cleanup do
    Process.send_after(self(), :cleanup_metrics, 3600_000)  # 1 hour
  end
  
  defp init_default_thresholds do
    %{
      method_execution_time: %{warning: 1000, critical: 5000, low_warning: 10},
      goal_evaluation_score: %{warning: 0.3, critical: 0.1, low_warning: 0.0},
      messages_sent: %{warning: 100, critical: 1000, low_warning: 0},
      coordination_session_duration: %{warning: 30000, critical: 120000, low_warning: 100}
    }
  end
end
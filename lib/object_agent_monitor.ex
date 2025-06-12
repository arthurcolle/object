defmodule Object.AgentMonitor do
  @moduledoc """
  Advanced agent monitoring system that tracks agent behavior, performance, 
  and coordination patterns across the distributed object system.
  """
  
  use GenServer
  require Logger
  
  alias Object.PerformanceMonitor
  alias Object.SchemaRegistry
  
  @agent_state_table :agent_states
  @agent_metrics_table :agent_metrics
  @agent_behaviors_table :agent_behaviors
  
  defstruct [
    :agent_id,
    :start_time,
    :last_activity,
    :message_count,
    :goal_completions,
    :coordination_sessions,
    :performance_score,
    :status,
    :behavior_patterns,
    :resource_usage
  ]
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def register_agent(agent_id, initial_state \\ %{}) do
    GenServer.call(__MODULE__, {:register_agent, agent_id, initial_state})
  end
  
  def update_agent_activity(agent_id, activity_type, metadata \\ %{}) do
    GenServer.cast(__MODULE__, {:update_activity, agent_id, activity_type, metadata})
  end
  
  def get_agent_status(agent_id) do
    GenServer.call(__MODULE__, {:get_agent_status, agent_id})
  end
  
  def get_all_agents do
    GenServer.call(__MODULE__, :get_all_agents)
  end
  
  def get_agent_metrics(agent_id, time_window \\ 300_000) do
    GenServer.call(__MODULE__, {:get_agent_metrics, agent_id, time_window})
  end
  
  def get_coordination_patterns do
    GenServer.call(__MODULE__, :get_coordination_patterns)
  end
  
  def get_system_health_dashboard do
    GenServer.call(__MODULE__, :get_system_health_dashboard)
  end
  
  def set_agent_alert_threshold(metric_type, threshold_config) do
    GenServer.call(__MODULE__, {:set_alert_threshold, metric_type, threshold_config})
  end
  
  @impl true
  def init(opts) do
    # Ensure PerformanceMonitor is available
    case GenServer.whereis(Object.PerformanceMonitor) do
      nil ->
        Logger.warning("PerformanceMonitor not available, agent alerts may not be recorded")
      _ ->
        :ok
    end
    
    :ets.new(@agent_state_table, [:named_table, :public, :set, 
              {:read_concurrency, true}, {:write_concurrency, true}])
    :ets.new(@agent_metrics_table, [:named_table, :public, :ordered_set,
              {:read_concurrency, true}, {:write_concurrency, true}])
    :ets.new(@agent_behaviors_table, [:named_table, :public, :bag,
              {:read_concurrency, true}, {:write_concurrency, true}])
    
    setup_agent_telemetry()
    schedule_health_check()
    schedule_behavior_analysis()
    
    state = %{
      alert_thresholds: init_default_alert_thresholds(),
      monitoring_interval: Keyword.get(opts, :monitoring_interval, 30_000),
      behavior_analysis_interval: Keyword.get(opts, :behavior_analysis_interval, 300_000),
      started_at: DateTime.utc_now()
    }
    
    Logger.info("Agent Monitor started")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:register_agent, agent_id, initial_state}, _from, state) do
    agent_record = %__MODULE__{
      agent_id: agent_id,
      start_time: DateTime.utc_now(),
      last_activity: DateTime.utc_now(),
      message_count: 0,
      goal_completions: 0,
      coordination_sessions: 0,
      performance_score: 1.0,
      status: :active,
      behavior_patterns: %{},
      resource_usage: Map.merge(%{cpu: 0.0, memory: 0.0, network: 0.0}, initial_state)
    }
    
    :ets.insert(@agent_state_table, {agent_id, agent_record})
    
    Logger.info("Registered agent #{agent_id} for monitoring")
    {:reply, :ok, state}
  end
  
  @impl true
  def handle_call({:get_agent_status, agent_id}, _from, state) do
    case :ets.lookup(@agent_state_table, agent_id) do
      [{^agent_id, agent_record}] ->
        status = %{
          agent_id: agent_record.agent_id,
          status: agent_record.status,
          uptime: DateTime.diff(DateTime.utc_now(), agent_record.start_time, :second),
          last_activity: agent_record.last_activity,
          performance_score: agent_record.performance_score,
          message_count: agent_record.message_count,
          goal_completions: agent_record.goal_completions,
          coordination_sessions: agent_record.coordination_sessions,
          resource_usage: agent_record.resource_usage
        }
        {:reply, {:ok, status}, state}
      
      [] ->
        {:reply, {:error, :agent_not_found}, state}
    end
  end
  
  @impl true
  def handle_call(:get_all_agents, _from, state) do
    agents = :ets.tab2list(@agent_state_table)
              |> Enum.map(fn {agent_id, agent_record} ->
                %{
                  agent_id: agent_id,
                  status: agent_record.status,
                  performance_score: agent_record.performance_score,
                  last_activity: agent_record.last_activity,
                  resource_usage: agent_record.resource_usage
                }
              end)
    
    {:reply, agents, state}
  end
  
  @impl true
  def handle_call({:get_agent_metrics, agent_id, time_window}, _from, state) do
    current_time = System.monotonic_time(:millisecond)
    start_time = current_time - time_window
    
    metrics = :ets.select(@agent_metrics_table, [
      {{{agent_id, :"$1", :"$2"}, :"$3", :"$4"},
       [{:>=, :"$2", start_time}],
       [{{:"$1", :"$2", :"$3", :"$4"}}]}
    ])
    
    grouped_metrics = Enum.group_by(metrics, fn {metric_type, _timestamp, _value, _metadata} ->
      metric_type
    end)
    
    aggregated = Enum.into(grouped_metrics, %{}, fn {metric_type, metric_list} ->
      values = Enum.map(metric_list, fn {_type, _time, value, _meta} -> value end)
      
      stats = %{
        count: length(values),
        avg: (if length(values) > 0, do: Enum.sum(values) / length(values), else: 0),
        min: (if length(values) > 0, do: Enum.min(values), else: 0),
        max: (if length(values) > 0, do: Enum.max(values), else: 0),
        latest: List.first(Enum.sort(metric_list, fn {_, t1, _, _}, {_, t2, _, _} -> t1 >= t2 end))
      }
      
      {metric_type, stats}
    end)
    
    {:reply, aggregated, state}
  end
  
  @impl true
  def handle_call(:get_coordination_patterns, _from, state) do
    patterns = analyze_coordination_patterns()
    {:reply, patterns, state}
  end
  
  @impl true
  def handle_call(:get_system_health_dashboard, _from, state) do
    dashboard = generate_system_health_dashboard()
    {:reply, dashboard, state}
  end
  
  @impl true
  def handle_call({:set_alert_threshold, metric_type, threshold_config}, _from, state) do
    updated_thresholds = Map.put(state.alert_thresholds, metric_type, threshold_config)
    updated_state = %{state | alert_thresholds: updated_thresholds}
    
    {:reply, :ok, updated_state}
  end
  
  @impl true
  def handle_cast({:update_activity, agent_id, activity_type, metadata}, state) do
    case :ets.lookup(@agent_state_table, agent_id) do
      [{^agent_id, agent_record}] ->
        updated_record = update_agent_record(agent_record, activity_type, metadata)
        :ets.insert(@agent_state_table, {agent_id, updated_record})
        
        record_agent_metric(agent_id, activity_type, metadata)
        record_behavior_pattern(agent_id, activity_type, metadata)
        
        check_agent_alerts(agent_id, updated_record, state.alert_thresholds)
      
      [] ->
        Logger.warning("Attempted to update activity for unregistered agent: #{agent_id}")
    end
    
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:health_check, state) do
    perform_agent_health_checks()
    schedule_health_check()
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:behavior_analysis, state) do
    perform_behavior_analysis()
    schedule_behavior_analysis()
    {:noreply, state}
  end
  
  defp setup_agent_telemetry do
    events = [
      [:agent, :message, :sent],
      [:agent, :message, :received],
      [:agent, :goal, :completed],
      [:agent, :goal, :failed],
      [:agent, :coordination, :joined],
      [:agent, :coordination, :left],
      [:agent, :resource, :updated],
      [:agent, :performance, :degraded],
      [:agent, :error, :occurred]
    ]
    
    Enum.each(events, fn event ->
      :telemetry.attach(
        "agent_monitor_#{Enum.join(event, "_")}",
        event,
        &__MODULE__.handle_agent_telemetry/4,
        %{}
      )
    end)
  end
  
  def handle_agent_telemetry(event, measurements, metadata, _config) do
    agent_id = Map.get(metadata, :agent_id, "unknown")
    
    case event do
      [:agent, :message, :sent] ->
        update_agent_activity(agent_id, :message_sent, %{
          recipient: metadata.recipient,
          message_type: metadata.message_type,
          size: measurements.size
        })
      
      [:agent, :message, :received] ->
        update_agent_activity(agent_id, :message_received, %{
          sender: metadata.sender,
          message_type: metadata.message_type,
          processing_time: measurements.processing_time
        })
      
      [:agent, :goal, :completed] ->
        update_agent_activity(agent_id, :goal_completed, %{
          goal_type: metadata.goal_type,
          duration: measurements.duration,
          success_score: measurements.success_score
        })
      
      [:agent, :goal, :failed] ->
        update_agent_activity(agent_id, :goal_failed, %{
          goal_type: metadata.goal_type,
          failure_reason: metadata.failure_reason
        })
      
      [:agent, :coordination, :joined] ->
        update_agent_activity(agent_id, :coordination_joined, %{
          session_id: metadata.session_id,
          role: metadata.role
        })
      
      [:agent, :coordination, :left] ->
        update_agent_activity(agent_id, :coordination_left, %{
          session_id: metadata.session_id,
          duration: measurements.duration
        })
      
      [:agent, :resource, :updated] ->
        update_agent_activity(agent_id, :resource_update, %{
          cpu: measurements.cpu,
          memory: measurements.memory,
          network: measurements.network
        })
      
      [:agent, :performance, :degraded] ->
        update_agent_activity(agent_id, :performance_degraded, %{
          metric_type: metadata.metric_type,
          current_value: measurements.current_value,
          threshold: metadata.threshold
        })
      
      [:agent, :error, :occurred] ->
        update_agent_activity(agent_id, :error_occurred, %{
          error_type: metadata.error_type,
          severity: metadata.severity,
          context: metadata.context
        })
      
      _ ->
        :ok
    end
  end
  
  defp update_agent_record(agent_record, activity_type, metadata) do
    updated_record = %{agent_record | last_activity: DateTime.utc_now()}
    
    case activity_type do
      :message_sent ->
        %{updated_record | message_count: updated_record.message_count + 1}
      
      :message_received ->
        %{updated_record | message_count: updated_record.message_count + 1}
      
      :goal_completed ->
        %{updated_record | goal_completions: updated_record.goal_completions + 1}
      
      :coordination_joined ->
        %{updated_record | coordination_sessions: updated_record.coordination_sessions + 1}
      
      :resource_update ->
        updated_resource_usage = Map.merge(updated_record.resource_usage, %{
          cpu: Map.get(metadata, :cpu, updated_record.resource_usage.cpu),
          memory: Map.get(metadata, :memory, updated_record.resource_usage.memory),
          network: Map.get(metadata, :network, updated_record.resource_usage.network)
        })
        %{updated_record | resource_usage: updated_resource_usage}
      
      :performance_degraded ->
        performance_impact = calculate_performance_impact(metadata)
        new_score = max(0.0, updated_record.performance_score - performance_impact)
        %{updated_record | performance_score: new_score}
      
      :error_occurred ->
        error_impact = calculate_error_impact(metadata)
        new_score = max(0.0, updated_record.performance_score - error_impact)
        status = if new_score < 0.3, do: :degraded, else: updated_record.status
        %{updated_record | performance_score: new_score, status: status}
      
      _ ->
        updated_record
    end
  end
  
  defp record_agent_metric(agent_id, activity_type, metadata) do
    timestamp = System.monotonic_time(:millisecond)
    
    case activity_type do
      :message_sent ->
        :ets.insert(@agent_metrics_table, {{agent_id, :messages_sent, timestamp}, 1, metadata})
      
      :message_received ->
        processing_time = Map.get(metadata, :processing_time, 0)
        :ets.insert(@agent_metrics_table, {{agent_id, :message_processing_time, timestamp}, processing_time, metadata})
      
      :goal_completed ->
        duration = Map.get(metadata, :duration, 0)
        success_score = Map.get(metadata, :success_score, 1.0)
        :ets.insert(@agent_metrics_table, {{agent_id, :goal_completion_time, timestamp}, duration, metadata})
        :ets.insert(@agent_metrics_table, {{agent_id, :goal_success_score, timestamp}, success_score, metadata})
      
      :resource_update ->
        Enum.each([:cpu, :memory, :network], fn resource_type ->
          if Map.has_key?(metadata, resource_type) do
            value = Map.get(metadata, resource_type)
            :ets.insert(@agent_metrics_table, {{agent_id, resource_type, timestamp}, value, metadata})
          end
        end)
      
      _ ->
        :ets.insert(@agent_metrics_table, {{agent_id, activity_type, timestamp}, 1, metadata})
    end
  end
  
  defp record_behavior_pattern(agent_id, activity_type, metadata) do
    timestamp = DateTime.utc_now()
    
    behavior_entry = %{
      activity_type: activity_type,
      timestamp: timestamp,
      metadata: metadata,
      context: extract_behavior_context(metadata)
    }
    
    :ets.insert(@agent_behaviors_table, {agent_id, behavior_entry})
  end
  
  defp extract_behavior_context(metadata) do
    %{
      time_of_day: DateTime.utc_now() |> DateTime.to_time() |> Time.to_string(),
      interaction_pattern: determine_interaction_pattern(metadata),
      complexity_level: determine_complexity_level(metadata)
    }
  end
  
  defp determine_interaction_pattern(metadata) do
    cond do
      Map.has_key?(metadata, :recipient) or Map.has_key?(metadata, :sender) ->
        :peer_to_peer
      
      Map.has_key?(metadata, :session_id) ->
        :group_coordination
      
      Map.has_key?(metadata, :goal_type) ->
        :goal_oriented
      
      true ->
        :system_maintenance
    end
  end
  
  defp determine_complexity_level(metadata) do
    complexity_indicators = [
      Map.has_key?(metadata, :goal_type),
      Map.has_key?(metadata, :session_id),
      Map.has_key?(metadata, :coordination_type),
      Map.get(metadata, :processing_time, 0) > 1000
    ]
    
    case Enum.count(complexity_indicators, & &1) do
      0 -> :simple
      1 -> :moderate
      2 -> :complex
      _ -> :highly_complex
    end
  end
  
  defp calculate_performance_impact(metadata) do
    base_impact = 0.05
    
    severity_multiplier = case Map.get(metadata, :severity, :low) do
      :critical -> 3.0
      :high -> 2.0
      :medium -> 1.5
      :low -> 1.0
    end
    
    base_impact * severity_multiplier
  end
  
  defp calculate_error_impact(metadata) do
    base_impact = 0.1
    
    severity_multiplier = case Map.get(metadata, :severity, :low) do
      :critical -> 4.0
      :high -> 2.5
      :medium -> 1.5
      :low -> 1.0
    end
    
    base_impact * severity_multiplier
  end
  
  defp check_agent_alerts(agent_id, agent_record, alert_thresholds) do
    Enum.each(alert_thresholds, fn {metric_type, threshold_config} ->
      case get_agent_metric_value(agent_record, metric_type) do
        nil ->
          :ok
        
        value ->
          alert_level = determine_alert_level(value, threshold_config)
          
          if alert_level != :none do
            create_agent_alert(agent_id, metric_type, value, alert_level)
          end
      end
    end)
  end
  
  defp get_agent_metric_value(agent_record, metric_type) do
    case metric_type do
      :performance_score -> agent_record.performance_score
      :message_rate -> calculate_message_rate(agent_record)
      :goal_success_rate -> calculate_goal_success_rate(agent_record)
      :resource_usage_cpu -> agent_record.resource_usage.cpu
      :resource_usage_memory -> agent_record.resource_usage.memory
      _ -> nil
    end
  end
  
  defp calculate_message_rate(agent_record) do
    uptime_seconds = DateTime.diff(DateTime.utc_now(), agent_record.start_time, :second)
    if uptime_seconds > 0, do: agent_record.message_count / uptime_seconds, else: 0.0
  end
  
  defp calculate_goal_success_rate(agent_record) do
    if agent_record.goal_completions > 0, do: 1.0, else: 0.0
  end
  
  defp determine_alert_level(value, threshold_config) do
    cond do
      value > Map.get(threshold_config, :critical, :infinity) -> :critical
      value > Map.get(threshold_config, :warning, :infinity) -> :warning
      value < Map.get(threshold_config, :low_warning, 0) -> :low_warning
      true -> :none
    end
  end
  
  defp create_agent_alert(agent_id, metric_type, value, alert_level) do
    Logger.warning("AGENT ALERT [#{alert_level}]: Agent #{agent_id} - #{metric_type}: #{value}")
    
    case GenServer.whereis(Object.PerformanceMonitor) do
      nil ->
        Logger.debug("PerformanceMonitor not available for alert recording")
      
      _pid ->
        PerformanceMonitor.record_metric("agent_monitor", :agent_alerts, 1, %{
          agent_id: agent_id,
          metric_type: metric_type,
          alert_level: alert_level,
          value: value
        })
    end
  end
  
  defp perform_agent_health_checks do
    agents = :ets.tab2list(@agent_state_table)
    current_time = DateTime.utc_now()
    
    Enum.each(agents, fn {agent_id, agent_record} ->
      time_since_activity = DateTime.diff(current_time, agent_record.last_activity, :second)
      
      cond do
        time_since_activity > 3600 ->
          update_agent_status(agent_id, :inactive)
        
        time_since_activity > 300 and agent_record.status != :idle ->
          update_agent_status(agent_id, :idle)
        
        time_since_activity <= 60 and agent_record.status != :active ->
          update_agent_status(agent_id, :active)
        
        true ->
          :ok
      end
    end)
  end
  
  defp update_agent_status(agent_id, new_status) do
    case :ets.lookup(@agent_state_table, agent_id) do
      [{^agent_id, agent_record}] ->
        updated_record = %{agent_record | status: new_status}
        :ets.insert(@agent_state_table, {agent_id, updated_record})
        
        Logger.info("Agent #{agent_id} status changed to #{new_status}")
      
      [] ->
        :ok
    end
  end
  
  defp perform_behavior_analysis do
    agents = :ets.tab2list(@agent_state_table)
    
    Enum.each(agents, fn {agent_id, _agent_record} ->
      behaviors = :ets.lookup(@agent_behaviors_table, agent_id)
      patterns = analyze_agent_behavior_patterns(behaviors)
      
      if length(patterns) > 0 do
        Logger.debug("Agent #{agent_id} behavior patterns: #{inspect(patterns)}")
      end
    end)
  end
  
  defp analyze_agent_behavior_patterns(behaviors) do
    behaviors
    |> Enum.map(fn {_agent_id, behavior_entry} -> behavior_entry end)
    |> Enum.group_by(fn behavior -> behavior.activity_type end)
    |> Enum.map(fn {activity_type, activity_list} ->
      %{
        activity_type: activity_type,
        frequency: length(activity_list),
        avg_complexity: calculate_avg_complexity(activity_list),
        time_distribution: analyze_time_distribution(activity_list)
      }
    end)
  end
  
  defp calculate_avg_complexity(activity_list) do
    complexity_scores = Enum.map(activity_list, fn behavior ->
      case behavior.context.complexity_level do
        :simple -> 1
        :moderate -> 2
        :complex -> 3
        :highly_complex -> 4
      end
    end)
    
    if length(complexity_scores) > 0 do
      Enum.sum(complexity_scores) / length(complexity_scores)
    else
      0.0
    end
  end
  
  defp analyze_time_distribution(activity_list) do
    time_buckets = Enum.group_by(activity_list, fn behavior ->
      hour = behavior.timestamp |> DateTime.to_time() |> Map.get(:hour)
      
      cond do
        hour >= 6 and hour < 12 -> :morning
        hour >= 12 and hour < 18 -> :afternoon
        hour >= 18 and hour < 24 -> :evening
        true -> :night
      end
    end)
    
    Enum.into(time_buckets, %{}, fn {time_period, activities} ->
      {time_period, length(activities)}
    end)
  end
  
  defp analyze_coordination_patterns do
    coordination_behaviors = :ets.select(@agent_behaviors_table, [
      {:"$1", %{activity_type: :"$2", metadata: :"$3"}},
      [{:==, :"$2", :coordination_joined}],
      [{{:"$1", :"$3"}}]
    ])
    
    patterns = coordination_behaviors
               |> Enum.group_by(fn {_agent_id, metadata} ->
                 Map.get(metadata, :session_id)
               end)
               |> Enum.map(fn {session_id, participants} ->
                 %{
                   session_id: session_id,
                   participant_count: length(participants),
                   agents: Enum.map(participants, fn {agent_id, _} -> agent_id end)
                 }
               end)
    
    %{
      total_coordination_sessions: length(patterns),
      avg_participants: if(length(patterns) > 0, do: Enum.sum(Enum.map(patterns, & &1.participant_count)) / length(patterns), else: 0),
      most_active_agents: get_most_coordinating_agents(coordination_behaviors)
    }
  end
  
  defp get_most_coordinating_agents(coordination_behaviors) do
    coordination_behaviors
    |> Enum.map(fn {agent_id, _metadata} -> agent_id end)
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_agent_id, count} -> count end, :desc)
    |> Enum.take(5)
  end
  
  defp generate_system_health_dashboard do
    agents = :ets.tab2list(@agent_state_table)
    total_agents = length(agents)
    
    status_distribution = agents
                         |> Enum.map(fn {_id, record} -> record.status end)
                         |> Enum.frequencies()
    
    avg_performance = if total_agents > 0 do
      agents
      |> Enum.map(fn {_id, record} -> record.performance_score end)
      |> Enum.sum()
      |> Kernel./(total_agents)
    else
      0.0
    end
    
    coordination_patterns = analyze_coordination_patterns()
    
    %{
      timestamp: DateTime.utc_now(),
      total_agents: total_agents,
      status_distribution: status_distribution,
      average_performance_score: avg_performance,
      system_health_score: calculate_system_health_score(status_distribution, avg_performance),
      coordination_activity: coordination_patterns,
      alerts: get_recent_agent_alerts(),
      recommendations: generate_system_recommendations(status_distribution, avg_performance)
    }
  end
  
  defp calculate_system_health_score(status_distribution, avg_performance) do
    active_ratio = Map.get(status_distribution, :active, 0) / max(1, Enum.sum(Map.values(status_distribution)))
    
    health_score = (active_ratio * 0.6) + (avg_performance * 0.4)
    Float.round(health_score, 3)
  end
  
  defp get_recent_agent_alerts do
    current_time = System.monotonic_time(:millisecond)
    last_hour = current_time - 3600_000
    
    PerformanceMonitor.get_metrics("agent_monitor", :agent_alerts, 3600_000)
    |> Enum.take(10)
  end
  
  defp generate_system_recommendations(status_distribution, avg_performance) do
    recommendations = []
    
    recommendations = if Map.get(status_distribution, :inactive, 0) > 0 do
      ["Consider investigating inactive agents" | recommendations]
    else
      recommendations
    end
    
    recommendations = if avg_performance < 0.7 do
      ["System performance below optimal - review agent configurations" | recommendations]
    else
      recommendations
    end
    
    recommendations = if Map.get(status_distribution, :active, 0) < Map.get(status_distribution, :idle, 0) do
      ["More agents are idle than active - consider load balancing" | recommendations]
    else
      recommendations
    end
    
    if length(recommendations) == 0 do
      ["System operating normally"]
    else
      recommendations
    end
  end
  
  defp schedule_health_check do
    Process.send_after(self(), :health_check, 30_000)
  end
  
  defp schedule_behavior_analysis do
    Process.send_after(self(), :behavior_analysis, 300_000)
  end
  
  defp init_default_alert_thresholds do
    %{
      performance_score: %{warning: 0.5, critical: 0.3, low_warning: 0.0},
      message_rate: %{warning: 10.0, critical: 50.0, low_warning: 0.0},
      resource_usage_cpu: %{warning: 80.0, critical: 95.0, low_warning: 0.0},
      resource_usage_memory: %{warning: 80.0, critical: 95.0, low_warning: 0.0}
    }
  end
end
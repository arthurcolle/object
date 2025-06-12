defmodule Object.AgentMonitorTest do
  use ExUnit.Case, async: false
  
  alias Object.AgentMonitor
  alias Object.PerformanceMonitor
  
  setup do
    {:ok, _pid} = start_supervised(AgentMonitor)
    :ok
  end
  
  describe "agent registration and status" do
    test "registers an agent successfully" do
      assert :ok = AgentMonitor.register_agent("test_agent_1")
      
      {:ok, status} = AgentMonitor.get_agent_status("test_agent_1")
      assert status.agent_id == "test_agent_1"
      assert status.status == :active
      assert status.performance_score == 1.0
    end
    
    test "registers agent with initial state" do
      initial_state = %{cpu: 25.0, memory: 40.0, network: 10.0}
      assert :ok = AgentMonitor.register_agent("test_agent_2", initial_state)
      
      {:ok, status} = AgentMonitor.get_agent_status("test_agent_2")
      assert status.resource_usage.cpu == 25.0
      assert status.resource_usage.memory == 40.0
      assert status.resource_usage.network == 10.0
    end
    
    test "returns error for non-existent agent" do
      assert {:error, :agent_not_found} = AgentMonitor.get_agent_status("non_existent")
    end
    
    test "lists all registered agents" do
      AgentMonitor.register_agent("agent_1")
      AgentMonitor.register_agent("agent_2")
      
      agents = AgentMonitor.get_all_agents()
      agent_ids = Enum.map(agents, & &1.agent_id)
      
      assert "agent_1" in agent_ids
      assert "agent_2" in agent_ids
      assert length(agents) >= 2
    end
  end
  
  describe "activity tracking" do
    setup do
      AgentMonitor.register_agent("activity_test_agent")
      :ok
    end
    
    test "tracks message sent activity" do
      AgentMonitor.update_agent_activity("activity_test_agent", :message_sent, %{
        recipient: "other_agent",
        message_type: :goal_request,
        size: 1024
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("activity_test_agent")
      assert status.message_count == 1
    end
    
    test "tracks message received activity" do
      AgentMonitor.update_agent_activity("activity_test_agent", :message_received, %{
        sender: "other_agent",
        message_type: :goal_response,
        processing_time: 150
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("activity_test_agent")
      assert status.message_count == 1
    end
    
    test "tracks goal completion activity" do
      AgentMonitor.update_agent_activity("activity_test_agent", :goal_completed, %{
        goal_type: :coordination,
        duration: 5000,
        success_score: 0.85
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("activity_test_agent")
      assert status.goal_completions == 1
    end
    
    test "tracks coordination participation" do
      AgentMonitor.update_agent_activity("activity_test_agent", :coordination_joined, %{
        session_id: "session_123",
        role: :participant
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("activity_test_agent")
      assert status.coordination_sessions == 1
    end
    
    test "updates resource usage" do
      AgentMonitor.update_agent_activity("activity_test_agent", :resource_update, %{
        cpu: 75.0,
        memory: 60.0,
        network: 15.0
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("activity_test_agent")
      assert status.resource_usage.cpu == 75.0
      assert status.resource_usage.memory == 60.0
      assert status.resource_usage.network == 15.0
    end
  end
  
  describe "performance tracking" do
    setup do
      AgentMonitor.register_agent("perf_test_agent")
      :ok
    end
    
    test "degrades performance score on performance issues" do
      AgentMonitor.update_agent_activity("perf_test_agent", :performance_degraded, %{
        metric_type: :response_time,
        current_value: 5000,
        threshold: 1000,
        severity: :high
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("perf_test_agent")
      assert status.performance_score < 1.0
    end
    
    test "updates status to degraded on severe errors" do
      AgentMonitor.update_agent_activity("perf_test_agent", :error_occurred, %{
        error_type: :critical_failure,
        severity: :critical,
        context: "coordination failure"
      })
      
      {:ok, status} = AgentMonitor.get_agent_status("perf_test_agent")
      assert status.performance_score < 0.5
      assert status.status == :degraded
    end
    
    test "maintains performance score above zero" do
      # Apply multiple severe errors
      for _i <- 1..10 do
        AgentMonitor.update_agent_activity("perf_test_agent", :error_occurred, %{
          error_type: :critical_failure,
          severity: :critical,
          context: "multiple failures"
        })
      end
      
      {:ok, status} = AgentMonitor.get_agent_status("perf_test_agent")
      assert status.performance_score >= 0.0
    end
  end
  
  describe "metrics collection" do
    setup do
      AgentMonitor.register_agent("metrics_test_agent")
      
      # Generate some test metrics
      AgentMonitor.update_agent_activity("metrics_test_agent", :message_sent, %{size: 512})
      AgentMonitor.update_agent_activity("metrics_test_agent", :message_received, %{processing_time: 100})
      AgentMonitor.update_agent_activity("metrics_test_agent", :goal_completed, %{
        duration: 2000,
        success_score: 0.9
      })
      
      :ok
    end
    
    test "retrieves agent metrics for time window" do
      metrics = AgentMonitor.get_agent_metrics("metrics_test_agent", 60_000)
      
      assert is_map(metrics)
      assert Map.has_key?(metrics, :messages_sent)
      assert Map.has_key?(metrics, :message_processing_time)
      assert Map.has_key?(metrics, :goal_completion_time)
      assert Map.has_key?(metrics, :goal_success_score)
    end
    
    test "calculates metric statistics correctly" do
      # Add more data points
      for i <- 1..5 do
        AgentMonitor.update_agent_activity("metrics_test_agent", :message_received, %{
          processing_time: i * 50
        })
      end
      
      metrics = AgentMonitor.get_agent_metrics("metrics_test_agent", 60_000)
      processing_time_stats = metrics[:message_processing_time]
      
      assert processing_time_stats.count >= 5
      assert processing_time_stats.min <= processing_time_stats.max
      assert processing_time_stats.avg > 0
    end
  end
  
  describe "coordination pattern analysis" do
    setup do
      # Register multiple agents
      for i <- 1..5 do
        AgentMonitor.register_agent("coord_agent_#{i}")
      end
      
      # Simulate coordination activities
      session_id = "test_session_123"
      
      for i <- 1..3 do
        AgentMonitor.update_agent_activity("coord_agent_#{i}", :coordination_joined, %{
          session_id: session_id,
          role: :participant
        })
      end
      
      :ok
    end
    
    test "analyzes coordination patterns" do
      patterns = AgentMonitor.get_coordination_patterns()
      
      assert is_map(patterns)
      assert Map.has_key?(patterns, :total_coordination_sessions)
      assert Map.has_key?(patterns, :avg_participants)
      assert Map.has_key?(patterns, :most_active_agents)
      
      assert patterns.total_coordination_sessions > 0
    end
  end
  
  describe "system health dashboard" do
    setup do
      # Register agents with different states
      AgentMonitor.register_agent("healthy_agent_1")
      AgentMonitor.register_agent("healthy_agent_2")
      
      # Create some degraded agents
      AgentMonitor.register_agent("degraded_agent")
      AgentMonitor.update_agent_activity("degraded_agent", :error_occurred, %{
        error_type: :critical_failure,
        severity: :critical,
        context: "test failure"
      })
      
      :ok
    end
    
    test "generates comprehensive system health dashboard" do
      dashboard = AgentMonitor.get_system_health_dashboard()
      
      assert is_map(dashboard)
      assert Map.has_key?(dashboard, :timestamp)
      assert Map.has_key?(dashboard, :total_agents)
      assert Map.has_key?(dashboard, :status_distribution)
      assert Map.has_key?(dashboard, :average_performance_score)
      assert Map.has_key?(dashboard, :system_health_score)
      assert Map.has_key?(dashboard, :coordination_activity)
      assert Map.has_key?(dashboard, :recommendations)
      
      assert dashboard.total_agents >= 3
      assert is_float(dashboard.average_performance_score)
      assert is_float(dashboard.system_health_score)
      assert is_list(dashboard.recommendations)
    end
    
    test "calculates system health score correctly" do
      dashboard = AgentMonitor.get_system_health_dashboard()
      
      # Should be between 0.0 and 1.0
      assert dashboard.system_health_score >= 0.0
      assert dashboard.system_health_score <= 1.0
    end
    
    test "provides meaningful recommendations" do
      dashboard = AgentMonitor.get_system_health_dashboard()
      
      assert is_list(dashboard.recommendations)
      assert length(dashboard.recommendations) > 0
      
      # Should contain actual recommendation strings
      Enum.each(dashboard.recommendations, fn recommendation ->
        assert is_binary(recommendation)
        assert String.length(recommendation) > 0
      end)
    end
  end
  
  describe "alert thresholds" do
    setup do
      AgentMonitor.register_agent("alert_test_agent")
      :ok
    end
    
    test "sets custom alert thresholds" do
      threshold_config = %{warning: 0.6, critical: 0.3, low_warning: 0.0}
      
      assert :ok = AgentMonitor.set_agent_alert_threshold(:performance_score, threshold_config)
    end
    
    test "triggers alerts when thresholds exceeded" do
      # Set a low threshold for testing
      threshold_config = %{warning: 0.8, critical: 0.5, low_warning: 0.0}
      AgentMonitor.set_agent_alert_threshold(:performance_score, threshold_config)
      
      # Trigger performance degradation that should cross threshold
      AgentMonitor.update_agent_activity("alert_test_agent", :error_occurred, %{
        error_type: :critical_failure,
        severity: :critical,
        context: "threshold testing"
      })
      
      # The alert should be logged and recorded in performance monitor
      # We can't easily test the logging, but we can verify the agent's state
      {:ok, status} = AgentMonitor.get_agent_status("alert_test_agent")
      assert status.performance_score < 0.8
    end
  end
  
  describe "telemetry integration" do
    setup do
      AgentMonitor.register_agent("telemetry_test_agent")
      :ok
    end
    
    test "handles agent telemetry events" do
      # Emit telemetry events that should be handled by the agent monitor
      :telemetry.execute([:agent, :message, :sent], %{size: 1024}, %{
        agent_id: "telemetry_test_agent",
        recipient: "other_agent",
        message_type: :test_message
      })
      
      :telemetry.execute([:agent, :goal, :completed], %{duration: 1500, success_score: 0.95}, %{
        agent_id: "telemetry_test_agent",
        goal_type: :test_goal
      })
      
      # Allow some time for telemetry processing
      Process.sleep(50)
      
      {:ok, status} = AgentMonitor.get_agent_status("telemetry_test_agent")
      assert status.message_count >= 1
      assert status.goal_completions >= 1
    end
    
    test "handles resource update telemetry" do
      :telemetry.execute([:agent, :resource, :updated], %{cpu: 65.0, memory: 45.0, network: 20.0}, %{
        agent_id: "telemetry_test_agent"
      })
      
      Process.sleep(50)
      
      {:ok, status} = AgentMonitor.get_agent_status("telemetry_test_agent")
      assert status.resource_usage.cpu == 65.0
      assert status.resource_usage.memory == 45.0
      assert status.resource_usage.network == 20.0
    end
  end
  
  describe "edge cases and error handling" do
    test "handles updates for non-existent agents gracefully" do
      # This should not crash, just log a warning
      AgentMonitor.update_agent_activity("non_existent_agent", :message_sent, %{})
      
      # Should still be able to get all agents
      agents = AgentMonitor.get_all_agents()
      assert is_list(agents)
    end
    
    test "handles empty metrics gracefully" do
      AgentMonitor.register_agent("empty_metrics_agent")
      
      metrics = AgentMonitor.get_agent_metrics("empty_metrics_agent", 60_000)
      assert is_map(metrics)
      assert map_size(metrics) == 0
    end
    
    test "handles coordination patterns with no data" do
      patterns = AgentMonitor.get_coordination_patterns()
      
      assert is_map(patterns)
      assert patterns.total_coordination_sessions >= 0
      assert patterns.avg_participants >= 0
      assert is_list(patterns.most_active_agents)
    end
  end
end
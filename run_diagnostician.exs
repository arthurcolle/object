#!/usr/bin/env elixir

# Script to run comprehensive test failure analysis using DSPy-powered diagnostician

defmodule DiagnosticRunner do
  def run do
    IO.puts("🔍 Starting comprehensive test failure analysis with DSPy...")
    
    # Start the AAOS system first
    {:ok, _} = Application.ensure_all_started(:object)
    
    # Wait a moment for services to initialize
    Process.sleep(2000)
    
    # Start the diagnostician
    {:ok, _} = TestFailureDiagnostician.start_link()
    
    # Run tests and capture failures
    IO.puts("📋 Running test suite to capture all failures...")
    
    test_output = capture_test_output()
    
    # Analyze all failures with DSPy
    IO.puts("🤖 Analyzing failures with GPT-4.1-mini via DSPy...")
    
    case TestFailureDiagnostician.analyze_all_failures(test_output) do
      {:ok, comprehensive_analysis} ->
        display_comprehensive_analysis(comprehensive_analysis)
        
        # Generate detailed report
        generate_detailed_report(comprehensive_analysis)
        
      {:error, reason} ->
        IO.puts("❌ Analysis failed: #{inspect(reason)}")
    end
    
    # Get diagnostic stats
    stats = TestFailureDiagnostician.get_diagnostic_stats()
    display_diagnostic_stats(stats)
  end
  
  defp capture_test_output do
    {output, _exit_code} = System.cmd("mix", ["test", "--formatter", "ExUnit.CLIFormatter", "--max-failures", "200"], 
                                      cd: System.cwd(),
                                      stderr_to_stdout: true)
    output
  end
  
  defp display_comprehensive_analysis(analysis) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("🎯 COMPREHENSIVE TEST FAILURE ANALYSIS")
    IO.puts(String.duplicate("=", 80))
    
    IO.puts("📊 SUMMARY:")
    IO.puts("  • Total Failures: #{analysis.total_failures}")
    IO.puts("  • Analysis Time: #{Float.round(analysis.analysis_time_ms, 2)}ms")
    IO.puts("  • Estimated Fix Time: #{analysis.estimated_fix_time}")
    
    IO.puts("\n🏷️  FAILURE CATEGORIES:")
    Enum.each(analysis.failure_categories, fn {category, failures} ->
      IO.puts("  • #{category}: #{length(failures)} failures")
    end)
    
    IO.puts("\n🔬 SYSTEM ANALYSIS:")
    display_system_analysis(analysis.system_analysis)
    
    IO.puts("\n⚡ PRIORITY MATRIX:")
    display_priority_matrix(analysis.priority_matrix)
    
    IO.puts("\n🛠️  FIX STRATEGY:")
    display_fix_strategy(analysis.fix_strategy)
    
    IO.puts("\n📋 TOP 10 INDIVIDUAL FAILURES:")
    analysis.individual_analyses
    |> Enum.take(10)
    |> Enum.with_index(1)
    |> Enum.each(&display_individual_failure/1)
  end
  
  defp display_system_analysis(system_analysis) do
    IO.puts("  🔴 Systemic Issues:")
    (Map.get(system_analysis, :systemic_issues, []) || [])
    |> Enum.each(fn issue -> IO.puts("    - #{issue}") end)
    
    IO.puts("  🏗️  Architectural Concerns:")
    (Map.get(system_analysis, :architectural_concerns, []) || [])
    |> Enum.each(fn concern -> IO.puts("    - #{concern}") end)
    
    IO.puts("  🔄 Common Patterns:")
    (Map.get(system_analysis, :common_patterns, []) || [])
    |> Enum.each(fn pattern -> IO.puts("    - #{pattern}") end)
  end
  
  defp display_priority_matrix(priority_matrix) do
    IO.puts("  🔥 High Priority:")
    (Map.get(priority_matrix, :high_priority_items, []) || [])
    |> Enum.each(fn item -> IO.puts("    - #{item}") end)
    
    IO.puts("  ⚡ Quick Wins:")
    (Map.get(priority_matrix, :quick_wins, []) || [])
    |> Enum.each(fn item -> IO.puts("    - #{item}") end)
    
    IO.puts("  🎯 Strategic Fixes:")
    (Map.get(priority_matrix, :strategic_fixes, []) || [])
    |> Enum.each(fn item -> IO.puts("    - #{item}") end)
  end
  
  defp display_fix_strategy(fix_strategy) do
    IO.puts("  📅 Phase 1 (Immediate):")
    (Map.get(fix_strategy, :phase_1_actions, []) || [])
    |> Enum.each(fn action -> IO.puts("    - #{action}") end)
    
    IO.puts("  📅 Phase 2 (Short-term):")
    (Map.get(fix_strategy, :phase_2_actions, []) || [])
    |> Enum.each(fn action -> IO.puts("    - #{action}") end)
    
    IO.puts("  📅 Phase 3 (Long-term):")
    (Map.get(fix_strategy, :phase_3_actions, []) || [])
    |> Enum.each(fn action -> IO.puts("    - #{action}") end)
    
    IO.puts("  ⏱️  Timeline: #{Map.get(fix_strategy, :timeline, "Not specified")}")
    IO.puts("  📈 Success Metrics:")
    (Map.get(fix_strategy, :success_metrics, []) || [])
    |> Enum.each(fn metric -> IO.puts("    - #{metric}") end)
  end
  
  defp display_individual_failure({analysis, index}) do
    IO.puts("  #{index}. #{Map.get(analysis, :test_name, "Unknown Test")}")
    IO.puts("     Category: #{Map.get(analysis, :error_category, "unknown")}")
    IO.puts("     Severity: #{Map.get(analysis, :severity, "unknown")}")
    IO.puts("     Confidence: #{Float.round(Map.get(analysis, :confidence_score, 0.0), 2)}")
    
    diagnosis = Map.get(analysis, :diagnosis, %{})
    IO.puts("     Issue: #{Map.get(diagnosis, :description, "No description")}")
    
    root_cause = Map.get(analysis, :root_cause, %{})
    IO.puts("     Root Cause: #{Map.get(root_cause, :primary_cause, "Unknown")}")
    
    solution = Map.get(analysis, :solution_strategy, %{})
    immediate_fixes = Map.get(solution, :immediate_fixes, [])
    if is_list(immediate_fixes) and length(immediate_fixes) > 0 do
      IO.puts("     Fix: #{List.first(immediate_fixes)}")
    end
    IO.puts("")
  end
  
  defp display_diagnostic_stats(stats) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("📊 DIAGNOSTIC SYSTEM STATISTICS")
    IO.puts(String.duplicate("=", 80))
    
    IO.puts("  • Total Analyses Performed: #{stats.total_analyses}")
    IO.puts("  • Cache Entries: #{stats.cache_entries}")
    IO.puts("  • Average Analysis Time: #{Float.round(stats.average_analysis_time, 2)}ms")
    
    IO.puts("\n🔥 Most Common Failure Types:")
    Enum.each(stats.most_common_failures, fn {type, count} ->
      IO.puts("  • #{type}: #{count} occurrences")
    end)
  end
  
  defp generate_detailed_report(comprehensive_analysis) do
    timestamp = DateTime.utc_now() |> DateTime.to_string()
    filename = "test_failure_analysis_#{String.replace(timestamp, ~r/[:\s]/, "_")}.md"
    
    content = """
    # Test Failure Analysis Report
    
    **Generated**: #{timestamp}
    **Analysis Duration**: #{Float.round(comprehensive_analysis.analysis_time_ms, 2)}ms
    **Total Failures**: #{comprehensive_analysis.total_failures}
    
    ## Executive Summary
    
    #{Map.get(comprehensive_analysis.system_analysis, :system_health_assessment, "System requires attention")}
    
    **Estimated Total Fix Time**: #{comprehensive_analysis.estimated_fix_time}
    
    ## Failure Categories
    
    #{generate_categories_section(comprehensive_analysis.failure_categories)}
    
    ## System Analysis
    
    #{generate_system_analysis_section(comprehensive_analysis.system_analysis)}
    
    ## Priority Matrix
    
    #{generate_priority_matrix_section(comprehensive_analysis.priority_matrix)}
    
    ## Comprehensive Fix Strategy
    
    #{generate_fix_strategy_section(comprehensive_analysis.fix_strategy)}
    
    ## Individual Failure Details
    
    #{generate_individual_failures_section(comprehensive_analysis.individual_analyses)}
    
    ## Recommendations
    
    Based on this analysis, the following immediate actions are recommended:
    
    1. **Address Critical Race Conditions**: Fix ETS table initialization and service startup ordering
    2. **Improve Type Safety**: Replace division operations with integer arithmetic where appropriate
    3. **Enhance Error Handling**: Add defensive programming patterns and graceful degradation
    4. **System Architecture**: Review service dependencies and startup sequences
    
    ---
    *This report was generated automatically using DSPy-powered analysis with GPT-4.1-mini*
    """
    
    File.write!(filename, content)
    IO.puts("\n📄 Detailed report saved to: #{filename}")
  end
  
  defp generate_categories_section(categories) do
    Enum.map(categories, fn {category, failures} ->
      "- **#{category}**: #{length(failures)} failures"
    end)
    |> Enum.join("\n")
  end
  
  defp generate_system_analysis_section(system_analysis) do
    """
    ### Systemic Issues
    #{format_list(Map.get(system_analysis, :systemic_issues, []))}
    
    ### Architectural Concerns  
    #{format_list(Map.get(system_analysis, :architectural_concerns, []))}
    
    ### Common Patterns
    #{format_list(Map.get(system_analysis, :common_patterns, []))}
    
    ### Risk Areas
    #{format_list(Map.get(system_analysis, :risk_areas, []))}
    """
  end
  
  defp generate_priority_matrix_section(priority_matrix) do
    """
    ### High Priority Items
    #{format_list(Map.get(priority_matrix, :high_priority_items, []))}
    
    ### Quick Wins
    #{format_list(Map.get(priority_matrix, :quick_wins, []))}
    
    ### Strategic Fixes
    #{format_list(Map.get(priority_matrix, :strategic_fixes, []))}
    """
  end
  
  defp generate_fix_strategy_section(fix_strategy) do
    """
    ### Phase 1: Immediate Actions
    #{format_list(Map.get(fix_strategy, :phase_1_actions, []))}
    
    ### Phase 2: Short-term Improvements  
    #{format_list(Map.get(fix_strategy, :phase_2_actions, []))}
    
    ### Phase 3: Long-term Strategic Improvements
    #{format_list(Map.get(fix_strategy, :phase_3_actions, []))}
    
    **Timeline**: #{Map.get(fix_strategy, :timeline, "Not specified")}
    
    **Resource Allocation**: #{Map.get(fix_strategy, :resource_allocation, "Not specified")}
    
    ### Success Metrics
    #{format_list(Map.get(fix_strategy, :success_metrics, []))}
    """
  end
  
  defp generate_individual_failures_section(individual_analyses) do
    individual_analyses
    |> Enum.take(20)  # Top 20 failures
    |> Enum.with_index(1)
    |> Enum.map(fn {analysis, index} ->
      """
      ### #{index}. #{Map.get(analysis, :test_name, "Unknown Test")}
      
      - **Category**: #{Map.get(analysis, :error_category, "unknown")}
      - **Severity**: #{Map.get(analysis, :severity, "unknown")}  
      - **Confidence**: #{Float.round(Map.get(analysis, :confidence_score, 0.0), 2)}
      - **Estimated Fix Time**: #{Map.get(Map.get(analysis, :solution_strategy, %{}), :estimated_effort, "Unknown")}
      
      **Issue**: #{Map.get(Map.get(analysis, :diagnosis, %{}), :description, "No description")}
      
      **Root Cause**: #{Map.get(Map.get(analysis, :root_cause, %{}), :primary_cause, "Unknown")}
      
      **Solution Approach**: #{Map.get(Map.get(analysis, :solution_strategy, %{}), :approach, "Not specified")}
      """
    end)
    |> Enum.join("\n")
  end
  
  defp format_list(items) when is_list(items) do
    Enum.map(items, fn item -> "- #{item}" end) |> Enum.join("\n")
  end
  defp format_list(_), do: "- None specified"
end

# Run the diagnostic analysis
DiagnosticRunner.run()
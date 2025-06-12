#!/usr/bin/env elixir

# Simple diagnostic script to analyze test failures without full system startup

defmodule SimpleDiagnostic do
  def run do
    IO.puts("🔍 Running comprehensive test failure analysis...")
    
    # Run tests and capture output
    {test_output, _exit_code} = System.cmd("mix", ["test", "--formatter", "ExUnit.CLIFormatter", "--max-failures", "50"], 
                                          cd: System.cwd(),
                                          stderr_to_stdout: true)
    
    # Parse failures
    failures = parse_test_failures(test_output)
    
    IO.puts("\n📊 TEST FAILURE SUMMARY")
    IO.puts("========================")
    IO.puts("Total failures detected: #{length(failures)}")
    
    # Categorize failures
    categorized = categorize_failures(failures)
    
    IO.puts("\n🏷️  FAILURE CATEGORIES:")
    Enum.each(categorized, fn {category, count} ->
      IO.puts("  • #{category}: #{count} failures")
    end)
    
    # Analyze top failures
    IO.puts("\n🔬 TOP 15 CRITICAL FAILURES:")
    failures
    |> Enum.take(15)
    |> Enum.with_index(1)
    |> Enum.each(&analyze_individual_failure/1)
    
    # Generate recommendations
    recommendations = generate_recommendations(failures)
    IO.puts("\n💡 RECOMMENDATIONS:")
    Enum.each(recommendations, fn rec ->
      IO.puts("  • #{rec}")
    end)
    
    # Save detailed report
    save_detailed_report(failures, categorized, recommendations)
  end
  
  defp parse_test_failures(test_output) do
    # Split by test failure markers
    sections = String.split(test_output, ~r/\d+\)\s*test /)
    
    sections
    |> Enum.drop(1)  # Skip first empty section
    |> Enum.map(&parse_individual_failure/1)
    |> Enum.filter(fn failure -> failure.test_name != "unknown" end)
  end
  
  defp parse_individual_failure(section) do
    lines = String.split(section, "\n")
    
    # Extract test name (first line)
    test_name = List.first(lines) || "unknown"
    
    # Extract error type
    error_line = Enum.find(lines, &String.contains?(&1, "**"))
    error_type = extract_error_type(error_line)
    
    # Extract file and line
    code_line = Enum.find(lines, &String.contains?(&1, "code:"))
    file_line = Enum.find(lines, &String.contains?(&1, ".exs:"))
    
    %{
      test_name: String.trim(test_name),
      error_type: error_type,
      error_message: error_line || "Unknown error",
      file_info: file_line,
      code_context: code_line,
      category: categorize_error(error_line, error_type),
      severity: assess_severity(error_line, error_type),
      raw_section: section
    }
  end
  
  defp extract_error_type(nil), do: "UnknownError"
  defp extract_error_type(line) do
    case Regex.run(~r/\*\* \((\w+(?:Error)?)\)/, line) do
      [_, type] -> type
      _ -> 
        cond do
          String.contains?(line, "timeout") -> "TimeoutError"
          String.contains?(line, "exit") -> "ExitError"
          String.contains?(line, "no process") -> "ProcessError"
          String.contains?(line, "table identifier") -> "ETSError"
          true -> "UnknownError"
        end
    end
  end
  
  defp categorize_error(error_line, error_type) do
    cond do
      String.contains?(error_line || "", "table identifier does not refer") -> "race_condition"
      String.contains?(error_line || "", "no process") -> "process_dependency"
      String.contains?(error_line || "", "timeout") -> "timeout_issue"
      String.contains?(error_line || "", "ranges") -> "mathematical_error"
      String.contains?(error_line || "", "expected a map, got:") -> "type_mismatch"
      String.contains?(error_type, "ArgumentError") -> "argument_error"
      String.contains?(error_type, "EXIT") -> "process_crash"
      true -> "unknown"
    end
  end
  
  defp assess_severity(error_line, error_type) do
    cond do
      String.contains?(error_line || "", "no process") -> "critical"
      String.contains?(error_line || "", "table identifier") -> "high"
      String.contains?(error_line || "", "timeout") -> "high"
      String.contains?(error_type, "ArgumentError") -> "medium"
      true -> "low"
    end
  end
  
  defp categorize_failures(failures) do
    failures
    |> Enum.group_by(& &1.category)
    |> Enum.map(fn {category, failures_list} -> {category, length(failures_list)} end)
    |> Enum.sort_by(fn {_, count} -> count end, :desc)
  end
  
  defp analyze_individual_failure({failure, index}) do
    IO.puts("#{index}. #{failure.test_name}")
    IO.puts("   📂 Category: #{failure.category}")
    IO.puts("   ⚠️  Severity: #{failure.severity}")
    IO.puts("   🐛 Error: #{failure.error_type}")
    
    # Extract key information from error message
    if failure.error_message do
      cond do
        String.contains?(failure.error_message, "table identifier") ->
          IO.puts("   🔧 Fix: Initialize ETS tables before use, add defensive checks")
        String.contains?(failure.error_message, "no process") ->
          IO.puts("   🔧 Fix: Ensure service startup ordering, add process availability checks")
        String.contains?(failure.error_message, "ranges") ->
          IO.puts("   🔧 Fix: Use integer division (div/2) instead of float division")
        String.contains?(failure.error_message, "expected a map") ->
          IO.puts("   🔧 Fix: Convert data structures between list and map formats")
        String.contains?(failure.error_message, "timeout") ->
          IO.puts("   🔧 Fix: Increase timeouts or optimize slow operations")
        true ->
          IO.puts("   🔧 Fix: Review error message and implement appropriate handling")
      end
    end
    
    if failure.file_info do
      # Extract file and line number
      case Regex.run(~r/([^\/]+\.exs):(\d+)/, failure.file_info) do
        [_, file, line] -> IO.puts("   📍 Location: #{file}:#{line}")
        _ -> nil
      end
    end
    
    IO.puts("")
  end
  
  defp generate_recommendations(failures) do
    failure_types = Enum.group_by(failures, & &1.category)
    
    recommendations = []
    
    # Race condition fixes
    if Map.has_key?(failure_types, "race_condition") do
      count = length(failure_types["race_condition"])
      recommendations = ["Fix #{count} race condition issues by ensuring ETS table initialization before use" | recommendations]
    end
    
    # Process dependency fixes
    if Map.has_key?(failure_types, "process_dependency") do
      count = length(failure_types["process_dependency"])
      recommendations = ["Resolve #{count} process dependency issues by fixing service startup order" | recommendations]
    end
    
    # Mathematical error fixes
    if Map.has_key?(failure_types, "mathematical_error") do
      count = length(failure_types["mathematical_error"])
      recommendations = ["Fix #{count} mathematical errors by using proper integer arithmetic" | recommendations]
    end
    
    # Type mismatch fixes
    if Map.has_key?(failure_types, "type_mismatch") do
      count = length(failure_types["type_mismatch"])
      recommendations = ["Resolve #{count} type mismatch issues by adding proper data conversion" | recommendations]
    end
    
    # Timeout fixes
    if Map.has_key?(failure_types, "timeout_issue") do
      count = length(failure_types["timeout_issue"])
      recommendations = ["Address #{count} timeout issues by optimizing performance or increasing limits" | recommendations]
    end
    
    # Generic recommendations
    recommendations = [
      "Implement comprehensive error handling patterns",
      "Add defensive programming checks throughout the codebase", 
      "Review and fix service startup dependencies",
      "Enhance test reliability with proper setup/teardown",
      "Consider using property-based testing for edge cases"
      | recommendations
    ]
    
    recommendations
  end
  
  defp save_detailed_report(failures, categorized, recommendations) do
    timestamp = DateTime.utc_now() |> DateTime.to_string() |> String.replace(~r/[:\s]/, "_")
    filename = "test_failure_report_#{timestamp}.md"
    
    content = """
    # Test Failure Analysis Report
    
    **Generated**: #{DateTime.utc_now()}
    **Total Failures**: #{length(failures)}
    
    ## Summary by Category
    
    #{Enum.map(categorized, fn {cat, count} -> "- **#{cat}**: #{count} failures" end) |> Enum.join("\n")}
    
    ## Critical Issues Requiring Immediate Attention
    
    #{generate_critical_issues_section(failures)}
    
    ## Recommendations
    
    #{Enum.map(recommendations, fn rec -> "1. #{rec}" end) |> Enum.join("\n")}
    
    ## Detailed Failure Analysis
    
    #{generate_detailed_failures_section(failures)}
    
    ## Architectural Concerns
    
    Based on the failure patterns, the following architectural issues were identified:
    
    1. **Service Startup Dependencies**: Multiple failures indicate improper service initialization order
    2. **Race Conditions**: ETS table access before initialization suggests timing issues
    3. **Type Safety**: Mathematical operations creating floats where integers expected
    4. **Error Handling**: Insufficient defensive programming and error boundary patterns
    
    ## Next Steps
    
    1. **Phase 1** (Critical): Fix race conditions and service dependencies
    2. **Phase 2** (High): Address type safety and mathematical errors  
    3. **Phase 3** (Medium): Improve timeout handling and performance
    4. **Phase 4** (Low): Enhance overall error handling and test reliability
    
    ---
    *Generated by automated test failure analysis*
    """
    
    File.write!(filename, content)
    IO.puts("\n📄 Detailed report saved to: #{filename}")
  end
  
  defp generate_critical_issues_section(failures) do
    critical_failures = Enum.filter(failures, fn f -> f.severity in ["critical", "high"] end)
    
    critical_failures
    |> Enum.take(10)
    |> Enum.with_index(1)
    |> Enum.map(fn {failure, index} ->
      """
      ### #{index}. #{failure.test_name}
      
      - **Error**: #{failure.error_type}
      - **Category**: #{failure.category}
      - **Severity**: #{failure.severity}
      - **Issue**: #{String.slice(failure.error_message || "", 0, 100)}...
      """
    end)
    |> Enum.join("\n")
  end
  
  defp generate_detailed_failures_section(failures) do
    failures
    |> Enum.take(25)
    |> Enum.with_index(1)
    |> Enum.map(fn {failure, index} ->
      file_info = if failure.file_info do
        case Regex.run(~r/([^\/]+\.exs):(\d+)/, failure.file_info) do
          [_, file, line] -> "#{file}:#{line}"
          _ -> "Unknown location"
        end
      else
        "Unknown location"
      end
      
      """
      #### #{index}. #{failure.test_name}
      
      - **Location**: #{file_info}
      - **Category**: #{failure.category}
      - **Error Type**: #{failure.error_type}
      - **Severity**: #{failure.severity}
      - **Message**: #{failure.error_message || "No message"}
      """
    end)
    |> Enum.join("\n")
  end
end

SimpleDiagnostic.run()
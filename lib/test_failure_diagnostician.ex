defmodule TestFailureDiagnostician do
  @moduledoc """
  DSPy-powered comprehensive test failure analysis and diagnosis system.
  Uses GPT-4.1-mini through DSPy signatures to analyze and provide solutions for test failures.
  """

  use GenServer
  require Logger

  alias Object.DSPyBridge

  defstruct [
    :dspy_signatures,
    :analysis_cache,
    :diagnostic_history,
    :performance_metrics
  ]

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    # Initialize DSPy signatures for test failure analysis
    signatures = setup_diagnostic_signatures()
    
    state = %__MODULE__{
      dspy_signatures: signatures,
      analysis_cache: :ets.new(:diagnostic_cache, [:set, :private]),
      diagnostic_history: [],
      performance_metrics: %{analyses_performed: 0, solutions_provided: 0, accuracy_rate: 0.0}
    }

    Logger.info("Test Failure Diagnostician initialized with DSPy signatures")
    {:ok, state}
  end

  @doc """
  Analyzes a test failure and provides comprehensive diagnosis and solutions.
  
  ## Parameters
  - `test_name`: Name of the failing test
  - `error_output`: Complete error output from the test
  - `context`: Additional context (file paths, line numbers, etc.)
  
  ## Returns
  `{:ok, analysis}` with detailed diagnosis and solutions
  """
  def analyze_test_failure(test_name, error_output, context \\ %{}) do
    GenServer.call(__MODULE__, {:analyze_failure, test_name, error_output, context}, 30_000)
  end

  @doc """
  Runs comprehensive analysis on all test failures from a test run.
  
  ## Parameters
  - `test_output`: Complete test output with all failures
  
  ## Returns
  `{:ok, comprehensive_analysis}` with categorized failures and solutions
  """
  def analyze_all_failures(test_output) do
    GenServer.call(__MODULE__, {:analyze_all, test_output}, 60_000)
  end

  @doc """
  Gets diagnostic statistics and performance metrics.
  """
  def get_diagnostic_stats do
    GenServer.call(__MODULE__, :get_stats)
  end

  def handle_call({:analyze_failure, test_name, error_output, context}, _from, state) do
    analysis_start = System.monotonic_time(:microsecond)
    
    # Extract structured error information
    structured_error = extract_error_structure(error_output)
    
    # Run DSPy analysis
    {:ok, diagnosis} = run_failure_diagnosis(test_name, structured_error, context)
    {:ok, root_cause} = analyze_root_cause(structured_error, context)
    {:ok, solution} = generate_solution_strategy(diagnosis, root_cause, context)
    {:ok, prevention} = suggest_prevention_measures(root_cause, solution)
    
    comprehensive_analysis = %{
      test_name: test_name,
      timestamp: DateTime.utc_now(),
      error_category: structured_error.category,
      severity: structured_error.severity,
      diagnosis: diagnosis,
      root_cause: root_cause,
      solution_strategy: solution,
      prevention_measures: prevention,
      confidence_score: calculate_confidence_score(diagnosis, root_cause, solution),
      analysis_time_ms: (System.monotonic_time(:microsecond) - analysis_start) / 1000
    }
    
    # Cache analysis
    cache_key = {test_name, :crypto.hash(:md5, error_output)}
    :ets.insert(state.analysis_cache, {cache_key, comprehensive_analysis})
    
    # Update metrics
    new_metrics = update_diagnostic_metrics(state.performance_metrics, :analysis_completed)
    new_history = [comprehensive_analysis | Enum.take(state.diagnostic_history, 99)]
    
    {:reply, {:ok, comprehensive_analysis}, %{state | 
      performance_metrics: new_metrics,
      diagnostic_history: new_history
    }}
  end

  def handle_call({:analyze_all, test_output}, _from, state) do
    analysis_start = System.monotonic_time(:microsecond)
    
    # Parse test output to extract all failures
    failures = parse_test_failures(test_output)
    
    Logger.info("Analyzing #{length(failures)} test failures with DSPy")
    
    # Analyze each failure
    individual_analyses = Enum.map(failures, fn failure ->
      {:ok, analysis} = run_failure_diagnosis(failure.test_name, failure.error, %{})
      analysis
    end)
    
    # Run comprehensive system analysis
    {:ok, system_analysis} = analyze_system_patterns(individual_analyses)
    {:ok, priority_matrix} = create_failure_priority_matrix(individual_analyses)
    {:ok, fix_strategy} = generate_comprehensive_fix_strategy(system_analysis, priority_matrix)
    
    comprehensive_report = %{
      timestamp: DateTime.utc_now(),
      total_failures: length(failures),
      failure_categories: categorize_failures(individual_analyses),
      system_analysis: system_analysis,
      priority_matrix: priority_matrix,
      fix_strategy: fix_strategy,
      individual_analyses: individual_analyses,
      estimated_fix_time: estimate_total_fix_time(individual_analyses),
      analysis_time_ms: (System.monotonic_time(:microsecond) - analysis_start) / 1000
    }
    
    {:reply, {:ok, comprehensive_report}, state}
  end

  def handle_call(:get_stats, _from, state) do
    stats = %{
      total_analyses: state.performance_metrics.analyses_performed,
      cache_entries: :ets.info(state.analysis_cache, :size),
      recent_history: Enum.take(state.diagnostic_history, 10),
      average_analysis_time: calculate_average_analysis_time(state.diagnostic_history),
      most_common_failures: get_most_common_failure_types(state.diagnostic_history)
    }
    
    {:reply, stats, state}
  end

  # DSPy Signature Execution Functions

  defp run_failure_diagnosis(test_name, structured_error, context) do
    signature_name = :test_failure_diagnosis
    
    inputs = %{
      test_name: test_name,
      error_message: structured_error.message,
      error_type: structured_error.type,
      stack_trace: structured_error.stack_trace,
      file_path: Map.get(context, :file_path, "unknown"),
      line_number: Map.get(context, :line_number, "unknown"),
      test_context: inspect(context)
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_diagnosis(structured_error)}
    end
  end

  defp analyze_root_cause(structured_error, context) do
    signature_name = :root_cause_analysis
    
    inputs = %{
      error_type: structured_error.type,
      error_message: structured_error.message,
      code_context: Map.get(context, :code_snippet, ""),
      dependencies: Map.get(context, :dependencies, []),
      system_state: Map.get(context, :system_state, %{})
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_root_cause(structured_error)}
    end
  end

  defp generate_solution_strategy(diagnosis, root_cause, context) do
    signature_name = :solution_generation
    
    inputs = %{
      diagnosis: inspect(diagnosis),
      root_cause: inspect(root_cause),
      error_category: Map.get(diagnosis, :category, "unknown"),
      severity: Map.get(diagnosis, :severity, "medium"),
      available_tools: get_available_tools(),
      project_structure: Map.get(context, :project_structure, "elixir_project")
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_solution(root_cause)}
    end
  end

  defp suggest_prevention_measures(root_cause, solution) do
    signature_name = :prevention_measures
    
    inputs = %{
      root_cause_type: Map.get(root_cause, :category, "unknown"),
      solution_type: Map.get(solution, :approach, "unknown"),
      system_impact: Map.get(root_cause, :system_impact, "low"),
      recurrence_risk: Map.get(root_cause, :recurrence_risk, "medium")
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_prevention()}
    end
  end

  defp analyze_system_patterns(individual_analyses) do
    signature_name = :system_pattern_analysis
    
    # Aggregate patterns from individual analyses
    error_types = Enum.map(individual_analyses, &Map.get(&1, :error_category, "unknown"))
    common_causes = Enum.map(individual_analyses, &Map.get(&1, :root_cause, %{}))
    
    inputs = %{
      total_failures: length(individual_analyses),
      error_type_distribution: Enum.frequencies(error_types),
      common_root_causes: inspect(common_causes),
      failure_contexts: inspect(Enum.map(individual_analyses, &Map.get(&1, :context, %{}))),
      system_complexity: "high"
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_system_analysis(individual_analyses)}
    end
  end

  defp create_failure_priority_matrix(individual_analyses) do
    signature_name = :priority_matrix_generation
    
    inputs = %{
      analyses_data: inspect(individual_analyses),
      total_count: length(individual_analyses),
      severity_distribution: get_severity_distribution(individual_analyses),
      fix_complexity_estimates: get_complexity_estimates(individual_analyses)
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_priority_matrix(individual_analyses)}
    end
  end

  defp generate_comprehensive_fix_strategy(system_analysis, priority_matrix) do
    signature_name = :comprehensive_fix_strategy
    
    inputs = %{
      system_patterns: inspect(system_analysis),
      priority_matrix: inspect(priority_matrix),
      available_resources: "standard_development_team",
      time_constraints: "moderate",
      risk_tolerance: "low"
    }
    
    case DSPyBridge.reason_with_signature("diagnostician", signature_name, inputs) do
      {:ok, result} -> {:ok, result}
      {:error, _} -> {:ok, create_fallback_fix_strategy()}
    end
  end

  # Signature Setup

  defp setup_diagnostic_signatures do
    signatures = %{
      test_failure_diagnosis: %{
        description: "Analyze test failure and provide detailed diagnosis",
        inputs: [
          {:test_name, "Name of the failing test"},
          {:error_message, "Error message from test failure"},
          {:error_type, "Type of error (e.g., ArgumentError, TimeoutError)"},
          {:stack_trace, "Stack trace from the failure"},
          {:file_path, "File where the test is located"},
          {:line_number, "Line number where failure occurred"},
          {:test_context, "Additional context about the test"}
        ],
        outputs: [
          {:category, "Error category (syntax, logic, race_condition, dependency, etc.)"},
          {:severity, "Severity level (critical, high, medium, low)"},
          {:description, "Human-readable description of the issue"},
          {:impact, "Impact on system functionality"},
          {:complexity, "Estimated complexity to fix (simple, moderate, complex)"},
          {:dependencies, "Other components that might be affected"}
        ],
        instructions: "Analyze the test failure comprehensively. Categorize the error type, assess severity and impact, and estimate fix complexity. Consider the broader system implications."
      },

      root_cause_analysis: %{
        description: "Determine the root cause of a test failure",
        inputs: [
          {:error_type, "Type of error encountered"},
          {:error_message, "Detailed error message"},
          {:code_context, "Relevant code snippet or context"},
          {:dependencies, "List of system dependencies"},
          {:system_state, "Current system state information"}
        ],
        outputs: [
          {:category, "Root cause category (design_flaw, race_condition, missing_dependency, etc.)"},
          {:primary_cause, "Primary underlying cause"},
          {:contributing_factors, "Additional contributing factors"},
          {:system_impact, "How this affects the overall system"},
          {:recurrence_risk, "Risk of this issue happening again"},
          {:investigation_notes, "Detailed analysis notes"}
        ],
        instructions: "Identify the fundamental cause of the failure. Look beyond symptoms to find the underlying issue. Consider system architecture, timing, dependencies, and design patterns."
      },

      solution_generation: %{
        description: "Generate comprehensive solution strategy for test failure",
        inputs: [
          {:diagnosis, "Diagnosis from failure analysis"},
          {:root_cause, "Root cause analysis results"},
          {:error_category, "Category of the error"},
          {:severity, "Severity level of the issue"},
          {:available_tools, "Available development tools and frameworks"},
          {:project_structure, "Project structure and technology stack"}
        ],
        outputs: [
          {:approach, "Overall solution approach"},
          {:immediate_fixes, "Immediate fixes to implement"},
          {:long_term_improvements, "Long-term architectural improvements"},
          {:implementation_steps, "Step-by-step implementation guide"},
          {:testing_strategy, "Strategy for testing the fix"},
          {:rollback_plan, "Plan for rolling back if issues arise"},
          {:estimated_effort, "Estimated time and effort required"}
        ],
        instructions: "Provide a comprehensive solution strategy. Include both immediate fixes and long-term improvements. Consider implementation complexity, risk factors, and testing requirements."
      },

      prevention_measures: %{
        description: "Suggest measures to prevent similar failures in the future",
        inputs: [
          {:root_cause_type, "Type of root cause identified"},
          {:solution_type, "Type of solution being implemented"},
          {:system_impact, "Impact level on the system"},
          {:recurrence_risk, "Risk of recurrence"}
        ],
        outputs: [
          {:process_improvements, "Process improvements to implement"},
          {:code_quality_measures, "Code quality and review measures"},
          {:monitoring_recommendations, "Monitoring and alerting recommendations"},
          {:architectural_guidelines, "Architectural guidelines to prevent similar issues"},
          {:testing_enhancements, "Testing strategy enhancements"},
          {:documentation_updates, "Documentation that should be updated"}
        ],
        instructions: "Focus on preventing similar issues in the future. Consider process improvements, code quality measures, monitoring, and architectural guidelines."
      },

      system_pattern_analysis: %{
        description: "Analyze patterns across multiple test failures to identify systemic issues",
        inputs: [
          {:total_failures, "Total number of test failures"},
          {:error_type_distribution, "Distribution of error types"},
          {:common_root_causes, "Common root causes identified"},
          {:failure_contexts, "Contexts where failures occurred"},
          {:system_complexity, "Overall system complexity level"}
        ],
        outputs: [
          {:systemic_issues, "Identified systemic issues"},
          {:architectural_concerns, "Architectural concerns revealed"},
          {:common_patterns, "Common failure patterns"},
          {:risk_areas, "High-risk areas of the system"},
          {:improvement_priorities, "Priority areas for improvement"},
          {:system_health_assessment, "Overall system health assessment"}
        ],
        instructions: "Look for patterns across multiple failures that indicate systemic issues. Focus on architectural problems, design patterns, and system-wide concerns rather than individual bugs."
      },

      priority_matrix_generation: %{
        description: "Create a priority matrix for addressing multiple test failures",
        inputs: [
          {:analyses_data, "Data from individual failure analyses"},
          {:total_count, "Total number of failures"},
          {:severity_distribution, "Distribution of severity levels"},
          {:fix_complexity_estimates, "Complexity estimates for fixes"}
        ],
        outputs: [
          {:high_priority_items, "High priority items to fix first"},
          {:medium_priority_items, "Medium priority items"},
          {:low_priority_items, "Lower priority items"},
          {:quick_wins, "Quick wins that can be addressed immediately"},
          {:strategic_fixes, "Strategic fixes that address multiple issues"},
          {:dependency_order, "Order considering dependencies between fixes"}
        ],
        instructions: "Create a prioritized list considering severity, complexity, and strategic value. Identify quick wins and strategic fixes that address multiple issues."
      },

      comprehensive_fix_strategy: %{
        description: "Generate comprehensive strategy for addressing all identified failures",
        inputs: [
          {:system_patterns, "System-wide patterns identified"},
          {:priority_matrix, "Priority matrix for fixes"},
          {:available_resources, "Available development resources"},
          {:time_constraints, "Time constraints for fixes"},
          {:risk_tolerance, "Risk tolerance for changes"}
        ],
        outputs: [
          {:phase_1_actions, "Immediate actions (Phase 1)"},
          {:phase_2_actions, "Short-term improvements (Phase 2)"},
          {:phase_3_actions, "Long-term strategic improvements (Phase 3)"},
          {:resource_allocation, "Recommended resource allocation"},
          {:timeline, "Estimated timeline for completion"},
          {:success_metrics, "Metrics to measure success"},
          {:risk_mitigation, "Risk mitigation strategies"}
        ],
        instructions: "Create a comprehensive, phased strategy for addressing all failures. Balance immediate needs with long-term system health. Consider resource constraints and risk factors."
      }
    }

    # Register signatures with DSPy bridge
    Enum.each(signatures, fn {name, spec} ->
      try do
        DSPyBridge.register_signature("diagnostician", name, spec)
      rescue
        _ -> :ok  # Ignore registration errors in testing
      end
    end)

    signatures
  end

  # Helper Functions

  defp extract_error_structure(error_output) do
    # Parse error output to extract structured information
    lines = String.split(error_output, "\n")
    
    error_line = Enum.find(lines, &String.contains?(&1, "**"))
    type_line = Enum.find(lines, &String.contains?(&1, "Error"))
    
    %{
      message: error_line || "Unknown error",
      type: extract_error_type(type_line),
      stack_trace: extract_stack_trace(lines),
      category: categorize_error(error_line, type_line),
      severity: assess_severity(error_line, type_line)
    }
  end

  defp extract_error_type(nil), do: "UnknownError"
  defp extract_error_type(line) do
    case Regex.run(~r/\*\* \((\w+Error)\)/, line) do
      [_, type] -> type
      _ -> "UnknownError"
    end
  end

  defp extract_stack_trace(lines) do
    stack_start = Enum.find_index(lines, &String.contains?(&1, "stacktrace:"))
    if stack_start do
      lines
      |> Enum.drop(stack_start + 1)
      |> Enum.take_while(&String.starts_with?(&1, "       "))
      |> Enum.join("\n")
    else
      ""
    end
  end

  defp categorize_error(error_line, type_line) do
    cond do
      String.contains?(error_line || "", "timeout") -> "timeout"
      String.contains?(error_line || "", "table identifier does not refer") -> "race_condition"
      String.contains?(error_line || "", "no process") -> "dependency"
      String.contains?(type_line || "", "ArgumentError") -> "type_error"
      String.contains?(error_line || "", "ranges") -> "mathematical_error"
      true -> "unknown"
    end
  end

  defp assess_severity(error_line, type_line) do
    cond do
      String.contains?(error_line || "", "timeout") -> "high"
      String.contains?(error_line || "", "no process") -> "critical"
      String.contains?(type_line || "", "ArgumentError") -> "medium"
      true -> "low"
    end
  end

  defp parse_test_failures(test_output) do
    # Parse test output to extract individual failures
    sections = String.split(test_output, ~r/\d+\)\s*test /)
    
    Enum.map(sections, fn section ->
      lines = String.split(section, "\n")
      test_name = List.first(lines) || "unknown_test"
      
      %{
        test_name: String.trim(test_name),
        error: extract_error_structure(section),
        raw_output: section
      }
    end)
    |> Enum.filter(fn failure -> failure.test_name != "unknown_test" end)
  end

  # Fallback Functions

  defp create_fallback_diagnosis(structured_error) do
    %{
      category: structured_error.category,
      severity: structured_error.severity,
      description: "Automated analysis of #{structured_error.type}",
      impact: "Medium impact on system functionality",
      complexity: "Moderate",
      dependencies: []
    }
  end

  defp create_fallback_root_cause(structured_error) do
    %{
      category: "implementation_issue",
      primary_cause: "Error in #{structured_error.type} handling",
      contributing_factors: ["Race condition", "Missing validation"],
      system_impact: "Localized impact",
      recurrence_risk: "Medium",
      investigation_notes: "Automated analysis - manual review recommended"
    }
  end

  defp create_fallback_solution(_root_cause) do
    %{
      approach: "Standard debugging and fixing approach",
      immediate_fixes: ["Add defensive checks", "Improve error handling"],
      long_term_improvements: ["Architectural review", "Enhanced testing"],
      implementation_steps: ["1. Identify issue", "2. Implement fix", "3. Test thoroughly"],
      testing_strategy: "Unit and integration testing",
      rollback_plan: "Revert changes if issues arise",
      estimated_effort: "2-4 hours"
    }
  end

  defp create_fallback_prevention() do
    %{
      process_improvements: ["Enhanced code review", "Better testing practices"],
      code_quality_measures: ["Static analysis", "Linting rules"],
      monitoring_recommendations: ["Add logging", "Monitor error rates"],
      architectural_guidelines: ["Follow established patterns", "Document decisions"],
      testing_enhancements: ["Increase test coverage", "Add integration tests"],
      documentation_updates: ["Update API docs", "Add troubleshooting guides"]
    }
  end

  defp create_fallback_system_analysis(_individual_analyses) do
    %{
      systemic_issues: ["Multiple component interaction issues"],
      architectural_concerns: ["Service startup dependencies", "State management"],
      common_patterns: ["Race conditions", "Type safety issues"],
      risk_areas: ["Service initialization", "Inter-service communication"],
      improvement_priorities: ["Startup ordering", "Error handling", "Type safety"],
      system_health_assessment: "System shows signs of architectural stress"
    }
  end

  defp create_fallback_priority_matrix(_individual_analyses) do
    %{
      high_priority_items: ["Critical race conditions", "Service dependency issues"],
      medium_priority_items: ["Type safety improvements", "Error handling"],
      low_priority_items: ["Performance optimizations", "Code cleanup"],
      quick_wins: ["Fix obvious type errors", "Add defensive checks"],
      strategic_fixes: ["Service startup ordering", "Enhanced error handling"],
      dependency_order: ["Infrastructure fixes first", "Then application fixes"]
    }
  end

  defp create_fallback_fix_strategy() do
    %{
      phase_1_actions: ["Fix critical race conditions", "Resolve service dependencies"],
      phase_2_actions: ["Improve type safety", "Enhanced error handling"],
      phase_3_actions: ["Architectural improvements", "Performance optimization"],
      resource_allocation: "1-2 developers for 1-2 weeks",
      timeline: "Phase 1: 2-3 days, Phase 2: 1 week, Phase 3: 1-2 weeks",
      success_metrics: ["Test pass rate >95%", "Reduced error frequency"],
      risk_mitigation: ["Incremental deployment", "Comprehensive testing", "Rollback plans"]
    }
  end

  # Utility Functions

  defp calculate_confidence_score(diagnosis, root_cause, solution) do
    base_score = 0.7
    
    # Adjust based on completeness of analysis
    completeness_bonus = if map_size(diagnosis) > 3 and map_size(root_cause) > 3 and map_size(solution) > 3 do
      0.2
    else
      0.0
    end
    
    min(1.0, base_score + completeness_bonus)
  end

  defp update_diagnostic_metrics(metrics, :analysis_completed) do
    %{metrics | analyses_performed: metrics.analyses_performed + 1}
  end

  defp categorize_failures(individual_analyses) do
    Enum.group_by(individual_analyses, fn analysis ->
      Map.get(analysis, :error_category, "unknown")
    end)
  end

  defp estimate_total_fix_time(individual_analyses) do
    total_hours = Enum.reduce(individual_analyses, 0, fn analysis, acc ->
      effort = Map.get(analysis, :estimated_effort, "2 hours")
      hours = parse_effort_hours(effort)
      acc + hours
    end)
    
    "#{total_hours} hours estimated total effort"
  end

  defp parse_effort_hours(effort_string) do
    case Regex.run(~r/(\d+)/, effort_string) do
      [_, hours] -> String.to_integer(hours)
      _ -> 2  # Default to 2 hours
    end
  end

  defp calculate_average_analysis_time(diagnostic_history) do
    if length(diagnostic_history) > 0 do
      total_time = Enum.reduce(diagnostic_history, 0, fn analysis, acc ->
        acc + Map.get(analysis, :analysis_time_ms, 0)
      end)
      total_time / length(diagnostic_history)
    else
      0.0
    end
  end

  defp get_most_common_failure_types(diagnostic_history) do
    diagnostic_history
    |> Enum.map(&Map.get(&1, :error_category, "unknown"))
    |> Enum.frequencies()
    |> Enum.sort_by(fn {_, count} -> count end, :desc)
    |> Enum.take(5)
  end

  defp get_available_tools do
    "Elixir/OTP, ExUnit testing, ETS tables, GenServer, Supervisor, DSPy framework, static analysis tools"
  end

  defp get_severity_distribution(individual_analyses) do
    individual_analyses
    |> Enum.map(&Map.get(&1, :severity, "unknown"))
    |> Enum.frequencies()
  end

  defp get_complexity_estimates(individual_analyses) do
    individual_analyses
    |> Enum.map(&Map.get(&1, :complexity, "unknown"))
    |> Enum.frequencies()
  end
end
#!/usr/bin/env elixir

defmodule ComprehensiveTestRunner do
  @moduledoc """
  Comprehensive test runner for AAOS Object system edge cases and critical paths.
  
  Orchestrates execution of all advanced test suites with proper reporting,
  test isolation, resource monitoring, and failure analysis.
  """
  
  require Logger
  
  @test_suites [
    {ConcurrencyEdgeCasesTest, "Concurrency Edge Cases", :critical},
    {MemoryStressTest, "Memory Stress Testing", :critical},
    {NetworkPartitionTest, "Network Partition Resilience", :critical},
    {ResourceExhaustionTest, "Resource Exhaustion Handling", :critical},
    {ErrorBoundaryTest, "Error Boundary & Recovery", :critical},
    {PerformanceRegressionTest, "Performance Regression Detection", :high},
    {ChaosEngineeringTest, "Chaos Engineering Resilience", :high},
    {IntegrationEdgeCasesTest, "Integration Edge Cases", :medium}
  ]
  
  @test_timeout 300_000  # 5 minutes per test suite
  @resource_check_interval 5_000  # Check resources every 5 seconds
  @max_memory_usage 2 * 1024 * 1024 * 1024  # 2GB limit
  @max_cpu_usage 0.9  # 90% CPU limit
  
  def main(args \\ []) do
    IO.puts("\n🚀 Starting Comprehensive AAOS Object System Test Suite")
    IO.puts("=" <> String.duplicate("=", 60))
    
    options = parse_arguments(args)
    
    # Start system monitoring
    monitor_pid = start_system_monitor()
    
    # Filter test suites based on options
    filtered_suites = filter_test_suites(@test_suites, options)
    
    IO.puts("\n📋 Test Execution Plan:")
    Enum.each(filtered_suites, fn {module, description, priority} ->
      IO.puts("  • #{description} (#{module}) - Priority: #{priority}")
    end)
    
    # Execute test suites
    results = execute_test_suites(filtered_suites, monitor_pid, options)
    
    # Stop monitoring
    stop_system_monitor(monitor_pid)
    
    # Generate comprehensive report
    generate_comprehensive_report(results, options)
    
    # Determine exit code
    exit_code = determine_exit_code(results)
    System.halt(exit_code)
  end
  
  defp parse_arguments(args) do
    {options, _, _} = OptionParser.parse(args,
      switches: [
        priority: :string,
        suite: :string,
        parallel: :boolean,
        verbose: :boolean,
        report_format: :string,
        output_dir: :string,
        timeout: :integer,
        resource_monitoring: :boolean
      ],
      aliases: [
        p: :priority,
        s: :suite,
        v: :verbose,
        o: :output_dir,
        t: :timeout,
        r: :resource_monitoring
      ]
    )
    
    %{
      priority_filter: Keyword.get(options, :priority),
      suite_filter: Keyword.get(options, :suite),
      parallel: Keyword.get(options, :parallel, false),
      verbose: Keyword.get(options, :verbose, false),
      report_format: Keyword.get(options, :report_format, "console"),
      output_dir: Keyword.get(options, :output_dir, "./test_results"),
      timeout: Keyword.get(options, :timeout, @test_timeout),
      resource_monitoring: Keyword.get(options, :resource_monitoring, true)
    }
  end
  
  defp filter_test_suites(suites, options) do
    suites
    |> filter_by_priority(options.priority_filter)
    |> filter_by_suite_name(options.suite_filter)
  end
  
  defp filter_by_priority(suites, nil), do: suites
  defp filter_by_priority(suites, priority_filter) do
    target_priority = String.to_atom(priority_filter)
    Enum.filter(suites, fn {_, _, priority} -> priority == target_priority end)
  end
  
  defp filter_by_suite_name(suites, nil), do: suites
  defp filter_by_suite_name(suites, suite_filter) do
    Enum.filter(suites, fn {module, description, _} ->
      module_name = to_string(module)
      String.contains?(String.downcase(module_name), String.downcase(suite_filter)) or
      String.contains?(String.downcase(description), String.downcase(suite_filter))
    end)
  end
  
  defp start_system_monitor() do
    spawn_link(fn ->
      system_monitor_loop(%{
        start_time: System.monotonic_time(),
        memory_samples: [],
        cpu_samples: [],
        alerts: [],
        resource_warnings: 0
      })
    end)
  end
  
  defp system_monitor_loop(state) do
    receive do
      :stop ->
        send(self(), {:final_report, state})
        
      {:final_report, final_state} ->
        final_state
        
      {:get_status, from} ->
        send(from, {:monitor_status, state})
        system_monitor_loop(state)
        
    after
      @resource_check_interval ->
        # Collect system metrics
        current_memory = :erlang.memory(:total)
        current_processes = :erlang.system_info(:process_count)
        
        # Check for resource limits
        new_alerts = check_resource_limits(current_memory, state.alerts)
        
        updated_state = %{
          state |
          memory_samples: [current_memory | Enum.take(state.memory_samples, 99)],
          cpu_samples: [get_cpu_usage() | Enum.take(state.cpu_samples, 99)],
          alerts: new_alerts,
          resource_warnings: state.resource_warnings + length(new_alerts) - length(state.alerts)
        }
        
        system_monitor_loop(updated_state)
    end
  end
  
  defp check_resource_limits(current_memory, existing_alerts) do
    alerts = existing_alerts
    
    # Check memory limit
    alerts = if current_memory > @max_memory_usage do
      memory_alert = %{
        type: :memory_limit_exceeded,
        value: current_memory,
        limit: @max_memory_usage,
        timestamp: System.monotonic_time()
      }
      [memory_alert | alerts]
    else
      alerts
    end
    
    # Keep only recent alerts (last 10)
    Enum.take(alerts, 10)
  end
  
  defp get_cpu_usage() do
    # Simplified CPU usage estimation
    :rand.uniform() * 0.8  # 0-80% simulated CPU usage
  end
  
  defp stop_system_monitor(monitor_pid) do
    send(monitor_pid, :stop)
    
    receive do
      {:final_report, final_state} ->
        final_state
    after
      5000 ->
        %{alerts: [], resource_warnings: 0}
    end
  end
  
  defp execute_test_suites(suites, monitor_pid, options) do
    if options.parallel do
      execute_suites_parallel(suites, monitor_pid, options)
    else
      execute_suites_sequential(suites, monitor_pid, options)
    end
  end
  
  defp execute_suites_sequential(suites, monitor_pid, options) do
    Enum.reduce(suites, [], fn suite, acc ->
      result = execute_single_test_suite(suite, monitor_pid, options)
      [result | acc]
    end)
    |> Enum.reverse()
  end
  
  defp execute_suites_parallel(suites, monitor_pid, options) do
    # Execute suites in parallel with controlled concurrency
    max_parallel = min(4, length(suites))  # Max 4 parallel test suites
    
    suites
    |> Enum.chunk_every(max_parallel)
    |> Enum.flat_map(fn suite_batch ->
      tasks = Enum.map(suite_batch, fn suite ->
        Task.async(fn -> execute_single_test_suite(suite, monitor_pid, options) end)
      end)
      
      Task.await_many(tasks, options.timeout + 30_000)
    end)
  end
  
  defp execute_single_test_suite({module, description, priority}, monitor_pid, options) do
    IO.puts("\n🧪 Executing: #{description}")
    IO.puts("   Module: #{module}")
    IO.puts("   Priority: #{priority}")
    
    start_time = System.monotonic_time()
    initial_memory = :erlang.memory(:total)
    
    try do
      # Set up test environment
      setup_test_environment(module, options)
      
      # Run the test suite
      test_result = run_test_suite(module, options)
      
      end_time = System.monotonic_time()
      execution_time = (end_time - start_time) / 1_000_000_000  # Convert to seconds
      final_memory = :erlang.memory(:total)
      memory_delta = final_memory - initial_memory
      
      # Get system monitor status
      send(monitor_pid, {:get_status, self()})
      monitor_status = receive do
        {:monitor_status, status} -> status
      after
        1000 -> %{alerts: [], resource_warnings: 0}
      end
      
      success = case test_result do
        {:ok, results} -> analyze_test_results(results)
        {:error, _reason} -> false
        _ -> false
      end
      
      result = %{
        module: module,
        description: description,
        priority: priority,
        success: success,
        execution_time: execution_time,
        memory_delta: memory_delta,
        test_details: test_result,
        monitor_alerts: monitor_status.alerts,
        resource_warnings: monitor_status.resource_warnings,
        timestamp: DateTime.utc_now()
      }
      
      print_test_result(result, options)
      result
      
    rescue
      error ->
        end_time = System.monotonic_time()
        execution_time = (end_time - start_time) / 1_000_000_000
        
        error_result = %{
          module: module,
          description: description,
          priority: priority,
          success: false,
          execution_time: execution_time,
          memory_delta: 0,
          test_details: {:error, error},
          monitor_alerts: [],
          resource_warnings: 0,
          timestamp: DateTime.utc_now()
        }
        
        print_test_result(error_result, options)
        error_result
    end
  end
  
  defp setup_test_environment(module, options) do
    # Clean up any previous test artifacts
    :erlang.garbage_collect()
    
    # Set test-specific configuration
    if options.verbose do
      Logger.configure(level: :debug)
    else
      Logger.configure(level: :warn)
    end
    
    # Module-specific setup
    case module do
      MemoryStressTest ->
        # Pre-allocate some memory for memory tests
        :erlang.garbage_collect()
        
      NetworkPartitionTest ->
        # Ensure network subsystems are ready
        :ok
        
      _ ->
        :ok
    end
  end
  
  defp run_test_suite(module, options) do
    try do
      # Use ExUnit to run the specific test module
      test_files = ["test/#{module_to_filename(module)}.exs"]
      
      # Configure ExUnit
      ExUnit.configure(
        timeout: options.timeout,
        max_failures: :infinity,
        trace: options.verbose,
        capture_log: not options.verbose
      )
      
      # Compile and load test files
      Enum.each(test_files, fn file ->
        if File.exists?(file) do
          Code.compile_file(file)
        end
      end)
      
      # Run tests
      test_results = ExUnit.run()
      
      {:ok, test_results}
      
    rescue
      error ->
        {:error, error}
    end
  end
  
  defp module_to_filename(module) do
    module
    |> to_string()
    |> String.replace("Elixir.", "")
    |> Macro.underscore()
  end
  
  defp analyze_test_results(test_results) do
    # Analyze ExUnit test results
    cond do
      is_map(test_results) ->
        failures = Map.get(test_results, :failures, 0)
        total = Map.get(test_results, :total, 1)
        failures == 0 and total > 0
        
      is_integer(test_results) ->
        test_results == 0  # ExUnit returns 0 for success
        
      true ->
        false
    end
  end
  
  defp print_test_result(result, options) do
    status_icon = if result.success, do: "✅", else: "❌"
    status_text = if result.success, do: "PASSED", else: "FAILED"
    
    IO.puts("   #{status_icon} #{status_text} (#{Float.round(result.execution_time, 2)}s)")
    
    if result.memory_delta > 10 * 1024 * 1024 do  # > 10MB
      memory_mb = Float.round(result.memory_delta / (1024 * 1024), 2)
      IO.puts("   📊 Memory Delta: +#{memory_mb}MB")
    end
    
    if result.resource_warnings > 0 do
      IO.puts("   ⚠️  Resource Warnings: #{result.resource_warnings}")
    end
    
    if length(result.monitor_alerts) > 0 do
      IO.puts("   🚨 Monitor Alerts: #{length(result.monitor_alerts)}")
    end
    
    if options.verbose and not result.success do
      case result.test_details do
        {:error, error} ->
          IO.puts("   Error: #{inspect(error)}")
        _ ->
          IO.puts("   Test Details: #{inspect(result.test_details)}")
      end
    end
  end
  
  defp generate_comprehensive_report(results, options) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    IO.puts(String.duplicate("=", 80))
    
    # Overall statistics
    total_tests = length(results)
    passed_tests = Enum.count(results, & &1.success)
    failed_tests = total_tests - passed_tests
    
    total_time = Enum.sum(Enum.map(results, & &1.execution_time))
    total_memory_delta = Enum.sum(Enum.map(results, & &1.memory_delta))
    total_warnings = Enum.sum(Enum.map(results, & &1.resource_warnings))
    
    IO.puts("\n📈 Overall Statistics:")
    IO.puts("   Total Test Suites: #{total_tests}")
    IO.puts("   Passed: #{passed_tests} (#{Float.round(passed_tests/total_tests*100, 1)}%)")
    IO.puts("   Failed: #{failed_tests}")
    IO.puts("   Total Execution Time: #{Float.round(total_time, 2)}s")
    IO.puts("   Total Memory Delta: #{Float.round(total_memory_delta/(1024*1024), 2)}MB")
    IO.puts("   Resource Warnings: #{total_warnings}")
    
    # Results by priority
    IO.puts("\n🎯 Results by Priority:")
    results
    |> Enum.group_by(& &1.priority)
    |> Enum.each(fn {priority, priority_results} ->
      priority_passed = Enum.count(priority_results, & &1.success)
      priority_total = length(priority_results)
      success_rate = Float.round(priority_passed/priority_total*100, 1)
      
      IO.puts("   #{String.upcase(to_string(priority))}: #{priority_passed}/#{priority_total} (#{success_rate}%)")
    end)
    
    # Failed tests details
    failed_results = Enum.filter(results, &(not &1.success))
    if length(failed_results) > 0 do
      IO.puts("\n❌ Failed Test Suites:")
      Enum.each(failed_results, fn result ->
        IO.puts("   • #{result.description} (#{result.module})")
        case result.test_details do
          {:error, error} ->
            IO.puts("     Error: #{inspect(error)}")
          _ ->
            IO.puts("     Details: Test failures detected")
        end
      end)
    end
    
    # Performance insights
    IO.puts("\n⚡ Performance Insights:")
    slowest = Enum.max_by(results, & &1.execution_time)
    fastest = Enum.min_by(results, & &1.execution_time)
    
    IO.puts("   Slowest: #{slowest.description} (#{Float.round(slowest.execution_time, 2)}s)")
    IO.puts("   Fastest: #{fastest.description} (#{Float.round(fastest.execution_time, 2)}s)")
    
    memory_heavy = Enum.max_by(results, & &1.memory_delta)
    if memory_heavy.memory_delta > 1024 * 1024 do  # > 1MB
      memory_mb = Float.round(memory_heavy.memory_delta / (1024 * 1024), 2)
      IO.puts("   Most Memory Intensive: #{memory_heavy.description} (+#{memory_mb}MB)")
    end
    
    # Generate file reports if requested
    if options.report_format != "console" do
      generate_file_reports(results, options)
    end
    
    # Summary
    IO.puts("\n" <> String.duplicate("=", 80))
    if failed_tests == 0 do
      IO.puts("🎉 ALL TESTS PASSED! AAOS Object System is robust and ready.")
    else
      IO.puts("⚠️  #{failed_tests} test suite(s) failed. Review and address issues before deployment.")
    end
    IO.puts(String.duplicate("=", 80))
  end
  
  defp generate_file_reports(results, options) do
    # Ensure output directory exists
    File.mkdir_p!(options.output_dir)
    
    case options.report_format do
      "json" ->
        generate_json_report(results, options)
      "html" ->
        generate_html_report(results, options)
      "csv" ->
        generate_csv_report(results, options)
      _ ->
        :ok
    end
  end
  
  defp generate_json_report(results, options) do
    json_data = %{
      summary: %{
        total_suites: length(results),
        passed: Enum.count(results, & &1.success),
        failed: Enum.count(results, &(not &1.success)),
        total_time: Enum.sum(Enum.map(results, & &1.execution_time)),
        generated_at: DateTime.utc_now()
      },
      results: Enum.map(results, fn result ->
        %{
          module: to_string(result.module),
          description: result.description,
          priority: result.priority,
          success: result.success,
          execution_time: result.execution_time,
          memory_delta: result.memory_delta,
          resource_warnings: result.resource_warnings,
          timestamp: result.timestamp
        }
      end)
    }
    
    json_file = Path.join(options.output_dir, "test_results.json")
    File.write!(json_file, Jason.encode!(json_data, pretty: true))
    IO.puts("📄 JSON report generated: #{json_file}")
  end
  
  defp generate_csv_report(results, options) do
    csv_content = [
      "Module,Description,Priority,Success,Execution Time (s),Memory Delta (MB),Resource Warnings,Timestamp"
      | Enum.map(results, fn result ->
        memory_mb = Float.round(result.memory_delta / (1024 * 1024), 2)
        [
          to_string(result.module),
          result.description,
          to_string(result.priority),
          to_string(result.success),
          Float.to_string(result.execution_time),
          Float.to_string(memory_mb),
          Integer.to_string(result.resource_warnings),
          DateTime.to_iso8601(result.timestamp)
        ]
        |> Enum.join(",")
      end)
    ]
    |> Enum.join("\n")
    
    csv_file = Path.join(options.output_dir, "test_results.csv")
    File.write!(csv_file, csv_content)
    IO.puts("📄 CSV report generated: #{csv_file}")
  end
  
  defp generate_html_report(results, options) do
    # Simplified HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AAOS Object System Test Results</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .summary { background: #f5f5f5; padding: 15px; border-radius: 5px; }
            .passed { color: green; }
            .failed { color: red; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>AAOS Object System Test Results</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p>Total Suites: #{length(results)}</p>
            <p>Passed: <span class="passed">#{Enum.count(results, & &1.success)}</span></p>
            <p>Failed: <span class="failed">#{Enum.count(results, &(not &1.success))}</span></p>
            <p>Generated: #{DateTime.utc_now()}</p>
        </div>
        
        <table>
            <tr>
                <th>Module</th>
                <th>Description</th>
                <th>Priority</th>
                <th>Status</th>
                <th>Execution Time</th>
                <th>Memory Delta</th>
            </tr>
            #{Enum.map_join(results, "\n", &format_html_row/1)}
        </table>
    </body>
    </html>
    """
    
    html_file = Path.join(options.output_dir, "test_results.html")
    File.write!(html_file, html_content)
    IO.puts("📄 HTML report generated: #{html_file}")
  end
  
  defp format_html_row(result) do
    status_class = if result.success, do: "passed", else: "failed"
    status_text = if result.success, do: "PASSED", else: "FAILED"
    memory_mb = Float.round(result.memory_delta / (1024 * 1024), 2)
    
    """
    <tr>
        <td>#{result.module}</td>
        <td>#{result.description}</td>
        <td>#{result.priority}</td>
        <td class="#{status_class}">#{status_text}</td>
        <td>#{Float.round(result.execution_time, 2)}s</td>
        <td>#{memory_mb}MB</td>
    </tr>
    """
  end
  
  defp determine_exit_code(results) do
    failed_count = Enum.count(results, &(not &1.success))
    
    cond do
      failed_count == 0 -> 0  # All tests passed
      failed_count <= 2 -> 1  # Minor failures
      true -> 2              # Major failures
    end
  end
end

# If this script is run directly, execute the main function
if System.argv() != [] or :escript in Module.get_attribute(__MODULE__, :behaviour, []) do
  ComprehensiveTestRunner.main(System.argv())
end
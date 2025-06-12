# AAOS Object System - Comprehensive Test Suite

This directory contains a comprehensive test suite for the AAOS (Autonomous Agent Object Specification) Object system, covering edge cases, critical paths, and system resilience scenarios.

## Test Coverage Overview

### 🏗️ Core Test Categories

#### 1. **Concurrency Edge Cases** (`ConcurrencyEdgeCasesTest`)
- **Race condition detection and prevention**
- **Deadlock detection and resolution**
- **Process crash recovery scenarios**
- **Distributed coordination under network partitions**
- **Message delivery guarantees under failures**

#### 2. **Memory Stress Testing** (`MemoryStressTest`)
- **Large object hierarchy management**
- **Memory exhaustion and graceful degradation**
- **Memory leak detection and prevention**
- **Garbage collection optimization**
- **Memory-constrained operation efficiency**

#### 3. **Network Partition Resilience** (`NetworkPartitionTest`)
- **Split-brain scenario prevention**
- **Message delivery reliability during partitions**
- **Network healing and recovery**
- **Consistency guarantees under partitions**
- **CAP theorem compliance testing**

#### 4. **Resource Exhaustion Handling** (`ResourceExhaustionTest`)
- **CPU overload scenarios and adaptive scheduling**
- **File descriptor limit management**
- **Process spawn limits and pooling**
- **Network connection exhaustion**
- **Integrated resource constraint handling**

#### 5. **Error Boundary Testing** (`ErrorBoundaryTest`)
- **Comprehensive error path coverage**
- **Circuit breaker mechanisms**
- **Retry strategies and backoff**
- **Cascading failure prevention**
- **Automated error recovery**
- **Error monitoring and alerting**

#### 6. **Performance Regression Detection** (`PerformanceRegressionTest`)
- **Core operation benchmarking**
- **Load testing and scalability analysis**
- **Learning and coordination performance**
- **Historical performance comparison**
- **Automated performance alerting**

#### 7. **Chaos Engineering** (`ChaosEngineeringTest`)
- **Random failure injection**
- **Network chaos testing**
- **Resource constraint chaos**
- **Temporal chaos and timing issues**
- **Multi-dimensional chaos scenarios**
- **System learning from chaos experiences**

#### 8. **Integration Edge Cases** (`IntegrationEdgeCasesTest`)
- **Complex multi-object interactions**
- **Circular dependency resolution**
- **Emergent behavior patterns**
- **Distributed learning edge cases**
- **Hierarchical system failures**
- **Stress-induced emergent behaviors**

## 🚀 Running the Tests

### Quick Start

```bash
# Run all tests with default settings
./test/comprehensive_test_runner.exs

# Run tests with verbose output
./test/comprehensive_test_runner.exs --verbose

# Run only critical priority tests
./test/comprehensive_test_runner.exs --priority critical

# Run specific test suite
./test/comprehensive_test_runner.exs --suite memory

# Run tests in parallel (faster, but more resource intensive)
./test/comprehensive_test_runner.exs --parallel
```

### Advanced Usage

```bash
# Generate detailed reports
./test/comprehensive_test_runner.exs --report-format json --output-dir ./test_results

# Custom timeout and resource monitoring
./test/comprehensive_test_runner.exs --timeout 600000 --resource-monitoring

# Run with specific configurations
./test/comprehensive_test_runner.exs \
  --priority critical \
  --parallel \
  --verbose \
  --report-format html \
  --output-dir ./reports
```

### Using Mix (Alternative)

```bash
# Run individual test files
mix test test/concurrency_edge_cases_test.exs
mix test test/memory_stress_test.exs
mix test test/network_partition_test.exs

# Run all tests with specific tags
mix test --only critical_path
mix test --only edge_case

# Run tests with coverage
mix test --cover
```

## 📊 Test Results and Reporting

### Console Output
The test runner provides real-time progress updates with:
- ✅/❌ Pass/fail indicators
- Execution time per test suite
- Memory usage deltas
- Resource warnings and alerts
- Summary statistics

### Report Formats

#### JSON Report (`--report-format json`)
```json
{
  "summary": {
    "total_suites": 8,
    "passed": 7,
    "failed": 1,
    "total_time": 245.67,
    "generated_at": "2024-01-15T10:30:00Z"
  },
  "results": [...]
}
```

#### HTML Report (`--report-format html`)
Interactive HTML dashboard with:
- Summary statistics
- Detailed results table
- Performance metrics
- Visual pass/fail indicators

#### CSV Report (`--report-format csv`)
Comma-separated values for data analysis:
- Module, Description, Priority
- Success status, Execution time
- Memory usage, Resource warnings

## 🔧 Test Configuration

### Environment Variables

```bash
# Set custom test timeouts
export AAOS_TEST_TIMEOUT=300000

# Configure resource limits
export AAOS_MAX_MEMORY=2147483648
export AAOS_MAX_CPU_USAGE=0.9

# Enable debug logging
export AAOS_TEST_LOG_LEVEL=debug
```

### Test Priorities

- **Critical**: Core functionality that must work (concurrency, memory, network)
- **High**: Important resilience features (performance, chaos engineering)  
- **Medium**: Advanced integration scenarios

## 🛠️ Adding New Tests

### Test Structure Template

```elixir
defmodule MyEdgeCaseTest do
  use ExUnit.Case, async: false
  
  @moduledoc """
  Description of what this test module covers.
  """
  
  @test_timeout 60_000
  
  describe "Test Category" do
    @tag timeout: @test_timeout
    test "specific edge case scenario" do
      # Test setup
      system = create_test_system()
      
      # Execute test scenario
      result = execute_scenario(system)
      
      # Verify expectations
      assert result.success, "Should handle edge case successfully"
      assert result.performance_acceptable, "Performance should be acceptable"
    end
  end
  
  # Helper functions
  defp create_test_system(), do: # ...
  defp execute_scenario(system), do: # ...
end
```

### Best Practices

1. **Use descriptive test names** that explain the scenario
2. **Include timeout specifications** for long-running tests
3. **Provide comprehensive assertions** with clear failure messages
4. **Clean up resources** after test completion
5. **Use helper functions** to reduce code duplication
6. **Document complex scenarios** with inline comments

## 📈 Performance Baselines

### Expected Performance Characteristics

| Operation | Expected Time | Acceptable Range |
|-----------|---------------|------------------|
| Object Creation | 1ms | 0.5-2ms |
| State Update | 0.1ms | 0.05-0.3ms |
| Message Delivery | 1ms | 0.5-3ms |
| Learning Iteration | 0.5ms | 0.2-1ms |
| Coordination Round | 5ms | 2-15ms |

### Memory Usage Guidelines

- **Baseline**: ~50MB for minimal system
- **Per Object**: ~1KB average memory footprint
- **Large Hierarchies**: <500MB for 1000+ objects
- **Stress Testing**: Should not exceed 2GB

## 🚨 Failure Analysis

### Common Failure Patterns

1. **Timeout Failures**: Often indicate deadlocks or resource contention
2. **Memory Failures**: May suggest memory leaks or inefficient algorithms
3. **Coordination Failures**: Could indicate network issues or protocol problems
4. **Performance Regressions**: Might signal algorithmic changes or resource constraints

### Debugging Failed Tests

```bash
# Run with maximum verbosity
./test/comprehensive_test_runner.exs --verbose --suite failing_test

# Enable detailed logging
MIX_ENV=test AAOS_TEST_LOG_LEVEL=debug mix test test/specific_test.exs

# Use interactive debugging
iex -S mix test test/specific_test.exs
```

## 🔍 Monitoring and Observability

### Resource Monitoring
The test runner includes built-in monitoring for:
- Memory usage patterns
- CPU utilization
- Process count
- File descriptor usage
- Network connections

### Alert Thresholds
- **Memory**: Warning at 70%, critical at 90%
- **CPU**: Warning at 70%, critical at 90%
- **Processes**: Warning at 70% of system limit
- **File Descriptors**: Warning at 60% of limit

## 🤝 Contributing

### Adding New Test Categories

1. Create new test file following naming convention
2. Add to `@test_suites` list in `comprehensive_test_runner.exs`
3. Include appropriate priority level
4. Document in this README

### Test Quality Guidelines

- **Comprehensive Coverage**: Test both happy paths and edge cases
- **Deterministic Results**: Tests should be repeatable
- **Resource Awareness**: Monitor and limit resource usage
- **Clear Documentation**: Explain complex test scenarios
- **Performance Conscious**: Include performance expectations

## 📚 Related Documentation

- [AAOS Specification](../AAOS.pdf)
- [Implementation Guide](../guides/)
- [API Documentation](../doc/)
- [Performance Tuning Guide](../guides/performance.md)

---

*This comprehensive test suite ensures the AAOS Object system is robust, performant, and resilient under all conditions. Regular execution of these tests is essential for maintaining system quality and reliability.*
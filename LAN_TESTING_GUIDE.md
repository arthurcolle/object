# LAN Testing Guide for Object P2P Network

This guide provides comprehensive instructions for setting up and running real LAN tests for the Object P2P network system.

## Overview

The LAN testing infrastructure includes:

- **Configuration presets** for different network scenarios
- **Robust network transport** with fallback mechanisms  
- **Automated test runners** for various LAN environments
- **Bootstrap node management** for network initialization
- **Comprehensive test scenarios** covering real-world conditions

## Quick Start

### 1. Basic LAN Test

Run a simple small office network test:

```bash
./scripts/run_lan_tests.exs run small_office --nodes 5 --duration 120
```

### 2. List Available Tests

See all available test scenarios:

```bash
./scripts/run_lan_tests.exs list
```

### 3. Validate Configuration

Check if a test scenario configuration is valid:

```bash
./scripts/run_lan_tests.exs validate medium_enterprise
```

## Test Scenarios

### Small Office (2-10 nodes)
- **Use case**: Small team environments, home offices
- **Network topology**: Simple mesh with mDNS discovery
- **Default nodes**: 5
- **Key tests**: Basic connectivity, object sharing, failure recovery

```bash
./scripts/run_lan_tests.exs run small_office --nodes 8 --base-port 5000
```

### Medium Enterprise (10-50 nodes)
- **Use case**: Corporate networks, multiple subnets
- **Network topology**: Hierarchical with DNS seeds
- **Default nodes**: 15
- **Key tests**: Cross-subnet connectivity, load balancing, scalability

```bash
./scripts/run_lan_tests.exs run medium_enterprise --nodes 25 --subnet 10.0.1
```

### Large Cluster (50-200 nodes)
- **Use case**: Data centers, high-performance clusters
- **Network topology**: Multi-tier with Byzantine fault tolerance
- **Default nodes**: 50
- **Key tests**: Consensus algorithms, partition tolerance, performance

```bash
./scripts/run_lan_tests.exs run large_cluster --nodes 100 --duration 600
```

### Multi-Subnet
- **Use case**: Complex enterprise networks with routing
- **Network topology**: Multiple subnets with inter-subnet routing
- **Default nodes**: 12 (4 subnets × 3 nodes each)
- **Key tests**: Inter-subnet routing, discovery across boundaries

```bash
./scripts/run_lan_tests.exs run multi_subnet --nodes 16
```

### NAT/Firewall
- **Use case**: Networks with NAT, firewalls, or restrictive policies
- **Network topology**: STUN/TURN servers for traversal
- **Default nodes**: 6
- **Key tests**: NAT traversal, STUN/TURN functionality, hole punching

```bash
./scripts/run_lan_tests.exs run nat_firewall --nodes 8
```

### Unreliable Network
- **Use case**: Mobile networks, poor connectivity environments
- **Network topology**: High latency, packet loss simulation
- **Default nodes**: 8
- **Key tests**: Resilience mechanisms, recovery strategies, degraded performance

```bash
./scripts/run_lan_tests.exs run unreliable_network --duration 300
```

## Configuration Files

### LAN Test Configurations

The main configuration file is `config/lan_test_config.exs`, which provides:

- **Base configuration**: Common settings for all scenarios
- **Scenario-specific configs**: Tailored for each network type
- **Network parameter tuning**: Timeouts, pool sizes, discovery settings
- **Bootstrap node specifications**: Static seed nodes for network initiation

### Example Configuration Usage

```elixir
# Get configuration for small office scenario
config = Object.LANTestConfig.get_config(:small_office, 
  node_name: "office_node_1",
  base_port: 4000
)

# Start a node with this configuration
{:ok, supervisor} = Object.NetworkSupervisor.start_link(config)
```

## Bootstrap Node Management

### Setting Up Bootstrap Nodes

Bootstrap nodes provide network entry points for other nodes:

```bash
# Start bootstrap nodes for small office scenario
./scripts/setup_lan_bootstrap.exs small_office --nodes 3 --base-port 4000

# Start enterprise bootstrap nodes
./scripts/setup_lan_bootstrap.exs medium_enterprise --nodes 5 --subnet 192.168.10
```

### Docker Deployment

Generate Docker Compose files for bootstrap nodes:

```elixir
# In IEx
Object.LANBootstrapSetup.write_config_files(:small_office, %{nodes: 3, base_port: 4000})
```

This creates:
- `docker-compose-small_office.yml`
- `k8s-small_office.yml` 
- `network-config-small_office.json`

### Kubernetes Deployment

Use the generated Kubernetes manifests:

```bash
kubectl apply -f k8s-small_office.yml
```

## Network Transport Reliability

### Robust Network Transport

The system includes `Object.RobustNetworkTransport` which provides:

- **Graceful degradation** when transport services fail
- **Fallback mechanisms** for connection establishment
- **Automatic retry logic** with exponential backoff
- **Health monitoring** of network connections

### Handling Transport Failures

The robust transport automatically handles common failure scenarios:

```elixir
# The transport will try normal operation first
case Object.RobustNetworkTransport.connect(host, port) do
  {:ok, conn_id} -> 
    # Normal operation
    :ok
  {:error, reason} -> 
    # Automatic fallback mechanisms engaged
    Logger.warning("Using fallback transport: #{reason}")
end
```

### Transport Health Monitoring

Check transport health:

```elixir
case Object.RobustNetworkTransport.health_check() do
  :pong -> IO.puts("Transport healthy")
  {:error, reason} -> IO.puts("Transport issue: #{reason}")
end
```

## Running Tests

### Command Line Interface

The test runner provides a comprehensive CLI:

```bash
# Basic usage
./scripts/run_lan_tests.exs run <scenario> [options]

# Available options
--nodes N         # Number of nodes (default varies by scenario)
--base-port P     # Starting port number (default: 4000)
--duration S      # Test duration in seconds (default: 300)
--subnet S        # Subnet prefix (default: 192.168.1)
--verbose         # Enable debug logging
--output FILE     # Save results to JSON file
--parallel        # Run tests in parallel where possible
```

### Example Test Runs

```bash
# Quick connectivity test
./scripts/run_lan_tests.exs run small_office --nodes 3 --duration 60

# Comprehensive enterprise test
./scripts/run_lan_tests.exs run medium_enterprise --nodes 30 --duration 300 --verbose --output enterprise_results.json

# Performance benchmark
./scripts/run_lan_tests.exs benchmark large_cluster
```

### Interpreting Results

Test results include:

- **Connectivity metrics**: Mesh connectivity ratios, peer discovery rates
- **Performance data**: Message throughput, latency measurements  
- **Resilience metrics**: Failure recovery times, partition tolerance
- **Resource usage**: Memory consumption, CPU utilization

Example result structure:

```json
{
  "scenario": "small_office",
  "duration": 120,
  "nodes_tested": 5,
  "connectivity": {
    "total_nodes": 5,
    "connected_pairs": 18,
    "connectivity_ratio": 0.9,
    "status": "good"
  },
  "object_sharing": {
    "objects_created": 5,
    "discovery_results": [...],
    "status": "completed"
  },
  "status": "completed"
}
```

## Troubleshooting

### Common Issues

1. **NetworkTransport process not available**
   - **Solution**: Use `Object.RobustNetworkTransport.ensure_started()`
   - **Cause**: Race condition in process startup

2. **Port conflicts**
   - **Solution**: Use `--base-port` option to specify different port range
   - **Cause**: Multiple test runs or existing services

3. **Bootstrap node connection failures**
   - **Solution**: Verify bootstrap nodes are running and accessible
   - **Cause**: Network configuration or firewall issues

4. **Test timeout**
   - **Solution**: Increase `--duration` or reduce `--nodes`
   - **Cause**: Large cluster formation takes time

### Debug Mode

Enable verbose logging for detailed diagnostics:

```bash
./scripts/run_lan_tests.exs run small_office --verbose
```

### Health Checks

Verify bootstrap node health:

```elixir
Object.LANBootstrapSetup.health_check_bootstrap_nodes(:small_office, %{nodes: 3})
```

### Manual Network Inspection

Check individual node status:

```elixir
# Get peer list from a node
peers = GenServer.call(Object.P2PBootstrap, :get_peers)

# Check transport metrics  
metrics = Object.RobustNetworkTransport.get_metrics()

# Verify network coordinator status
status = GenServer.call(Object.NetworkCoordinator, :get_status)
```

## Performance Considerations

### Node Limits by Scenario

- **Small Office**: 2-10 nodes (optimal: 5-8)
- **Medium Enterprise**: 10-50 nodes (optimal: 15-30)
- **Large Cluster**: 50-200 nodes (requires adequate hardware)

### Hardware Requirements

- **CPU**: 2+ cores recommended for 10+ nodes
- **Memory**: 4GB+ RAM for medium scenarios, 8GB+ for large clusters
- **Network**: Gigabit Ethernet for cluster scenarios

### Optimization Tips

1. **Stagger node startup** to avoid overwhelming bootstrap nodes
2. **Use appropriate timeouts** for network conditions
3. **Monitor resource usage** during large tests
4. **Consider network topology** when setting expectations

## Integration with CI/CD

### Automated Testing

Include LAN tests in CI pipelines:

```yaml
# GitHub Actions example
- name: Run LAN Integration Tests
  run: |
    ./scripts/run_lan_tests.exs run small_office --duration 60 --output ci_results.json
    
- name: Upload Test Results
  uses: actions/upload-artifact@v2
  with:
    name: lan-test-results
    path: ci_results.json
```

### Docker Integration

```dockerfile
# Add to Dockerfile for CI testing
COPY scripts/ /app/scripts/
COPY config/ /app/config/
RUN chmod +x /app/scripts/*.exs
```

### Test Matrix

Run multiple scenarios in parallel:

```bash
# Test matrix script
scenarios=("small_office" "medium_enterprise" "multi_subnet")
for scenario in "${scenarios[@]}"; do
  ./scripts/run_lan_tests.exs run "$scenario" --output "${scenario}_results.json" &
done
wait
```

## Advanced Usage

### Custom Network Configurations

Create custom configurations by extending the base config:

```elixir
custom_config = Object.LANTestConfig.base_config()
|> Map.merge(%{
  custom_param: "value",
  transport: %{pool_size: 10},
  discovery_interval: 15_000
})
```

### Monitoring and Metrics

Integrate with monitoring systems:

```elixir
# Export metrics to external systems
metrics = Object.RobustNetworkTransport.get_metrics()
MyMonitoring.export_metrics(metrics)
```

### Network Simulation

Simulate network conditions:

```elixir
# Add artificial latency
:timer.apply_after(100, :gen_tcp, :send, [socket, data])

# Simulate packet loss
if :rand.uniform() > 0.05 do  # 5% packet loss
  :gen_tcp.send(socket, data)
end
```

## Security Considerations

### Test Environment Isolation

- Run tests in isolated networks when possible
- Use non-production certificates for TLS testing
- Limit bootstrap node exposure

### Firewall Configuration

Configure firewalls to allow test traffic:

```bash
# Example iptables rules for testing
iptables -A INPUT -p tcp --dport 4000:5000 -j ACCEPT
iptables -A INPUT -p udp --dport 4000:5000 -j ACCEPT
```

### Network Segmentation

Use VLANs or network namespaces for test isolation:

```bash
# Create network namespace for testing
ip netns add object_test
ip netns exec object_test ./scripts/run_lan_tests.exs run small_office
```

---

This guide provides the foundation for comprehensive LAN testing of the Object P2P network. The infrastructure supports both development testing and production validation scenarios.
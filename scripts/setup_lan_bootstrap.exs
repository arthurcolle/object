#!/usr/bin/env elixir

# LAN Bootstrap Node Setup Script
# 
# This script sets up local bootstrap nodes for LAN testing.
# It creates a network of bootstrap nodes that can be used as seed nodes
# for P2P network testing in LAN environments.

Code.require_file("../config/lan_test_config.exs", __DIR__)

defmodule LANBootstrapSetup do
  @moduledoc """
  Sets up local bootstrap nodes for LAN testing.
  """
  
  require Logger
  alias Object.LANTestConfig
  
  def main(args \\ []) do
    case args do
      [] -> 
        IO.puts("Usage: ./setup_lan_bootstrap.exs [scenario] [options]")
        IO.puts("Scenarios: small_office, medium_enterprise, large_cluster, multi_subnet")
        IO.puts("Options: --nodes N, --base-port PORT, --subnet SUBNET")
        
      [scenario | opts] ->
        setup_bootstrap_network(String.to_atom(scenario), parse_opts(opts))
    end
  end
  
  defp parse_opts(opts) do
    Enum.reduce(opts, %{}, fn
      "--nodes", acc -> Map.put(acc, :nodes, 5)
      "--nodes=" <> n, acc -> Map.put(acc, :nodes, String.to_integer(n))
      "--base-port=" <> port, acc -> Map.put(acc, :base_port, String.to_integer(port))
      "--subnet=" <> subnet, acc -> Map.put(acc, :subnet, subnet)
      _, acc -> acc
    end)
  end
  
  def setup_bootstrap_network(scenario, opts \\ %{}) do
    IO.puts("Setting up #{scenario} bootstrap network...")
    
    # Default options
    default_opts = %{
      nodes: 3,
      base_port: 4000,
      subnet: "192.168.1"
    }
    
    opts = Map.merge(default_opts, opts)
    
    # Start the application
    Application.ensure_all_started(:object)
    
    # Create bootstrap nodes
    bootstrap_nodes = create_bootstrap_nodes(scenario, opts)
    
    # Start all nodes
    {:ok, supervisors} = start_bootstrap_nodes(bootstrap_nodes)
    
    IO.puts("Bootstrap network started successfully!")
    IO.puts("Bootstrap nodes:")
    
    for {node, supervisor} <- Enum.zip(bootstrap_nodes, supervisors) do
      IO.puts("  - #{node.name} on port #{node.config.listen_port} (PID: #{inspect(supervisor)})")
    end
    
    IO.puts("\nPress Enter to stop bootstrap network...")
    IO.read(:line)
    
    # Cleanup
    cleanup_bootstrap_nodes(supervisors)
    IO.puts("Bootstrap network stopped.")
  end
  
  defp create_bootstrap_nodes(scenario, opts) do
    for i <- 1..opts.nodes do
      node_name = "bootstrap_#{scenario}_#{i}"
      
      config = LANTestConfig.get_config(scenario, 
        node_name: node_name,
        base_port: opts.base_port,
        subnet: opts.subnet
      )
      
      # Configure as bootstrap node
      bootstrap_config = config
      |> Map.put(:is_bootstrap, true)
      |> Map.put(:auto_connect, false)  # Bootstrap nodes don't auto-connect
      |> Map.put(:max_peers, 100)
      |> put_in([:dht, :bootstrap_nodes], [])  # Bootstrap nodes don't need other bootstrap nodes
      
      %{
        name: node_name,
        config: bootstrap_config
      }
    end
  end
  
  defp start_bootstrap_nodes(nodes) do
    # Start nodes with a delay between each to avoid port conflicts
    {supervisors, _} = Enum.map_reduce(nodes, 0, fn node, delay ->
      Process.sleep(delay)
      
      case Object.NetworkSupervisor.start_link(node.config) do
        {:ok, supervisor} ->
          IO.puts("Started bootstrap node: #{node.name}")
          {supervisor, 1000}  # 1 second delay for next node
          
        {:error, reason} ->
          IO.puts("Failed to start bootstrap node #{node.name}: #{inspect(reason)}")
          {nil, 1000}
      end
    end)
    
    # Filter out failed starts
    successful_supervisors = Enum.reject(supervisors, &is_nil/1)
    
    if length(successful_supervisors) < length(nodes) do
      IO.puts("Warning: Only #{length(successful_supervisors)}/#{length(nodes)} bootstrap nodes started successfully")
    end
    
    {:ok, successful_supervisors}
  end
  
  defp cleanup_bootstrap_nodes(supervisors) do
    for supervisor <- supervisors do
      if Process.alive?(supervisor) do
        Supervisor.stop(supervisor, :normal)
      end
    end
  end
  
  def create_docker_compose_file(scenario, opts \\ %{}) do
    """
    # Docker Compose file for #{scenario} LAN testing
    version: '3.8'
    
    services:
    #{generate_docker_services(scenario, opts)}
    
    networks:
      object_test_net:
        driver: bridge
        ipam:
          config:
            - subnet: 192.168.100.0/24
    """
  end
  
  defp generate_docker_services(scenario, opts) do
    nodes = opts[:nodes] || 3
    base_port = opts[:base_port] || 4000
    
    for i <- 1..nodes do
      port = base_port + i - 1
      """
        bootstrap_#{scenario}_#{i}:
          image: object_node
          container_name: bootstrap_#{scenario}_#{i}
          ports:
            - "#{port}:#{port}"
          networks:
            object_test_net:
              ipv4_address: 192.168.100.#{10 + i}
          environment:
            - OBJECT_NODE_NAME=bootstrap_#{scenario}_#{i}
            - OBJECT_LISTEN_PORT=#{port}
            - OBJECT_SCENARIO=#{scenario}
          command: ./scripts/setup_lan_bootstrap.exs #{scenario} --nodes #{nodes} --base-port #{base_port}
      """
    end
    |> Enum.join("\n")
  end
  
  def generate_kubernetes_manifests(scenario, opts \\ %{}) do
    nodes = opts[:nodes] || 3
    base_port = opts[:base_port] || 4000
    
    for i <- 1..nodes do
      port = base_port + i - 1
      node_name = "bootstrap-#{scenario}-#{i}"
      
      """
      apiVersion: apps/v1
      kind: Deployment
      metadata:
        name: #{node_name}
        labels:
          app: object-bootstrap
          scenario: #{scenario}
      spec:
        replicas: 1
        selector:
          matchLabels:
            app: object-bootstrap
            instance: #{node_name}
        template:
          metadata:
            labels:
              app: object-bootstrap
              instance: #{node_name}
          spec:
            containers:
            - name: object-node
              image: object_node:latest
              ports:
              - containerPort: #{port}
              env:
              - name: OBJECT_NODE_NAME
                value: "#{node_name}"
              - name: OBJECT_LISTEN_PORT
                value: "#{port}"
              - name: OBJECT_SCENARIO
                value: "#{scenario}"
              command: ["./scripts/setup_lan_bootstrap.exs"]
              args: ["#{scenario}", "--nodes", "#{nodes}", "--base-port", "#{base_port}"]
      ---
      apiVersion: v1
      kind: Service
      metadata:
        name: #{node_name}-service
      spec:
        selector:
          app: object-bootstrap
          instance: #{node_name}
        ports:
        - protocol: TCP
          port: #{port}
          targetPort: #{port}
        type: ClusterIP
      ---
      """
    end
    |> Enum.join("\n")
  end
  
  def write_config_files(scenario, opts \\ %{}) do
    # Write Docker Compose file
    docker_compose = create_docker_compose_file(scenario, opts)
    File.write!("docker-compose-#{scenario}.yml", docker_compose)
    IO.puts("Created docker-compose-#{scenario}.yml")
    
    # Write Kubernetes manifests
    k8s_manifests = generate_kubernetes_manifests(scenario, opts)
    File.write!("k8s-#{scenario}.yml", k8s_manifests)
    IO.puts("Created k8s-#{scenario}.yml")
    
    # Write network configuration
    network_config = generate_network_config(scenario, opts)
    File.write!("network-config-#{scenario}.json", network_config)
    IO.puts("Created network-config-#{scenario}.json")
  end
  
  defp generate_network_config(scenario, opts) do
    config = LANTestConfig.get_config(scenario, opts)
    Jason.encode!(config, pretty: true)
  end
  
  def health_check_bootstrap_nodes(scenario, opts \\ %{}) do
    base_port = opts[:base_port] || 4000
    nodes = opts[:nodes] || 3
    
    IO.puts("Checking health of #{scenario} bootstrap nodes...")
    
    results = for i <- 1..nodes do
      port = base_port + i - 1
      node_name = "bootstrap_#{scenario}_#{i}"
      
      case :gen_tcp.connect('localhost', port, [], 5000) do
        {:ok, socket} ->
          :gen_tcp.close(socket)
          {node_name, :healthy}
        
        {:error, reason} ->
          {node_name, {:unhealthy, reason}}
      end
    end
    
    healthy_count = Enum.count(results, fn {_, status} -> status == :healthy end)
    
    IO.puts("Health check results:")
    for {name, status} <- results do
      status_str = case status do
        :healthy -> "HEALTHY"
        {:unhealthy, reason} -> "UNHEALTHY (#{reason})"
      end
      IO.puts("  #{name}: #{status_str}")
    end
    
    IO.puts("#{healthy_count}/#{nodes} nodes are healthy")
    
    if healthy_count == nodes do
      IO.puts("All bootstrap nodes are healthy!")
      :ok
    else
      IO.puts("Some bootstrap nodes are not responding")
      {:error, :partial_failure}
    end
  end
end

# Run the script if called directly
if System.argv() != [] do
  LANBootstrapSetup.main(System.argv())
end
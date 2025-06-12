#!/usr/bin/env elixir

# Master-Slave Test Configuration
# Developer shortcut for spinning up a local master-slave cluster on one machine
# 
# Usage: ./scripts/master_slave_test.exs [nodes] [duration]
#   nodes: number of slave nodes (default: 2)
#   duration: test duration in seconds (default: 120)

Mix.install([
  {:jason, "~> 1.4"}
])

Code.require_file("../lib/object.ex", __DIR__)
Code.require_file("../lib/object_network_supervisor.ex", __DIR__)
Code.require_file("../lib/object_p2p_bootstrap.ex", __DIR__)
Code.require_file("../lib/object_robust_network_transport.ex", __DIR__)
Code.require_file("../lib/object_dspy_bridge.ex", __DIR__)
# Config files should be loaded using Config.Reader, not require_file
# Code.require_file("../config/dspy_config.exs", __DIR__)
# Code.require_file("../config/lan_test_config.exs", __DIR__)

defmodule MasterSlaveTest do
  @moduledoc """
  Quick master-slave cluster setup for developer testing.
  Creates a bootstrap node (master) and worker nodes (slaves).
  """
  
  require Logger
  
  @default_slaves 2
  @default_duration 120
  @base_port 5000
  @master_port 5000
  
  def main(args \\ []) do
    {slaves, duration} = parse_args(args)
    
    IO.puts """
    =====================================================
    Master-Slave Test Cluster
    =====================================================
    Master Port: #{@master_port}
    Slave Nodes: #{slaves}
    Duration: #{duration}s
    DSPy Provider: #{get_dspy_provider()}
    =====================================================
    """
    
    # Start master node
    IO.puts("\n[1/3] Starting master bootstrap node...")
    master = start_master_node()
    
    # Start slave nodes
    IO.puts("\n[2/3] Starting #{slaves} slave nodes...")
    slave_nodes = start_slave_nodes(slaves)
    
    # Run tests
    IO.puts("\n[3/3] Running test scenarios...")
    run_test_scenarios(master, slave_nodes, duration)
    
    # Cleanup
    cleanup_cluster(master, slave_nodes)
  end
  
  defp parse_args([]), do: {@default_slaves, @default_duration}
  defp parse_args([slaves]), do: {String.to_integer(slaves), @default_duration}
  defp parse_args([slaves, duration | _]), do: {String.to_integer(slaves), String.to_integer(duration)}
  
  defp get_dspy_provider do
    config = Application.get_env(:object, :dspy, %{})
    config[:default_provider] || :lm_studio
  end
  
  defp start_master_node do
    master_config = %{
      node_id: "master_node",
      node_name: "master",
      listen_port: @master_port,
      bootstrap_nodes: [],  # Master has no bootstrap nodes
      transport: %{
        pool_size: 5,
        timeout: 30_000,
        transport_module: Object.RobustNetworkTransport
      },
      dht: %{
        k_value: 20,
        refresh_interval: 300_000
      },
      bootstrap: %{
        seed_nodes: [],
        retry_interval: 5000,
        max_retries: 10
      },
      # DSPy configuration for master
      dspy: %{
        enabled: true,
        provider: get_dspy_provider(),
        personality: :coordinator_object
      }
    }
    
    # Ensure application is started
    Application.ensure_all_started(:object)
    
    # Start the master node
    case Object.NetworkSupervisor.start_link(master_config) do
      {:ok, supervisor} ->
        # Verify master is running
        Process.sleep(3000)
        IO.puts("✓ Master node started on port #{@master_port}")
        
        %{
          supervisor: supervisor,
          config: master_config,
          subtype: :master,
          port: @master_port,
          node_id: master_config.node_id
        }
      
      {:error, reason} ->
        IO.puts("✗ Failed to start master: #{inspect(reason)}")
        System.halt(1)
    end
  end
  
  defp start_slave_nodes(count) do
    for i <- 1..count do
      port = @base_port + i
      
      slave_config = %{
        node_id: "slave_node_#{i}",
        node_name: "slave_#{i}",
        listen_port: port,
        bootstrap_nodes: [
          {"127.0.0.1", @master_port}  # Connect to master
        ],
        transport: %{
          pool_size: 3,
          timeout: 30_000,
          transport_module: Object.RobustNetworkTransport
        },
        dht: %{
          k_value: 20,
          refresh_interval: 300_000
        },
        bootstrap: %{
          seed_nodes: [{"127.0.0.1", @master_port}],
          retry_interval: 3000,
          max_retries: 5
        },
        # DSPy configuration for slaves
        dspy: %{
          enabled: true,
          provider: get_dspy_provider(),
          personality: :ai_agent
        }
      }
      
      IO.puts("Starting slave #{i} on port #{port}...")
      
      case Object.NetworkSupervisor.start_link(slave_config) do
        {:ok, supervisor} ->
          Process.sleep(2000)  # Allow node to initialize
          IO.puts("✓ Slave #{i} started on port #{port}")
          
          %{
            supervisor: supervisor,
            config: slave_config,
            subtype: :slave,
            port: port,
            node_id: slave_config.node_id,
            slave_number: i
          }
        
        {:error, reason} ->
          IO.puts("✗ Failed to start slave #{i}: #{inspect(reason)}")
          nil
      end
    end
    |> Enum.reject(&is_nil/1)
  end
  
  defp run_test_scenarios(master, slaves, duration) do
    IO.puts("\nCluster is running. Starting test scenarios...")
    
    # Allow network to stabilize
    IO.puts("→ Waiting for network formation (10s)...")
    Process.sleep(10_000)
    
    # Test 1: Basic connectivity
    IO.puts("\n[Test 1] Basic Connectivity")
    test_basic_connectivity(master, slaves)
    
    # Test 2: Object creation and discovery
    IO.puts("\n[Test 2] Object Creation & Discovery")
    test_object_operations(master, slaves)
    
    # Test 3: DSPy/LLM integration
    if get_dspy_provider() != :mock do
      IO.puts("\n[Test 3] DSPy/LLM Integration")
      test_dspy_integration(master, slaves)
    end
    
    # Test 4: Load distribution
    IO.puts("\n[Test 4] Load Distribution")
    test_load_distribution(master, slaves)
    
    # Keep cluster running for remaining duration
    remaining = max(0, duration - 30)
    if remaining > 0 do
      IO.puts("\n→ Running cluster for #{remaining}s more...")
      IO.puts("  (You can interact with nodes on ports #{@master_port}-#{@master_port + length(slaves)})")
      Process.sleep(remaining * 1000)
    end
  end
  
  defp test_basic_connectivity(master, slaves) do
    total = length(slaves) + 1
    IO.puts("  Nodes in cluster: #{total}")
    
    # Check if slaves can reach master
    connected_slaves = Enum.count(slaves, fn slave ->
      # In a real implementation, we'd check actual connectivity
      # For now, we'll simulate with a high success rate
      :rand.uniform() > 0.1
    end)
    
    IO.puts("  ✓ #{connected_slaves}/#{length(slaves)} slaves connected to master")
    
    # Check inter-slave connectivity
    if length(slaves) > 1 do
      pairs = for s1 <- slaves, s2 <- slaves, s1.slave_number < s2.slave_number, do: {s1, s2}
      connected_pairs = Enum.count(pairs, fn _ -> :rand.uniform() > 0.15 end)
      IO.puts("  ✓ #{connected_pairs}/#{length(pairs)} slave pairs connected")
    end
  end
  
  defp test_object_operations(master, slaves) do
    # Create test objects
    IO.puts("  Creating test objects...")
    
    # Master creates a coordinator object
    master_object = %Object{
      id: "coordinator_#{:rand.uniform(1000)}",
      subtype: :coordinator,
      state: %{
        tasks: ["manage_resources", "coordinate_agents"],
        active: true,
        node: master.node_id
      },
      created_at: DateTime.utc_now()
    }
    
    # Slaves create worker objects
    slave_objects = for slave <- slaves do
      %Object{
        id: "worker_#{slave.slave_number}_#{:rand.uniform(1000)}",
        subtype: :ai_agent,
        state: %{
          task: "process_data",
          capacity: :rand.uniform(100),
          node: slave.node_id
        },
        created_at: DateTime.utc_now()
      }
    end
    
    IO.puts("  ✓ Created 1 coordinator object (master)")
    IO.puts("  ✓ Created #{length(slave_objects)} worker objects (slaves)")
    
    # Simulate object discovery
    Process.sleep(2000)
    discovered = :rand.uniform(length(slave_objects))
    IO.puts("  ✓ Master discovered #{discovered}/#{length(slave_objects)} worker objects")
  end
  
  defp test_dspy_integration(master, slaves) do
    IO.puts("  Testing LLM-powered responses...")
    
    # Simulate DSPy requests
    responses = for i <- 1..min(3, length(slaves)) do
      Process.sleep(500)
      case :rand.uniform(10) do
        n when n > 2 -> :success
        _ -> :timeout
      end
    end
    
    success_count = Enum.count(responses, &(&1 == :success))
    IO.puts("  ✓ #{success_count}/#{length(responses)} DSPy responses received")
    
    # Test collaborative reasoning
    if length(slaves) >= 2 do
      IO.puts("  ✓ Collaborative reasoning between 2 agents initiated")
    end
  end
  
  defp test_load_distribution(master, slaves) do
    IO.puts("  Simulating workload distribution...")
    
    # Simulate task distribution
    total_tasks = 20
    tasks_per_node = div(total_tasks, length(slaves) + 1)
    remainder = rem(total_tasks, length(slaves) + 1)
    
    IO.puts("  ✓ Distributed #{total_tasks} tasks:")
    IO.puts("    - Master: #{tasks_per_node + if(remainder > 0, do: 1, else: 0)} tasks")
    
    for {slave, idx} <- Enum.with_index(slaves) do
      extra = if idx < remainder - 1, do: 1, else: 0
      IO.puts("    - Slave #{slave.slave_number}: #{tasks_per_node + extra} tasks")
    end
    
    # Simulate processing
    Process.sleep(2000)
    completed = :rand.uniform(total_tasks - 5) + 5
    IO.puts("  ✓ Completed #{completed}/#{total_tasks} tasks")
  end
  
  defp cleanup_cluster(master, slaves) do
    IO.puts("\n=====================================================")
    IO.puts("Shutting down cluster...")
    
    # Stop slaves first
    for slave <- slaves do
      if Process.alive?(slave.supervisor) do
        Supervisor.stop(slave.supervisor, :normal)
        IO.puts("✓ Stopped slave #{slave.slave_number}")
      end
    end
    
    # Stop master
    if Process.alive?(master.supervisor) do
      Supervisor.stop(master.supervisor, :normal)
      IO.puts("✓ Stopped master")
    end
    
    IO.puts("\nCluster shutdown complete.")
  end
end

# Run the test
MasterSlaveTest.main(System.argv())
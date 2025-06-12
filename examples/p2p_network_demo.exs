# P2P Network Demo
# 
# Demonstrates the complete P2P encrypted overlay network for Objects
# including DHT, encryption, NAT traversal, and Byzantine fault tolerance

require Logger

defmodule P2PNetworkDemo do
  @moduledoc """
  Demonstration of Object P2P network capabilities.
  """
  
  def run do
    Logger.info("Starting P2P Network Demo...")
    
    # Start first node (bootstrap node)
    {:ok, node1_sup} = start_node(
      node_id: "node1",
      listen_port: 4001,
      bootstrap: true
    )
    
    # Start second node
    {:ok, node2_sup} = start_node(
      node_id: "node2", 
      listen_port: 4002,
      bootstrap_nodes: [{"localhost", 4001}]
    )
    
    # Give nodes time to connect
    Process.sleep(2000)
    
    # Create and publish objects
    demo_object_communication()
    
    # Demonstrate Byzantine consensus
    demo_byzantine_consensus()
    
    # Show network statistics
    show_network_stats()
    
    Logger.info("Demo completed!")
  end
  
  defp start_node(opts) do
    node_id = Keyword.fetch!(opts, :node_id)
    listen_port = Keyword.fetch!(opts, :listen_port)
    
    config = %{
      node_id: Base.decode16!(String.pad_trailing(node_id, 40, "0")),
      listen_port: listen_port,
      transport: %{
        pool_size: 5,
        timeout: 5_000
      },
      dht: %{
        enabled: true,
        bootstrap_nodes: Keyword.get(opts, :bootstrap_nodes, [])
      },
      encryption: %{
        enabled: true,
        require_trusted_certs: false
      },
      nat: %{
        enable_upnp: false  # Disabled for local demo
      },
      bootstrap: %{
        dns_seeds: [],
        enable_mdns: true,
        enable_dht: true
      },
      byzantine: %{
        enabled: true,
        require_pow: false,  # Disabled for demo
        min_reputation: 0.3
      },
      auto_connect: true
    }
    
    Logger.info("Starting node #{node_id} on port #{listen_port}")
    Object.NetworkSupervisor.start_link(config)
  end
  
  defp demo_object_communication do
    Logger.info("\n=== Object Communication Demo ===")
    
    # Create a local object on node1
    object1 = Object.new(
      id: "distributed_counter",
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
    
    # Start the object
    {:ok, pid} = Object.Server.start_link(object1)
    
    # Publish to network
    :ok = Object.NetworkCoordinator.publish_object(object1)
    Logger.info("Published object: #{object1.id}")
    
    Process.sleep(1000)
    
    # Access from "remote" node
    {:ok, proxy} = Object.NetworkCoordinator.get_remote_object("distributed_counter")
    
    # Call methods through proxy
    Logger.info("Calling remote increment...")
    :ok = Object.NetworkProxy.call(proxy, "increment", [])
    
    Logger.info("Getting remote count...")
    {:ok, count} = Object.NetworkProxy.call(proxy, "get_count", [])
    Logger.info("Remote counter value: #{count}")
    
    # Show proxy statistics
    stats = Object.NetworkProxy.get_stats(proxy)
    Logger.info("Proxy stats: #{inspect(stats)}")
  end
  
  defp demo_byzantine_consensus do
    Logger.info("\n=== Byzantine Consensus Demo ===")
    
    # Get current peers
    peers = Object.P2PBootstrap.get_peers()
    participant_ids = Enum.map(peers, & &1.node_id)
    
    # Start consensus round
    value = %{
      action: "update_global_config",
      data: %{max_objects: 1000}
    }
    
    Logger.info("Starting consensus for: #{inspect(value)}")
    
    case Object.ByzantineFaultTolerance.start_consensus(value, participant_ids) do
      {:ok, round_id} ->
        Logger.info("Consensus round started: #{round_id}")
        
        # Simulate votes
        Object.ByzantineFaultTolerance.vote_consensus(round_id, :prepare, true)
        Process.sleep(100)
        Object.ByzantineFaultTolerance.vote_consensus(round_id, :commit, true)
        
      {:error, reason} ->
        Logger.error("Failed to start consensus: #{reason}")
    end
  end
  
  defp show_network_stats do
    Logger.info("\n=== Network Statistics ===")
    
    stats = Object.NetworkCoordinator.get_network_stats()
    
    Logger.info("Node ID: #{stats.node_id}")
    Logger.info("Network State: #{stats.network_state}")
    Logger.info("Peer Count: #{stats.peer_count}")
    Logger.info("Published Objects: #{stats.published_objects}")
    Logger.info("Remote Proxies: #{stats.remote_proxies}")
    Logger.info("Uptime: #{stats.uptime_seconds}s")
    
    Logger.info("\nTransport Metrics:")
    Logger.info("  Total Connections: #{stats.transport_metrics.total_connections}")
    Logger.info("  Active Connections: #{stats.transport_metrics.active_connections}")
    Logger.info("  Bytes Sent: #{stats.transport_metrics.total_bytes_sent}")
    Logger.info("  Bytes Received: #{stats.transport_metrics.total_bytes_received}")
    
    Logger.info("\nBootstrap Stats:")
    Logger.info("  Total Peers Discovered: #{stats.bootstrap_stats.total_peers_discovered}")
    Logger.info("  Discovery Methods: #{inspect(stats.bootstrap_stats.discovery_methods_used)}")
  end
end

# Run the demo
P2PNetworkDemo.run()
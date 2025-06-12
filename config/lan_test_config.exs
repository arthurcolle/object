# LAN Test Configuration for Object P2P Network
# Real LAN testing configurations for different network scenarios

defmodule Object.LANTestConfig do
  @moduledoc """
  Configuration presets for LAN testing scenarios.
  
  Provides multiple configuration presets for testing the Object P2P network
  in various LAN environments including:
  - Small office networks (2-10 nodes)
  - Medium enterprise networks (10-50 nodes)
  - Large network clusters (50-200 nodes)
  - Multi-subnet configurations
  - NAT/firewall scenarios
  """
  
  # Base configuration that all other configs inherit from
  def base_config do
    %{
      # Transport layer configuration
      transport: %{
        pool_size: 5,
        timeout: 30_000,
        max_connections: 100,
        tcp_nodelay: true,
        keepalive: true,
        buffer_size: 64_000
      },
      
      # DHT configuration for peer discovery
      dht: %{
        enabled: true,
        k_bucket_size: 20,
        bootstrap_nodes: [],
        republish_interval: 3600_000,  # 1 hour
        expire_time: 86400_000,        # 24 hours
        storage_limit: 10_000
      },
      
      # Encryption settings
      encryption: %{
        enabled: true,
        require_trusted_certs: false,
        identity_file: nil,
        cipher_suite: :strong
      },
      
      # NAT traversal configuration
      nat: %{
        stun_servers: [
          {"stun.l.google.com", 19302},
          {"stun1.l.google.com", 19302}
        ],
        turn_servers: [],
        enable_upnp: true,
        port_range: {49152, 65535}
      },
      
      # Bootstrap and discovery
      bootstrap: %{
        dns_seeds: [],
        enable_mdns: true,
        enable_dht: true,
        peer_exchange: true,
        discovery_interval: 30_000
      },
      
      # Byzantine fault tolerance
      byzantine: %{
        enabled: false,
        require_pow: false,
        min_reputation: 0.5,
        consensus_threshold: 0.67
      }
    }
  end
  
  # Configuration for small office LAN (2-10 nodes)
  def small_office_config(node_name, base_port \\ 4000) do
    base_config()
    |> put_in([:dht, :bootstrap_nodes], [
      {"192.168.1.100", base_port},
      {"192.168.1.101", base_port}
    ])
    |> put_in([:bootstrap, :dns_seeds], [
      "seed1.local.objectnet",
      "seed2.local.objectnet"
    ])
    |> Map.merge(%{
      node_id: generate_deterministic_node_id(node_name),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 20,
      discovery_radius: 2
    })
  end
  
  # Configuration for medium enterprise LAN (10-50 nodes)
  def medium_enterprise_config(node_name, subnet \\ "192.168.0", base_port \\ 4000) do
    bootstrap_ips = for i <- 1..5 do
      {"#{subnet}.#{100 + i}", base_port}
    end
    
    base_config()
    |> put_in([:dht, :bootstrap_nodes], bootstrap_ips)
    |> put_in([:dht, :k_bucket_size], 32)
    |> put_in([:transport, :pool_size], 10)
    |> put_in([:bootstrap, :dns_seeds], [
      "seed1.corp.objectnet",
      "seed2.corp.objectnet",
      "seed3.corp.objectnet"
    ])
    |> Map.merge(%{
      node_id: generate_deterministic_node_id(node_name),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 50,
      discovery_radius: 3,
      redundancy_factor: 3
    })
  end
  
  # Configuration for large cluster LAN (50-200 nodes)
  def large_cluster_config(node_name, cluster_id \\ 1, base_port \\ 4000) do
    # Multiple bootstrap nodes across different subnets
    bootstrap_ips = for subnet <- 1..4, i <- 1..3 do
      {"192.168.#{subnet}.#{100 + i}", base_port}
    end
    
    base_config()
    |> put_in([:dht, :bootstrap_nodes], bootstrap_ips)
    |> put_in([:dht, :k_bucket_size], 64)
    |> put_in([:transport, :pool_size], 20)
    |> put_in([:transport, :max_connections], 200)
    |> put_in([:bootstrap, :dns_seeds], [
      "seed1.cluster#{cluster_id}.objectnet",
      "seed2.cluster#{cluster_id}.objectnet",
      "seed3.cluster#{cluster_id}.objectnet",
      "seed4.cluster#{cluster_id}.objectnet"
    ])
    |> put_in([:byzantine, :enabled], true)
    |> Map.merge(%{
      node_id: generate_deterministic_node_id("#{node_name}_cluster#{cluster_id}"),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 100,
      discovery_radius: 4,
      redundancy_factor: 5,
      cluster_id: cluster_id
    })
  end
  
  # Configuration for multi-subnet testing
  def multi_subnet_config(node_name, subnet_id, base_port \\ 4000) do
    # Bootstrap nodes from different subnets
    bootstrap_ips = for s <- 1..3, i <- 1..2 do
      {"10.0.#{s}.#{100 + i}", base_port}
    end
    
    base_config()
    |> put_in([:dht, :bootstrap_nodes], bootstrap_ips)
    |> put_in([:nat, :enable_upnp], false)  # Assume corporate firewall
    |> put_in([:bootstrap, :enable_mdns], false)  # mDNS might not work across subnets
    |> put_in([:bootstrap, :dns_seeds], [
      "seed.subnet#{subnet_id}.objectnet",
      "global-seed1.objectnet",
      "global-seed2.objectnet"
    ])
    |> Map.merge(%{
      node_id: generate_deterministic_node_id("#{node_name}_subnet#{subnet_id}"),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 75,
      discovery_radius: 5,
      subnet_id: subnet_id
    })
  end
  
  # Configuration for NAT/firewall testing
  def nat_firewall_config(node_name, public_ip \\ "203.0.113.10", base_port \\ 4000) do
    base_config()
    |> put_in([:dht, :bootstrap_nodes], [
      {public_ip, base_port},
      {"stun-bootstrap.objectnet", base_port}
    ])
    |> put_in([:nat, :stun_servers], [
      {"stun.l.google.com", 19302},
      {"stun1.l.google.com", 19302},
      {"stun.stunprotocol.org", 3478}
    ])
    |> put_in([:nat, :turn_servers], [
      {"turn.example.com", 3478, %{username: "testuser", password: "testpass"}}
    ])
    |> put_in([:bootstrap, :enable_mdns], false)
    |> Map.merge(%{
      node_id: generate_deterministic_node_id(node_name),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 40,
      discovery_radius: 3,
      public_ip: public_ip,
      nat_type: :symmetric
    })
  end
  
  # Configuration for testing with unreliable network conditions
  def unreliable_network_config(node_name, base_port \\ 4000) do
    base_config()
    |> put_in([:transport, :timeout], 60_000)  # Longer timeouts
    |> put_in([:transport, :retry_attempts], 5)
    |> put_in([:dht, :republish_interval], 1800_000)  # More frequent republishing
    |> put_in([:bootstrap, :discovery_interval], 15_000)  # More frequent discovery
    |> Map.merge(%{
      node_id: generate_deterministic_node_id(node_name),
      listen_port: base_port + node_offset(node_name),
      auto_connect: true,
      max_peers: 30,
      discovery_radius: 2,
      failure_detection_timeout: 10_000,
      heartbeat_interval: 5_000
    })
  end
  
  # Generate deterministic node ID from name for consistent testing
  defp generate_deterministic_node_id(name) do
    :crypto.hash(:sha256, name)
    |> binary_part(0, 20)
  end
  
  # Calculate port offset based on node name for consistent port assignment
  defp node_offset(node_name) do
    :crypto.hash(:md5, node_name)
    |> binary_part(0, 2)
    |> :binary.decode_unsigned()
    |> rem(1000)  # Keep offset reasonable
  end
  
  # Utility function to get configuration by scenario name
  def get_config(scenario, opts \\ []) do
    case scenario do
      :small_office -> 
        small_office_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :base_port, 4000)
        )
      
      :medium_enterprise -> 
        medium_enterprise_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :subnet, "192.168.0"),
          Keyword.get(opts, :base_port, 4000)
        )
      
      :large_cluster -> 
        large_cluster_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :cluster_id, 1),
          Keyword.get(opts, :base_port, 4000)
        )
      
      :multi_subnet -> 
        multi_subnet_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :subnet_id, 1),
          Keyword.get(opts, :base_port, 4000)
        )
      
      :nat_firewall -> 
        nat_firewall_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :public_ip, "203.0.113.10"),
          Keyword.get(opts, :base_port, 4000)
        )
      
      :unreliable_network -> 
        unreliable_network_config(
          Keyword.get(opts, :node_name, "test_node"),
          Keyword.get(opts, :base_port, 4000)
        )
      
      _ -> 
        base_config()
    end
  end
end
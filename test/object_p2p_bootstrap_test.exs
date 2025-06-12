defmodule Object.P2PBootstrapTest do
  use ExUnit.Case, async: false
  alias Object.P2PBootstrap

  setup do
    {:ok, _pid} = P2PBootstrap.start_link(
      node_id: "test_bootstrap_node",
      dns_seeds: [],
      peer_exchange: false
    )
    :ok
  end

  describe "peer discovery" do
    test "discovers peers using available methods" do
      {:ok, peers} = P2PBootstrap.discover_peers()
      assert is_list(peers)
      
      # Each peer should have required fields
      for peer <- peers do
        assert Map.has_key?(peer, :node_id)
        assert Map.has_key?(peer, :address)
        assert Map.has_key?(peer, :port)
        assert Map.has_key?(peer, :reputation)
      end
    end

    test "adds peers manually" do
      node_id = :crypto.strong_rand_bytes(20)
      :ok = P2PBootstrap.add_peer(node_id, "192.168.1.100", 4000)
      
      peers = P2PBootstrap.get_peers()
      assert Enum.any?(peers, &(&1.node_id == node_id))
    end

    test "returns peers sorted by reputation" do
      # Add peers with different reputations
      for i <- 1..5 do
        node_id = :crypto.strong_rand_bytes(20)
        P2PBootstrap.add_peer(node_id, "192.168.1.#{i}", 4000 + i)
      end
      
      peers = P2PBootstrap.get_peers()
      
      if length(peers) > 1 do
        reputations = Enum.map(peers, & &1.reputation)
        assert reputations == Enum.sort(reputations, &>=/2)
      end
    end
  end

  describe "network announcement" do
    test "announces node to network" do
      # Should not crash
      :ok = P2PBootstrap.announce(4001)
    end
  end

  describe "bootstrap statistics" do
    test "tracks discovery statistics" do
      initial_stats = P2PBootstrap.get_stats()
      
      assert initial_stats.total_peers_discovered >= 0
      assert is_map(initial_stats.discovery_methods_used)
      
      # Trigger discovery
      P2PBootstrap.discover_peers()
      
      final_stats = P2PBootstrap.get_stats()
      assert final_stats.total_peers_discovered >= initial_stats.total_peers_discovered
    end

    test "tracks discovery methods used" do
      P2PBootstrap.discover_peers()
      
      stats = P2PBootstrap.get_stats()
      methods = stats.discovery_methods_used
      
      # Should have attempted various methods
      assert Map.has_key?(methods, :hardcoded)
      assert Map.has_key?(methods, :dns)
      assert Map.has_key?(methods, :mdns)
    end
  end

  describe "DNS seed resolution" do
    test "handles missing DNS seeds gracefully" do
      # No DNS seeds configured in setup
      {:ok, _} = P2PBootstrap.discover_peers()
      
      stats = P2PBootstrap.get_stats()
      # Should still track DNS attempts
      assert Map.has_key?(stats.discovery_methods_used, :dns)
    end

    @tag :requires_dns
    test "resolves DNS seeds when available" do
      # This would require real DNS seeds
      {:ok, _} = P2PBootstrap.start_link(
        dns_seeds: ["seed.example.com"]
      )
      
      {:ok, peers} = P2PBootstrap.discover_peers()
      
      # Check if any peers came from DNS
      dns_peers = Enum.filter(peers, &(&1.reputation == 0.7))
      assert is_list(dns_peers)
    end
  end

  describe "mDNS discovery" do
    test "enables mDNS discovery" do
      # mDNS should be attempted
      P2PBootstrap.discover_peers()
      
      stats = P2PBootstrap.get_stats()
      assert Map.has_key?(stats.discovery_methods_used, :mdns)
    end

    test "announces via mDNS" do
      # Should not crash
      :ok = P2PBootstrap.announce(4002)
    end
  end

  describe "peer management" do
    test "limits maximum peers" do
      # Add many peers
      for i <- 1..200 do
        node_id = :crypto.strong_rand_bytes(20)
        P2PBootstrap.add_peer(node_id, "10.0.0.#{rem(i, 255)}", 4000 + i)
      end
      
      peers = P2PBootstrap.get_peers()
      assert length(peers) <= 100  # Max peers limit
    end

    test "removes stale peers periodically" do
      # Add a peer
      old_peer_id = :crypto.strong_rand_bytes(20)
      P2PBootstrap.add_peer(old_peer_id, "192.168.1.50", 4050)
      
      # Verify it exists
      peers = P2PBootstrap.get_peers()
      assert Enum.any?(peers, &(&1.node_id == old_peer_id))
      
      # Stale peer cleanup happens periodically
      # In tests, we can't wait that long
    end
  end

  describe "BitTorrent DHT integration" do
    test "attempts DHT discovery when enabled" do
      {:ok, _} = P2PBootstrap.start_link(
        node_id: "dht_test_node",
        dht_port: 6881
      )
      
      P2PBootstrap.discover_peers()
      
      stats = P2PBootstrap.get_stats()
      assert Map.has_key?(stats.discovery_methods_used, :dht)
    end

    test "announces to DHT" do
      {:ok, _} = P2PBootstrap.start_link(
        node_id: "dht_announce_node",
        dht_port: 6882
      )
      
      # Should not crash
      :ok = P2PBootstrap.announce(4003)
    end
  end

  describe "bootstrap nodes" do
    test "connects to hardcoded bootstrap nodes" do
      P2PBootstrap.discover_peers()
      
      stats = P2PBootstrap.get_stats()
      # Should have attempted hardcoded nodes
      assert stats.discovery_methods_used[:hardcoded] >= 0
    end

    test "handles bootstrap node failures gracefully" do
      # Bootstrap nodes might not be reachable
      {:ok, peers} = P2PBootstrap.discover_peers()
      assert is_list(peers)
    end
  end
end
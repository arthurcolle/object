defmodule Object.P2PBootstrap do
  @moduledoc """
  P2P bootstrap and discovery service for the Object network.
  
  Provides multiple discovery mechanisms to find and join the P2P network
  including hardcoded bootstrap nodes, DNS seeds, mDNS/Bonjour local discovery,
  BitTorrent DHT integration, and gossip-based peer exchange.
  
  ## Features
  
  - Multiple bootstrap strategies
  - DNS seed resolution
  - mDNS/Bonjour local network discovery
  - BitTorrent mainline DHT integration
  - Gossip-based peer exchange protocol
  - Bootstrap node reputation tracking
  - Network partition detection and healing
  """
  
  use GenServer
  require Logger
  
  @mdns_service "_object-p2p._tcp"
  # @mdns_domain "local"
  @dns_seed_prefix "seed"
  # @bootstrap_retry_interval 30_000
  @peer_exchange_interval 60_000
  @max_peers 100
  @min_peers 20
  
  @type peer_info :: %{
    node_id: binary(),
    address: String.t(),
    port: non_neg_integer(),
    last_seen: DateTime.t(),
    reputation: float(),
    capabilities: MapSet.t(),
    latency_ms: non_neg_integer() | nil
  }
  
  @type bootstrap_method :: :hardcoded | :dns | :mdns | :dht | :peer_exchange
  
  @type state :: %{
    node_id: binary(),
    peers: %{binary() => peer_info()},
    bootstrap_nodes: [peer_info()],
    dns_seeds: [String.t()],
    active_discoveries: MapSet.t(),
    peer_exchange_enabled: boolean(),
    mdns_ref: reference() | nil,
    dht_port: non_neg_integer() | nil,
    stats: bootstrap_stats()
  }
  
  @type bootstrap_stats :: %{
    total_peers_discovered: non_neg_integer(),
    successful_connections: non_neg_integer(),
    failed_connections: non_neg_integer(),
    discovery_methods_used: %{bootstrap_method() => non_neg_integer()}
  }
  
  # Client API
  
  @doc """
  Starts the P2P bootstrap service.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Initiates network discovery using all available methods.
  """
  @spec discover_peers() :: {:ok, [peer_info()]} | {:error, term()}
  def discover_peers do
    GenServer.call(__MODULE__, :discover_peers, 30_000)
  end
  
  @doc """
  Adds a known peer to the peer list.
  """
  @spec add_peer(binary(), String.t(), non_neg_integer()) :: :ok
  def add_peer(node_id, address, port) do
    GenServer.cast(__MODULE__, {:add_peer, node_id, address, port})
  end
  
  @doc """
  Gets the current list of known peers.
  """
  @spec get_peers() :: [peer_info()]
  def get_peers do
    GenServer.call(__MODULE__, :get_peers)
  end
  
  @doc """
  Announces this node to the network.
  """
  @spec announce(non_neg_integer()) :: :ok
  def announce(listen_port) do
    GenServer.cast(__MODULE__, {:announce, listen_port})
  end
  
  @doc """
  Gets bootstrap statistics.
  """
  @spec get_stats() :: bootstrap_stats()
  def get_stats do
    GenServer.call(__MODULE__, :get_stats)
  end
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    node_id = Keyword.get(opts, :node_id) || get_or_generate_node_id()
    
    state = %{
      node_id: node_id,
      peers: %{},
      bootstrap_nodes: default_bootstrap_nodes(),
      dns_seeds: Keyword.get(opts, :dns_seeds, default_dns_seeds()),
      active_discoveries: MapSet.new(),
      peer_exchange_enabled: Keyword.get(opts, :peer_exchange, true),
      mdns_ref: nil,
      dht_port: Keyword.get(opts, :dht_port),
      stats: %{
        total_peers_discovered: 0,
        successful_connections: 0,
        failed_connections: 0,
        discovery_methods_used: %{
          hardcoded: 0,
          dns: 0,
          mdns: 0,
          dht: 0,
          peer_exchange: 0
        }
      }
    }
    
    # Start discovery processes
    send(self(), :initial_bootstrap)
    
    # Schedule periodic tasks
    schedule_peer_exchange()
    schedule_peer_cleanup()
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:discover_peers, _from, state) do
    # Start all discovery methods
    new_state = state
    |> start_hardcoded_discovery()
    |> start_dns_discovery()
    |> start_mdns_discovery()
    |> start_dht_discovery()
    
    # Return current peers
    peers = Map.values(new_state.peers)
    {:reply, {:ok, peers}, new_state}
  end
  
  @impl true
  def handle_call(:get_peers, _from, state) do
    peers = state.peers
    |> Map.values()
    |> Enum.sort_by(& &1.reputation, &>=/2)
    |> Enum.take(@max_peers)
    
    {:reply, peers, state}
  end
  
  @impl true
  def handle_call(:get_stats, _from, state) do
    {:reply, state.stats, state}
  end
  
  @impl true
  def handle_cast({:add_peer, node_id, address, port}, state) do
    peer_info = %{
      node_id: node_id,
      address: address,
      port: port,
      last_seen: DateTime.utc_now(),
      reputation: 0.5,  # Neutral starting reputation
      capabilities: MapSet.new(),
      latency_ms: nil
    }
    
    new_state = add_peer_to_state(state, peer_info)
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:announce, listen_port}, state) do
    # Announce via mDNS
    if state.mdns_ref do
      announce_mdns(state.node_id, listen_port)
    end
    
    # Announce via DHT
    if state.dht_port do
      announce_dht(state.node_id, listen_port)
    end
    
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:initial_bootstrap, state) do
    # Try bootstrap nodes first
    new_state = connect_to_bootstrap_nodes(state)
    
    # If not enough peers, try other methods
    if map_size(new_state.peers) < @min_peers do
      newer_state = new_state
      |> start_dns_discovery()
      |> start_mdns_discovery()
      |> start_dht_discovery()
      
      {:noreply, newer_state}
    else
      {:noreply, new_state}
    end
  end
  
  @impl true
  def handle_info(:peer_exchange, state) do
    if state.peer_exchange_enabled and map_size(state.peers) > 0 do
      perform_peer_exchange(state)
    end
    
    schedule_peer_exchange()
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:cleanup_peers, state) do
    # Remove stale peers
    now = DateTime.utc_now()
    max_age_seconds = 3600  # 1 hour
    
    active_peers = state.peers
    |> Enum.reject(fn {_id, peer} ->
      DateTime.diff(now, peer.last_seen, :second) > max_age_seconds
    end)
    |> Map.new()
    
    schedule_peer_cleanup()
    {:noreply, %{state | peers: active_peers}}
  end
  
  @impl true
  def handle_info({:peer_discovered, method, peer_info}, state) do
    new_state = state
    |> add_peer_to_state(peer_info)
    |> update_stats(:peer_discovered, method)
    
    # Remove from active discoveries
    updated_state = update_in(new_state.active_discoveries, fn discoveries ->
      MapSet.delete(discoveries, {method, peer_info.node_id})
    end)
    
    {:noreply, updated_state}
  end
  
  @impl true
  def handle_info({:mdns_event, event}, state) do
    case event do
      {:service_up, service_info} ->
        handle_mdns_service_up(state, service_info)
        
      {:service_down, node_id} ->
        handle_mdns_service_down(state, node_id)
    end
  end
  
  @impl true
  def handle_info({:dht_peer, _info_hash, peers}, state) do
    # Handle peers discovered via BitTorrent DHT
    new_state = Enum.reduce(peers, state, fn {addr, port}, acc ->
      peer_info = %{
        node_id: generate_peer_id(addr, port),
        address: addr,
        port: port,
        last_seen: DateTime.utc_now(),
        reputation: 0.3,  # Lower initial reputation for DHT peers
        capabilities: MapSet.new(),
        latency_ms: nil
      }
      
      add_peer_to_state(acc, peer_info)
    end)
    
    {:noreply, new_state}
  end
  
  # Bootstrap Methods
  
  defp connect_to_bootstrap_nodes(state) do
    Enum.reduce(state.bootstrap_nodes, state, fn node, acc ->
      Task.start(fn ->
        case connect_to_peer(node) do
          :ok ->
            send(self(), {:peer_discovered, :hardcoded, node})
          {:error, _reason} ->
            :ok
        end
      end)
      
      acc
    end)
  end
  
  defp start_hardcoded_discovery(state) do
    # Already handled in initial bootstrap
    state
  end
  
  defp start_dns_discovery(state) do
    Task.start(fn ->
      discover_via_dns(state.dns_seeds)
    end)
    
    update_in(state.active_discoveries, &MapSet.put(&1, :dns))
  end
  
  defp start_mdns_discovery(state) do
    case start_mdns_browser() do
      {:ok, ref} ->
        %{state | mdns_ref: ref}
        
      {:error, reason} ->
        Logger.error("Failed to start mDNS: #{inspect(reason)}")
        state
    end
  end
  
  defp start_dht_discovery(state) do
    if state.dht_port do
      Task.start(fn ->
        discover_via_dht(state.node_id)
      end)
      
      update_in(state.active_discoveries, &MapSet.put(&1, :dht))
    else
      state
    end
  end
  
  # DNS Discovery
  
  defp discover_via_dns(dns_seeds) do
    dns_seeds
    |> Enum.flat_map(&resolve_dns_seed/1)
    |> Enum.each(fn {address, port} ->
      peer_info = %{
        node_id: generate_peer_id(address, port),
        address: address,
        port: port,
        last_seen: DateTime.utc_now(),
        reputation: 0.7,  # Higher reputation for DNS seeds
        capabilities: MapSet.new(),
        latency_ms: nil
      }
      
      send(self(), {:peer_discovered, :dns, peer_info})
    end)
  end
  
  defp resolve_dns_seed(seed) do
    # Try both A and SRV records
    case resolve_srv_records(seed) do
      [] -> resolve_a_records(seed)
      srv_results -> srv_results
    end
  end
  
  defp resolve_srv_records(domain) do
    srv_domain = "#{@dns_seed_prefix}.#{domain}"
    
    case :inet_res.lookup(to_charlist(srv_domain), :in, :srv) do
      [] -> 
        []
      
      records ->
        Enum.map(records, fn {_priority, _weight, port, host} ->
          case resolve_hostname(host) do
            {:ok, address} -> {address, port}
            _ -> nil
          end
        end)
        |> Enum.filter(&(&1 != nil))
    end
  end
  
  defp resolve_a_records(domain) do
    case resolve_hostname(domain) do
      {:ok, address} -> [{address, 4000}]  # Default port
      _ -> []
    end
  end
  
  defp resolve_hostname(hostname) when is_list(hostname) do
    case :inet.gethostbyname(hostname) do
      {:ok, {:hostent, _, _, _, _, [addr | _]}} ->
        {:ok, :inet.ntoa(addr) |> to_string()}
      _ ->
        {:error, :resolution_failed}
    end
  end
  defp resolve_hostname(hostname), do: resolve_hostname(to_charlist(hostname))
  
  # mDNS Discovery
  
  defp start_mdns_browser do
    # Simplified mDNS browser
    # In reality, would use a proper mDNS library
    {:ok, make_ref()}
  end
  
  defp announce_mdns(node_id, _port) do
    # Announce service via mDNS
    service_name = "object-#{Base.encode16(node_id, case: :lower)}"
    
    # Would use mDNS library to announce service
    Logger.info("Announcing mDNS service: #{service_name}.#{@mdns_service}")
  end
  
  defp handle_mdns_service_up(state, service_info) do
    peer_info = %{
      node_id: service_info.node_id,
      address: service_info.address,
      port: service_info.port,
      last_seen: DateTime.utc_now(),
      reputation: 0.8,  # High reputation for local network peers
      capabilities: MapSet.new(service_info.txt_records),
      latency_ms: nil
    }
    
    new_state = add_peer_to_state(state, peer_info)
    |> update_stats(:peer_discovered, :mdns)
    
    {:noreply, new_state}
  end
  
  defp handle_mdns_service_down(state, node_id) do
    new_peers = Map.delete(state.peers, node_id)
    {:noreply, %{state | peers: new_peers}}
  end
  
  # BitTorrent DHT Discovery
  
  defp discover_via_dht(node_id) do
    # Generate info hash from node ID
    info_hash = :crypto.hash(:sha, node_id)
    
    # Would use BitTorrent DHT library
    # to find peers announcing this info hash
    Logger.info("Searching DHT for info hash: #{Base.encode16(info_hash)}")
    
    # Simulate DHT response
    Process.send_after(self(), {:dht_peer, info_hash, []}, 5000)
  end
  
  defp announce_dht(node_id, _port) do
    info_hash = :crypto.hash(:sha, node_id)
    
    # Would announce to BitTorrent DHT
    Logger.info("Announcing to DHT with info hash: #{Base.encode16(info_hash)}")
  end
  
  # Peer Exchange
  
  defp perform_peer_exchange(state) do
    # Select random peers for exchange
    exchange_peers = state.peers
    |> Map.values()
    |> Enum.take_random(5)
    
    Enum.each(exchange_peers, fn peer ->
      request_peer_list(peer)
    end)
  end
  
  defp request_peer_list(peer) do
    Task.start(fn ->
      # Send peer exchange request
      case send_peer_exchange_request(peer) do
        {:ok, peer_list} ->
          Enum.each(peer_list, fn new_peer ->
            send(self(), {:peer_discovered, :peer_exchange, new_peer})
          end)
          
        {:error, _reason} ->
          :ok
      end
    end)
  end
  
  defp send_peer_exchange_request(_peer) do
    # Would implement actual peer exchange protocol
    {:ok, []}
  end
  
  # Connection Management
  
  defp connect_to_peer(peer_info) do
    # Attempt to establish connection
    case Object.NetworkTransport.connect(peer_info.address, peer_info.port) do
      {:ok, _conn_id} ->
        # Perform handshake
        :ok
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp add_peer_to_state(state, peer_info) do
    if Map.has_key?(state.peers, peer_info.node_id) do
      # Update existing peer
      update_in(state.peers[peer_info.node_id], fn existing ->
        %{existing | 
          last_seen: DateTime.utc_now(),
          reputation: update_reputation(existing.reputation, :seen)
        }
      end)
    else
      # Add new peer
      put_in(state.peers[peer_info.node_id], peer_info)
    end
  end
  
  defp update_reputation(current, :seen) do
    # Slowly increase reputation when peer is active
    min(1.0, current + 0.01)
  end
  
  defp update_reputation(current, :success) do
    # Increase reputation on successful interaction
    min(1.0, current + 0.05)
  end
  
  defp update_reputation(current, :failure) do
    # Decrease reputation on failure
    max(0.0, current - 0.1)
  end
  
  # Statistics
  
  defp update_stats(state, :peer_discovered, method) do
    update_in(state.stats, fn stats ->
      %{stats |
        total_peers_discovered: stats.total_peers_discovered + 1,
        discovery_methods_used: Map.update(
          stats.discovery_methods_used,
          method,
          1,
          &(&1 + 1)
        )
      }
    end)
  end
  
  # Utilities
  
  defp default_bootstrap_nodes do
    [
      # These would be well-known bootstrap nodes
      %{
        node_id: Base.decode16!("1234567890ABCDEF1234567890ABCDEF12345678"),
        address: "bootstrap1.object-network.org",
        port: 4000,
        last_seen: DateTime.utc_now(),
        reputation: 1.0,
        capabilities: MapSet.new(["bootstrap", "relay"]),
        latency_ms: nil
      },
      %{
        node_id: Base.decode16!("ABCDEF1234567890ABCDEF1234567890ABCDEF12"),
        address: "bootstrap2.object-network.org",
        port: 4000,
        last_seen: DateTime.utc_now(),
        reputation: 1.0,
        capabilities: MapSet.new(["bootstrap", "relay"]),
        latency_ms: nil
      }
    ]
  end
  
  defp default_dns_seeds do
    [
      "seed.object-network.org",
      "seed2.object-network.org",
      "dnsseed.object-network.io"
    ]
  end
  
  defp generate_peer_id(address, port) do
    :crypto.hash(:sha, "#{address}:#{port}")
  end
  
  defp schedule_peer_exchange do
    Process.send_after(self(), :peer_exchange, @peer_exchange_interval)
  end
  
  defp schedule_peer_cleanup do
    Process.send_after(self(), :cleanup_peers, 300_000)  # 5 minutes
  end
  
  defp get_or_generate_node_id do
    case GenServer.whereis(Object.DistributedRegistry) do
      nil ->
        # Generate a temporary node ID if DistributedRegistry isn't available
        :crypto.strong_rand_bytes(20)
      
      _pid ->
        try do
          Object.DistributedRegistry.get_node_id()
        rescue
          _ ->
            :crypto.strong_rand_bytes(20)
        end
    end
  end
end
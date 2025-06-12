defmodule Object.NetworkSupervisor do
  @moduledoc """
  Supervisor for all network-related components of the Object system.
  
  Manages the lifecycle of network services including transport, protocol,
  DHT, encryption, NAT traversal, bootstrap, and Byzantine fault tolerance.
  Provides a unified interface for starting and configuring the P2P network.
  """
  
  use Supervisor
  require Logger
  
  @type network_config :: %{
    node_id: binary() | nil,
    listen_port: non_neg_integer(),
    transport: %{
      pool_size: non_neg_integer(),
      timeout: timeout()
    },
    dht: %{
      enabled: boolean(),
      k_bucket_size: non_neg_integer(),
      bootstrap_nodes: [{String.t(), non_neg_integer()}]
    },
    encryption: %{
      enabled: boolean(),
      identity_file: String.t() | nil,
      require_trusted_certs: boolean()
    },
    nat: %{
      stun_servers: [{String.t(), non_neg_integer()}],
      turn_servers: [{String.t(), non_neg_integer(), map()}],
      enable_upnp: boolean()
    },
    bootstrap: %{
      dns_seeds: [String.t()],
      enable_mdns: boolean(),
      enable_dht: boolean()
    },
    byzantine: %{
      enabled: boolean(),
      require_pow: boolean(),
      min_reputation: float()
    }
  }
  
  def start_link(config \\ %{}) do
    Supervisor.start_link(__MODULE__, config, name: __MODULE__)
  end
  
  @impl true
  def init(config) do
    # Generate node ID if not provided
    node_id = Map.get(config, :node_id) || generate_node_id()
    listen_port = Map.get(config, :listen_port, 4000)
    
    children = [
      # Core transport layer
      {Object.NetworkTransport, 
        transport_config(config[:transport] || %{})},
      
      # Encryption service (must start before others)
      {Object.Encryption,
        encryption_config(config[:encryption] || %{}, node_id)},
      
      # Distributed hash table (must start before P2PBootstrap)
      if Map.get(config[:dht] || %{}, :enabled, true) do
        {Object.DistributedRegistry,
          dht_config(config[:dht] || %{}, node_id, listen_port)}
      else
        nil
      end,
      
      # NAT traversal
      {Object.NATTraversal,
        nat_config(config[:nat] || %{})},
      
      # P2P bootstrap and discovery (depends on DistributedRegistry)
      {Object.P2PBootstrap,
        bootstrap_config(config[:bootstrap] || %{}, node_id, listen_port)},
      
      # Byzantine fault tolerance
      if Map.get(config[:byzantine] || %{}, :enabled, true) do
        {Object.ByzantineFaultTolerance,
          byzantine_config(config[:byzantine] || %{}, node_id)}
      else
        nil
      end,
      
      # Network coordinator process
      {Object.NetworkCoordinator, %{
        node_id: node_id,
        listen_port: listen_port,
        config: config
      }}
    ]
    |> Enum.filter(&(&1 != nil))
    
    Supervisor.init(children, strategy: :one_for_one)
  end
  
  # Configuration builders
  
  defp transport_config(config) do
    [
      pool_size: Map.get(config, :pool_size, 10),
      timeout: Map.get(config, :timeout, 5_000),
      reconnect_interval: Map.get(config, :reconnect_interval, 1_000),
      max_reconnect_attempts: Map.get(config, :max_reconnect_attempts, 5)
    ]
  end
  
  defp encryption_config(config, node_id) do
    [
      node_id: node_id,
      identity_file: Map.get(config, :identity_file),
      session_timeout: Map.get(config, :session_timeout, 3600_000),
      max_sessions: Map.get(config, :max_sessions, 1000),
      require_trusted_certs: Map.get(config, :require_trusted_certs, false)
    ]
  end
  
  defp nat_config(config) do
    [
      stun_servers: Map.get(config, :stun_servers, default_stun_servers()),
      turn_servers: Map.get(config, :turn_servers, []),
      enable_upnp: Map.get(config, :enable_upnp, true),
      enable_turn: Map.get(config, :enable_turn, true)
    ]
  end
  
  defp bootstrap_config(config, node_id, listen_port) do
    [
      node_id: node_id,
      dns_seeds: Map.get(config, :dns_seeds, []),
      peer_exchange: Map.get(config, :peer_exchange, true),
      dht_port: (if Map.get(config, :enable_dht, true), do: listen_port + 1)
    ]
  end
  
  defp dht_config(config, node_id, listen_port) do
    [
      node_id: node_id,
      port: listen_port,
      bootstrap_nodes: Map.get(config, :bootstrap_nodes, []),
      storage_limit: Map.get(config, :storage_limit, 1000)
    ]
  end
  
  defp byzantine_config(config, node_id) do
    [
      node_id: node_id,
      require_pow: Map.get(config, :require_pow, true),
      min_reputation: Map.get(config, :min_reputation, 0.3),
      consensus_timeout: Map.get(config, :consensus_timeout, 30_000)
    ]
  end
  
  defp default_stun_servers do
    [
      {"stun.l.google.com", 19302},
      {"stun1.l.google.com", 19302}
    ]
  end
  
  defp generate_node_id do
    :crypto.strong_rand_bytes(20)
  end
end

defmodule Object.NetworkCoordinator do
  @moduledoc """
  Coordinates network operations across all network components.
  
  Handles high-level network operations like establishing secure connections,
  managing object replication, and coordinating distributed operations.
  """
  
  use GenServer
  require Logger
  
  # Client API
  
  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Connects to the P2P network.
  """
  def connect_network do
    GenServer.call(__MODULE__, :connect_network, 30_000)
  end
  
  @doc """
  Publishes a local object to the network.
  """
  def publish_object(object) do
    GenServer.call(__MODULE__, {:publish_object, object})
  end
  
  @doc """
  Creates a proxy for a remote object.
  """
  def get_remote_object(object_id) do
    GenServer.call(__MODULE__, {:get_remote_object, object_id})
  end
  
  @doc """
  Gets network statistics.
  """
  def get_network_stats do
    GenServer.call(__MODULE__, :get_network_stats)
  end
  
  # Server implementation
  
  @impl true
  def init(opts) do
    state = %{
      node_id: opts.node_id,
      listen_port: opts.listen_port,
      config: opts.config,
      published_objects: %{},
      remote_proxies: %{},
      network_state: :disconnected,
      peer_count: 0,
      start_time: DateTime.utc_now()
    }
    
    # Auto-connect if configured
    if Map.get(opts.config, :auto_connect, true) do
      send(self(), :auto_connect)
    end
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:connect_network, _from, state) do
    case state.network_state do
      :connected ->
        {:reply, {:ok, :already_connected}, state}
        
      _ ->
        # Discover NAT type
        nat_result = Object.NATTraversal.discover_nat()
        
        # Bootstrap into network
        Object.P2PBootstrap.announce(state.listen_port)
        {:ok, peers} = Object.P2PBootstrap.discover_peers()
        
        # Update state
        new_state = %{state |
          network_state: :connected,
          peer_count: length(peers)
        }
        
        Logger.info("Connected to P2P network with #{length(peers)} peers")
        Logger.info("NAT type: #{inspect(nat_result)}")
        
        {:reply, {:ok, :connected}, new_state}
    end
  end
  
  @impl true
  def handle_call({:publish_object, object}, _from, state) do
    # Register in DHT
    :ok = Object.DistributedRegistry.register_object(object.id, %{
      node_id: state.node_id,
      address: get_public_address(),
      port: state.listen_port,
      object_type: object.type,
      metadata: object.metadata
    })
    
    # Track published object
    new_state = put_in(state.published_objects[object.id], object)
    
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_call({:get_remote_object, object_id}, _from, state) do
    # Check if we already have a proxy
    case Map.get(state.remote_proxies, object_id) do
      nil ->
        # Create new proxy
        case Object.NetworkProxy.create(object_id) do
          {:ok, proxy} ->
            new_state = put_in(state.remote_proxies[object_id], proxy)
            {:reply, {:ok, proxy}, new_state}
            
          error ->
            {:reply, error, state}
        end
        
      proxy ->
        {:reply, {:ok, proxy}, state}
    end
  end
  
  @impl true
  def handle_call(:get_network_stats, _from, state) do
    uptime = DateTime.diff(DateTime.utc_now(), state.start_time, :second)
    
    stats = %{
      node_id: Base.encode16(state.node_id),
      network_state: state.network_state,
      peer_count: state.peer_count,
      published_objects: map_size(state.published_objects),
      remote_proxies: map_size(state.remote_proxies),
      uptime_seconds: uptime,
      transport_metrics: Object.NetworkTransport.get_metrics(),
      bootstrap_stats: Object.P2PBootstrap.get_stats()
    }
    
    {:reply, stats, state}
  end
  
  @impl true
  def handle_info(:auto_connect, state) do
    Task.start(fn ->
      Process.sleep(1000)  # Give services time to start
      connect_network()
    end)
    
    {:noreply, state}
  end
  
  defp get_public_address do
    # Would determine actual public address
    "0.0.0.0"
  end
end
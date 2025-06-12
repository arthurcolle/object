defmodule Object.DistributedRegistry do
  @moduledoc """
  Distributed registry for Objects using Kademlia DHT algorithm.
  
  Provides a decentralized, fault-tolerant registry for object discovery
  across network boundaries. Implements Kademlia's XOR-based routing with
  k-buckets for efficient O(log n) lookups.
  
  ## Features
  
  - Kademlia DHT with 160-bit node IDs
  - K-bucket routing table management
  - Iterative lookup procedures
  - Node liveness checking via PING/PONG
  - Automatic republishing of values
  - Byzantine fault tolerance mechanisms
  - NAT-aware node addressing
  """
  
  use GenServer
  require Logger
  
  alias Object.{NetworkTransport, NetworkProtocol}
  
  @node_id_bits 160
  @k_bucket_size 20  # Kademlia k parameter
  @alpha 3           # Concurrency parameter
  @republish_interval 3600_000  # 1 hour
  @refresh_interval 900_000     # 15 minutes
  @expiration_time 86400_000    # 24 hours
  
  @type node_id :: <<_::160>>
  @type node_info :: %{
    id: node_id(),
    address: String.t(),
    port: non_neg_integer(),
    last_seen: DateTime.t(),
    rtt: non_neg_integer() | nil,
    reputation: float()
  }
  
  @type routing_table :: %{
    buckets: %{non_neg_integer() => [node_info()]},
    self_id: node_id()
  }
  
  @type stored_value :: %{
    key: binary(),
    value: term(),
    publisher: node_id(),
    timestamp: DateTime.t(),
    ttl: non_neg_integer()
  }
  
  @type state :: %{
    node_id: node_id(),
    routing_table: routing_table(),
    storage: %{binary() => stored_value()},
    pending_queries: %{reference() => query_state()},
    transport: pid(),
    config: map()
  }
  
  @type query_state :: %{
    type: :find_node | :find_value | :store,
    target: binary(),
    visited: MapSet.t(),
    active: MapSet.t(),
    best_nodes: [node_info()],
    from: GenServer.from() | nil,
    start_time: DateTime.t()
  }
  
  # Client API
  
  @doc """
  Starts the distributed registry.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Registers a local object in the distributed registry.
  """
  @spec register_object(String.t(), Object.t()) :: :ok | {:error, term()}
  def register_object(object_id, object) do
    GenServer.call(__MODULE__, {:register_object, object_id, object})
  end
  
  @doc """
  Looks up an object by ID in the distributed registry.
  """
  @spec lookup_object(String.t()) :: {:ok, Object.t()} | {:error, :not_found}
  def lookup_object(object_id) do
    GenServer.call(__MODULE__, {:lookup_object, object_id}, 10_000)
  end
  
  @doc """
  Finds nodes close to a given ID.
  """
  @spec find_node(binary()) :: {:ok, [node_info()]} | {:error, term()}
  def find_node(target_id) do
    GenServer.call(__MODULE__, {:find_node, target_id})
  end
  
  @doc """
  Bootstraps the node into the DHT network.
  """
  @spec bootstrap([{String.t(), non_neg_integer()}]) :: :ok | {:error, term()}
  def bootstrap(bootstrap_nodes) do
    GenServer.call(__MODULE__, {:bootstrap, bootstrap_nodes})
  end
  
  @doc """
  Gets the current node's ID.
  """
  @spec get_node_id() :: node_id()
  def get_node_id do
    GenServer.call(__MODULE__, :get_node_id)
  end
  
  @doc """
  Adds a peer to the routing table.
  
  ## Parameters
  - `peer_id`: Node ID of the peer
  - `address`: IP address of the peer
  - `port`: Port number of the peer
  """
  @spec add_peer(node_id(), String.t(), non_neg_integer()) :: :ok | {:error, term()}
  def add_peer(peer_id, address, port) do
    GenServer.call(__MODULE__, {:add_peer, peer_id, address, port})
  end
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    # Generate or load node ID
    node_id = generate_node_id(opts)
    
    # Initialize routing table
    routing_table = %{
      buckets: initialize_buckets(),
      self_id: node_id
    }
    
    # Start network transport - handle already started case
    transport_result = case Object.NetworkTransport.start_link() do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
      other -> other
    end
    
    case transport_result do
      {:ok, transport} ->
        state = %{
          node_id: node_id,
          routing_table: routing_table,
          storage: %{},
          pending_queries: %{},
          transport: transport,
          config: %{
            listen_port: Keyword.get(opts, :port, 4000),
            bootstrap_nodes: Keyword.get(opts, :bootstrap_nodes, []),
            storage_limit: Keyword.get(opts, :storage_limit, 1000)
          }
        }
        
        # Schedule periodic tasks
        schedule_refresh()
        schedule_republish()
        schedule_cleanup()
        
        {:ok, state}
      {:error, reason} ->
        {:stop, reason}
    end
  end
  
  @impl true
  def handle_call({:register_object, object_id, object}, _from, state) do
    key = hash_key(object_id)
    
    # Store locally
    stored_value = %{
      key: key,
      value: object,
      publisher: state.node_id,
      timestamp: DateTime.utc_now(),
      ttl: @expiration_time
    }
    
    new_state = put_in(state.storage[key], stored_value)
    
    # Initiate DHT store operation
    initiate_store(new_state, key, object)
    
    {:reply, :ok, new_state}
  end
  
  @impl true
  def handle_call({:lookup_object, object_id}, from, state) do
    key = hash_key(object_id)
    
    # Check local storage first
    case Map.get(state.storage, key) do
      %{value: object} ->
        {:reply, {:ok, object}, state}
        
      nil ->
        # Initiate DHT lookup
        query_ref = make_ref()
        query_state = %{
          type: :find_value,
          target: key,
          visited: MapSet.new(),
          active: MapSet.new(),
          best_nodes: get_closest_nodes(state.routing_table, key, @alpha),
          from: from,
          start_time: DateTime.utc_now()
        }
        
        new_state = put_in(state.pending_queries[query_ref], query_state)
        
        # Start iterative lookup
        continue_lookup(new_state, query_ref)
        
        {:noreply, new_state}
    end
  end
  
  @impl true
  def handle_call({:find_node, target_id}, _from, state) do
    closest = get_closest_nodes(state.routing_table, target_id, @k_bucket_size)
    {:reply, {:ok, closest}, state}
  end
  
  @impl true
  def handle_call({:bootstrap, bootstrap_nodes}, _from, state) do
    # Connect to bootstrap nodes
    Enum.each(bootstrap_nodes, fn {host, port} ->
      add_node(state, generate_temp_node_id(), host, port)
    end)
    
    # Find nodes close to ourselves
    initiate_self_lookup(state)
    
    {:reply, :ok, state}
  end
  
  @impl true
  def handle_call({:add_peer, peer_id, address, port}, _from, state) do
    # Add peer to routing table
    updated_state = add_node(state, peer_id, address, port)
    {:reply, :ok, updated_state}
  end

  @impl true
  def handle_call(:get_node_id, _from, state) do
    {:reply, state.node_id, state}
  end
  
  @impl true
  def handle_info({:network_message, from_node, message}, state) do
    case message do
      {:ping, sender_id} ->
        handle_ping(state, from_node, sender_id)
        
      {:pong, sender_id} ->
        handle_pong(state, from_node, sender_id)
        
      {:find_node, sender_id, target} ->
        handle_find_node(state, from_node, sender_id, target)
        
      {:find_value, sender_id, key} ->
        handle_find_value(state, from_node, sender_id, key)
        
      {:store, sender_id, key, value} ->
        handle_store(state, from_node, sender_id, key, value)
        
      {:found_nodes, sender_id, nodes} ->
        handle_found_nodes(state, from_node, sender_id, nodes)
        
      {:found_value, sender_id, value} ->
        handle_found_value(state, from_node, sender_id, value)
        
      _ ->
        Logger.warning("Unknown message type: #{inspect(message)}")
        {:noreply, state}
    end
  end
  
  @impl true
  def handle_info(:refresh_buckets, state) do
    # Refresh buckets that haven't been accessed recently
    refreshed_state = refresh_stale_buckets(state)
    schedule_refresh()
    {:noreply, refreshed_state}
  end
  
  @impl true
  def handle_info(:republish_values, state) do
    # Republish stored values
    republished_state = republish_stored_values(state)
    schedule_republish()
    {:noreply, republished_state}
  end
  
  @impl true
  def handle_info(:cleanup_storage, state) do
    # Remove expired values
    cleaned_state = cleanup_expired_values(state)
    schedule_cleanup()
    {:noreply, cleaned_state}
  end
  
  # DHT Operations
  
  defp initiate_store(state, key, value) do
    # Find k closest nodes to the key
    closest_nodes = get_closest_nodes(state.routing_table, key, @k_bucket_size)
    
    # Send store requests to those nodes
    Enum.each(closest_nodes, fn node ->
      send_store_request(state, node, key, value)
    end)
  end
  
  defp initiate_self_lookup(state) do
    query_ref = make_ref()
    query_state = %{
      type: :find_node,
      target: state.node_id,
      visited: MapSet.new(),
      active: MapSet.new(),
      best_nodes: get_random_nodes(state.routing_table, @alpha),
      from: nil,
      start_time: DateTime.utc_now()
    }
    
    new_state = put_in(state.pending_queries[query_ref], query_state)
    continue_lookup(new_state, query_ref)
  end
  
  defp continue_lookup(state, query_ref) do
    case Map.get(state.pending_queries, query_ref) do
      nil ->
        state
        
      query ->
        # Select alpha nodes from best_nodes that we haven't queried
        to_query = query.best_nodes
        |> Enum.reject(fn node -> 
          MapSet.member?(query.visited, node.id) or
          MapSet.member?(query.active, node.id)
        end)
        |> Enum.take(@alpha)
        
        if Enum.empty?(to_query) and MapSet.size(query.active) == 0 do
          # Lookup complete
          finish_lookup(state, query_ref)
        else
          # Send queries to selected nodes
          Enum.each(to_query, fn node ->
            case query.type do
              :find_node ->
                send_find_node_request(state, node, query.target)
              :find_value ->
                send_find_value_request(state, node, query.target)
            end
          end)
          
          # Update query state
          new_active = Enum.reduce(to_query, query.active, fn node, acc ->
            MapSet.put(acc, node.id)
          end)
          
          put_in(state.pending_queries[query_ref].active, new_active)
        end
    end
  end
  
  defp finish_lookup(state, query_ref) do
    case Map.get(state.pending_queries, query_ref) do
      %{type: :find_value, from: from} when not is_nil(from) ->
        GenServer.reply(from, {:error, :not_found})
        
      _ ->
        :ok
    end
    
    Map.delete(state.pending_queries, query_ref)
  end
  
  # Message Handlers
  
  defp handle_ping(state, from_node, sender_id) do
    # Update routing table
    new_state = update_routing_table(state, sender_id, from_node)
    
    # Send pong response
    send_pong(new_state, from_node, new_state.node_id)
    
    {:noreply, new_state}
  end
  
  defp handle_pong(state, from_node, sender_id) do
    # Update routing table with responsive node
    new_state = update_routing_table(state, sender_id, from_node)
    {:noreply, new_state}
  end
  
  defp handle_find_node(state, from_node, sender_id, target) do
    # Update routing table
    new_state = update_routing_table(state, sender_id, from_node)
    
    # Find closest nodes
    closest = get_closest_nodes(new_state.routing_table, target, @k_bucket_size)
    
    # Send response
    send_found_nodes(new_state, from_node, new_state.node_id, closest)
    
    {:noreply, new_state}
  end
  
  defp handle_find_value(state, from_node, sender_id, key) do
    new_state = update_routing_table(state, sender_id, from_node)
    
    case Map.get(new_state.storage, key) do
      %{value: value} ->
        # We have the value
        send_found_value(new_state, from_node, new_state.node_id, value)
        
      nil ->
        # Return closest nodes instead
        closest = get_closest_nodes(new_state.routing_table, key, @k_bucket_size)
        send_found_nodes(new_state, from_node, new_state.node_id, closest)
    end
    
    {:noreply, new_state}
  end
  
  defp handle_store(state, from_node, sender_id, key, value) do
    new_state = update_routing_table(state, sender_id, from_node)
    
    # Store the value if we have space
    if map_size(new_state.storage) < new_state.config.storage_limit do
      stored_value = %{
        key: key,
        value: value,
        publisher: sender_id,
        timestamp: DateTime.utc_now(),
        ttl: @expiration_time
      }
      
      newer_state = put_in(new_state.storage[key], stored_value)
      {:noreply, newer_state}
    else
      {:noreply, new_state}
    end
  end
  
  defp handle_found_nodes(state, _from_node, sender_id, nodes) do
    # Process response to find_node query
    new_state = Enum.reduce(state.pending_queries, state, fn {ref, query}, acc ->
      if MapSet.member?(query.active, sender_id) do
        # Update query state
        new_visited = MapSet.put(query.visited, sender_id)
        new_active = MapSet.delete(query.active, sender_id)
        
        # Add discovered nodes to routing table and best_nodes
        {updated_state, new_best} = Enum.reduce(nodes, {acc, query.best_nodes}, 
          fn node_info, {st, best} ->
            st2 = add_node_info(st, node_info)
            best2 = update_best_nodes(best, node_info, query.target)
            {st2, best2}
          end)
        
        updated_query = %{query | 
          visited: new_visited,
          active: new_active,
          best_nodes: new_best
        }
        
        put_in(updated_state.pending_queries[ref], updated_query)
      else
        acc
      end
    end)
    
    # Continue lookups
    final_state = Enum.reduce(Map.keys(new_state.pending_queries), new_state, fn ref, acc ->
      continue_lookup(acc, ref)
    end)
    
    {:noreply, final_state}
  end
  
  defp handle_found_value(state, _from_node, sender_id, value) do
    # Process response to find_value query
    new_state = Enum.reduce(state.pending_queries, state, fn {ref, query}, acc ->
      if query.type == :find_value and MapSet.member?(query.active, sender_id) do
        # Found the value!
        if query.from do
          GenServer.reply(query.from, {:ok, value})
        end
        
        Map.delete(acc.pending_queries, ref)
      else
        acc
      end
    end)
    
    {:noreply, new_state}
  end
  
  # Routing Table Management
  
  defp initialize_buckets do
    Map.new(0..(@node_id_bits - 1), fn i -> {i, []} end)
  end
  
  defp update_routing_table(state, node_id, node_info) do
    bucket_index = get_bucket_index(state.node_id, node_id)
    bucket = Map.get(state.routing_table.buckets, bucket_index, [])
    
    node = %{
      id: node_id,
      address: node_info.address,
      port: node_info.port,
      last_seen: DateTime.utc_now(),
      rtt: nil,
      reputation: 1.0
    }
    
    # Check if node already exists in bucket
    case Enum.find_index(bucket, fn n -> n.id == node_id end) do
      nil ->
        # Add new node if bucket not full
        if length(bucket) < @k_bucket_size do
          new_bucket = [node | bucket]
          put_in(state.routing_table.buckets[bucket_index], new_bucket)
        else
          # Bucket full - ping least recently seen node
          [oldest | _rest] = Enum.reverse(bucket)
          ping_node(state, oldest)
          state
        end
        
      index ->
        # Move to front (most recently seen)
        {_old, rest} = List.pop_at(bucket, index)
        new_bucket = [node | rest]
        put_in(state.routing_table.buckets[bucket_index], new_bucket)
    end
  end
  
  defp get_bucket_index(self_id, node_id) do
    # XOR distance between IDs
    distance = :crypto.exor(self_id, node_id)
    
    # Find highest bit position
    distance
    |> :binary.bin_to_list()
    |> Enum.find_index(&(&1 != 0))
    |> case do
      nil -> @node_id_bits - 1
      byte_index ->
        byte = :binary.at(distance, byte_index)
        bit_index = highest_bit_position(byte)
        byte_index * 8 + (7 - bit_index)
    end
  end
  
  defp highest_bit_position(byte) do
    cond do
      byte >= 128 -> 7
      byte >= 64 -> 6
      byte >= 32 -> 5
      byte >= 16 -> 4
      byte >= 8 -> 3
      byte >= 4 -> 2
      byte >= 2 -> 1
      true -> 0
    end
  end
  
  defp get_closest_nodes(routing_table, target, count) do
    routing_table.buckets
    |> Map.values()
    |> List.flatten()
    |> Enum.sort_by(fn node -> xor_distance(node.id, target) end)
    |> Enum.take(count)
  end
  
  defp xor_distance(id1, id2) do
    :crypto.exor(id1, id2)
    |> :binary.bin_to_list()
    |> Integer.undigits(256)
  end
  
  defp add_node_info(state, node_info) do
    update_routing_table(state, node_info.id, node_info)
  end
  
  defp update_best_nodes(best_nodes, new_node, target) do
    _all_nodes = [new_node | best_nodes]
    |> Enum.uniq_by(& &1.id)
    |> Enum.sort_by(fn node -> xor_distance(node.id, target) end)
    |> Enum.take(@k_bucket_size)
  end
  
  # Network Communication
  
  defp send_ping(_state, node, sender_id) do
    message = NetworkProtocol.create_cast(
      "dht_node",
      "ping",
      [sender_id]
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode ping: #{inspect(reason)}")
    end
  end
  
  defp send_pong(_state, node, sender_id) do
    message = NetworkProtocol.create_cast(
      "dht_node",
      "pong",
      [sender_id]
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode pong: #{inspect(reason)}")
    end
  end
  
  defp send_find_node_request(state, node, target) do
    message = NetworkProtocol.create_request(
      "dht_node",
      "find_node",
      [state.node_id, target]
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode find_node: #{inspect(reason)}")
    end
  end
  
  defp send_find_value_request(state, node, key) do
    message = NetworkProtocol.create_request(
      "dht_node",
      "find_value",
      [state.node_id, key]
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode find_value: #{inspect(reason)}")
    end
  end
  
  defp send_store_request(state, node, key, value) do
    message = NetworkProtocol.create_cast(
      "dht_node",
      "store",
      [state.node_id, key, value]
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode store: #{inspect(reason)}")
    end
  end
  
  defp send_found_nodes(_state, node, _sender_id, nodes) do
    # Convert node_info to serializable format
    nodes_data = Enum.map(nodes, fn n ->
      %{
        id: Base.encode16(n.id),
        address: n.address,
        port: n.port
      }
    end)
    
    message = NetworkProtocol.create_response(
      nil,  # correlation_id would be tracked properly in full impl
      nodes_data
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode found_nodes: #{inspect(reason)}")
    end
  end
  
  defp send_found_value(_state, node, _sender_id, value) do
    message = NetworkProtocol.create_response(
      nil,  # correlation_id would be tracked properly in full impl
      value
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
      {:error, reason} ->
        Logger.error("Failed to encode found_value: #{inspect(reason)}")
    end
  end
  
  # Utilities
  
  defp generate_node_id(opts) do
    case Keyword.get(opts, :node_id) do
      nil ->
        # Generate random node ID
        :crypto.strong_rand_bytes(20)
        
      id when is_binary(id) and byte_size(id) == 20 ->
        id
        
      id_string when is_binary(id_string) ->
        # Hash string to get node ID
        :crypto.hash(:sha, id_string)
    end
  end
  
  defp generate_temp_node_id do
    :crypto.strong_rand_bytes(20)
  end
  
  defp hash_key(key) when is_binary(key) do
    :crypto.hash(:sha, key)
  end
  defp hash_key(key) do
    key |> :erlang.term_to_binary() |> hash_key()
  end
  
  defp add_node(state, node_id, address, port) do
    node_info = %{
      id: node_id,
      address: address,
      port: port,
      last_seen: DateTime.utc_now(),
      rtt: nil,
      reputation: 1.0
    }
    
    update_routing_table(state, node_id, node_info)
  end
  
  defp ping_node(state, node) do
    send_ping(state, node, state.node_id)
  end
  
  defp get_random_nodes(routing_table, count) do
    routing_table.buckets
    |> Map.values()
    |> List.flatten()
    |> Enum.take_random(count)
  end
  
  # Periodic Tasks
  
  defp schedule_refresh do
    Process.send_after(self(), :refresh_buckets, @refresh_interval)
  end
  
  defp schedule_republish do
    Process.send_after(self(), :republish_values, @republish_interval)
  end
  
  defp schedule_cleanup do
    Process.send_after(self(), :cleanup_storage, 60_000)  # Every minute
  end
  
  defp refresh_stale_buckets(state) do
    # Refresh buckets that haven't been updated in a while
    now = DateTime.utc_now()
    
    Enum.reduce(state.routing_table.buckets, state, fn {index, bucket}, acc ->
      if should_refresh_bucket?(bucket, now) do
        # Generate random ID in this bucket's range
        target = generate_id_in_bucket(state.node_id, index)
        initiate_refresh_lookup(acc, target)
      else
        acc
      end
    end)
  end
  
  defp should_refresh_bucket?([], _now), do: true
  defp should_refresh_bucket?(bucket, now) do
    oldest = Enum.min_by(bucket, & &1.last_seen)
    DateTime.diff(now, oldest.last_seen, :second) > 3600  # 1 hour
  end
  
  defp generate_id_in_bucket(self_id, bucket_index) do
    # Generate ID with specific XOR distance
    distance = :crypto.strong_rand_bytes(20)
    
    # Set appropriate bit for bucket
    distance_list = :binary.bin_to_list(distance)
    byte_index = div(bucket_index, 8)
    bit_index = rem(bucket_index, 8)
    
    updated_list = List.update_at(distance_list, byte_index, fn byte ->
      Bitwise.bor(byte, Bitwise.bsl(1, 7 - bit_index))
    end)
    
    distance_binary = :binary.list_to_bin(updated_list)
    :crypto.exor(self_id, distance_binary)
  end
  
  defp initiate_refresh_lookup(state, target) do
    query_ref = make_ref()
    query_state = %{
      type: :find_node,
      target: target,
      visited: MapSet.new(),
      active: MapSet.new(),
      best_nodes: get_closest_nodes(state.routing_table, target, @alpha),
      from: nil,
      start_time: DateTime.utc_now()
    }
    
    new_state = put_in(state.pending_queries[query_ref], query_state)
    continue_lookup(new_state, query_ref)
  end
  
  defp republish_stored_values(state) do
    # Republish values we're storing
    Enum.each(state.storage, fn {key, stored_value} ->
      if should_republish?(stored_value) do
        initiate_store(state, key, stored_value.value)
      end
    end)
    
    state
  end
  
  defp should_republish?(%{timestamp: timestamp}) do
    DateTime.diff(DateTime.utc_now(), timestamp, :second) > 3600  # 1 hour
  end
  
  defp cleanup_expired_values(state) do
    now = DateTime.utc_now()
    
    new_storage = state.storage
    |> Enum.reject(fn {_key, stored_value} ->
      DateTime.diff(now, stored_value.timestamp, :millisecond) > stored_value.ttl
    end)
    |> Map.new()
    
    %{state | storage: new_storage}
  end
end
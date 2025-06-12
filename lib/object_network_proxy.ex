defmodule Object.NetworkProxy do
  @moduledoc """
  Network proxy for transparent remote Object access.
  
  Provides a local proxy that forwards method calls to remote Objects
  over the network. Handles connection management, caching, retries,
  and circuit breaking automatically.
  
  ## Features
  
  - Transparent proxy objects that behave like local objects
  - Method call forwarding over network
  - Response caching with TTL
  - Automatic retry with exponential backoff
  - Circuit breaker pattern for fault tolerance
  - Latency-aware routing
  - Connection pooling
  """
  
  use GenServer
  require Logger
  
  alias Object.{NetworkTransport, NetworkProtocol, Serialization, DistributedRegistry}
  
  @default_timeout 5_000
  @max_retries 3
  @cache_ttl 60_000  # 1 minute
  @circuit_breaker_threshold 5
  @circuit_breaker_timeout 30_000
  
  @type proxy_state :: %{
    object_id: String.t(),
    remote_node: node_info() | nil,
    cache: %{String.t() => cached_result()},
    pending_calls: %{reference() => pending_call()},
    circuit_breaker: circuit_breaker_state(),
    stats: proxy_stats(),
    config: proxy_config()
  }
  
  @type node_info :: %{
    id: binary(),
    address: String.t(),
    port: non_neg_integer(),
    connection_id: String.t() | nil
  }
  
  @type cached_result :: %{
    result: term(),
    timestamp: DateTime.t(),
    ttl: non_neg_integer()
  }
  
  @type pending_call :: %{
    from: GenServer.from(),
    method: String.t(),
    args: list(),
    attempt: non_neg_integer(),
    start_time: DateTime.t()
  }
  
  @type circuit_breaker_state :: %{
    state: :closed | :open | :half_open,
    failure_count: non_neg_integer(),
    last_failure: DateTime.t() | nil,
    success_count: non_neg_integer()
  }
  
  @type proxy_stats :: %{
    total_calls: non_neg_integer(),
    successful_calls: non_neg_integer(),
    failed_calls: non_neg_integer(),
    cache_hits: non_neg_integer(),
    avg_latency_ms: float()
  }
  
  @type proxy_config :: %{
    timeout: timeout(),
    max_retries: non_neg_integer(),
    cache_ttl: non_neg_integer(),
    prefetch: boolean()
  }
  
  # Client API
  
  @doc """
  Creates a proxy for a remote object.
  """
  @spec create(String.t(), Keyword.t()) :: {:ok, pid()} | {:error, term()}
  def create(object_id, opts \\ []) do
    GenServer.start_link(__MODULE__, {object_id, opts})
  end
  
  @doc """
  Calls a method on the remote object.
  """
  @spec call(pid(), String.t(), list(), timeout()) :: term()
  def call(proxy, method, args \\ [], timeout \\ @default_timeout) do
    GenServer.call(proxy, {:call, method, args}, timeout)
  end
  
  @doc """
  Casts a message to the remote object (fire-and-forget).
  """
  @spec cast(pid(), String.t(), list()) :: :ok
  def cast(proxy, method, args \\ []) do
    GenServer.cast(proxy, {:cast, method, args})
  end
  
  @doc """
  Gets statistics about the proxy.
  """
  @spec get_stats(pid()) :: proxy_stats()
  def get_stats(proxy) do
    GenServer.call(proxy, :get_stats)
  end
  
  @doc """
  Refreshes the remote node location.
  """
  @spec refresh_location(pid()) :: :ok | {:error, term()}
  def refresh_location(proxy) do
    GenServer.call(proxy, :refresh_location)
  end
  
  @doc """
  Clears the method cache.
  """
  @spec clear_cache(pid()) :: :ok
  def clear_cache(proxy) do
    GenServer.cast(proxy, :clear_cache)
  end
  
  # Server Callbacks
  
  @impl true
  def init({object_id, opts}) do
    config = %{
      timeout: Keyword.get(opts, :timeout, @default_timeout),
      max_retries: Keyword.get(opts, :max_retries, @max_retries),
      cache_ttl: Keyword.get(opts, :cache_ttl, @cache_ttl),
      prefetch: Keyword.get(opts, :prefetch, false)
    }
    
    state = %{
      object_id: object_id,
      remote_node: nil,
      cache: %{},
      pending_calls: %{},
      circuit_breaker: %{
        state: :closed,
        failure_count: 0,
        last_failure: nil,
        success_count: 0
      },
      stats: %{
        total_calls: 0,
        successful_calls: 0,
        failed_calls: 0,
        cache_hits: 0,
        avg_latency_ms: 0.0
      },
      config: config
    }
    
    # Schedule location lookup
    send(self(), :lookup_remote_node)
    
    # Schedule periodic cache cleanup
    Process.send_after(self(), :cleanup_cache, 60_000)
    
    {:ok, state}
  end
  
  @impl true
  def handle_call({:call, method, args}, from, state) do
    # Check cache first
    cache_key = generate_cache_key(method, args)
    
    case get_cached_result(state.cache, cache_key) do
      {:ok, result} ->
        # Cache hit
        new_stats = update_stats(state.stats, :cache_hit)
        {:reply, result, %{state | stats: new_stats}}
        
      :miss ->
        # Check circuit breaker
        case check_circuit_breaker(state.circuit_breaker) do
          :ok ->
            # Proceed with remote call
            handle_remote_call(state, method, args, from)
            
          {:error, :circuit_open} ->
            {:reply, {:error, :circuit_breaker_open}, state}
        end
    end
  end
  
  @impl true
  def handle_call(:get_stats, _from, state) do
    {:reply, state.stats, state}
  end
  
  @impl true
  def handle_call(:refresh_location, _from, state) do
    case lookup_remote_node(state.object_id) do
      {:ok, node_info} ->
        {:reply, :ok, %{state | remote_node: node_info}}
      {:error, reason} = error ->
        {:reply, error, state}
    end
  end
  
  @impl true
  def handle_cast({:cast, method, args}, state) do
    # Fire and forget - no response needed
    case state.remote_node do
      nil ->
        Logger.warn("Cannot cast to unknown remote node")
        {:noreply, state}
        
      node ->
        send_remote_cast(state, node, method, args)
        new_stats = update_stats(state.stats, :cast)
        {:noreply, %{state | stats: new_stats}}
    end
  end
  
  @impl true
  def handle_cast(:clear_cache, state) do
    {:noreply, %{state | cache: %{}}}
  end
  
  @impl true
  def handle_info(:lookup_remote_node, state) do
    case lookup_remote_node(state.object_id) do
      {:ok, node_info} ->
        # Establish connection if needed
        new_node = ensure_connection(node_info)
        
        # Prefetch common methods if enabled
        new_state = if state.config.prefetch do
          prefetch_common_methods(%{state | remote_node: new_node})
        else
          %{state | remote_node: new_node}
        end
        
        {:noreply, new_state}
        
      {:error, reason} ->
        Logger.error("Failed to lookup remote node: #{inspect(reason)}")
        # Retry in 5 seconds
        Process.send_after(self(), :lookup_remote_node, 5_000)
        {:noreply, state}
    end
  end
  
  @impl true
  def handle_info({:remote_response, call_ref, response}, state) do
    case Map.get(state.pending_calls, call_ref) do
      nil ->
        # Late or duplicate response
        {:noreply, state}
        
      %{from: from, method: method, args: args, start_time: start_time} ->
        # Calculate latency
        latency = DateTime.diff(DateTime.utc_now(), start_time, :millisecond)
        
        # Handle response
        case response do
          {:ok, result} ->
            # Success - reply and cache
            GenServer.reply(from, result)
            
            # Update cache
            cache_key = generate_cache_key(method, args)
            new_cache = put_cached_result(state.cache, cache_key, result, state.config.cache_ttl)
            
            # Update stats and circuit breaker
            new_stats = update_stats(state.stats, :success, latency)
            new_cb = update_circuit_breaker(state.circuit_breaker, :success)
            
            new_state = %{state | 
              cache: new_cache,
              stats: new_stats,
              circuit_breaker: new_cb,
              pending_calls: Map.delete(state.pending_calls, call_ref)
            }
            
            {:noreply, new_state}
            
          {:error, reason} ->
            # Failure - maybe retry
            handle_call_failure(state, call_ref, reason)
        end
    end
  end
  
  @impl true
  def handle_info(:cleanup_cache, state) do
    # Remove expired cache entries
    now = DateTime.utc_now()
    
    new_cache = state.cache
    |> Enum.reject(fn {_key, entry} ->
      DateTime.diff(now, entry.timestamp, :millisecond) > entry.ttl
    end)
    |> Map.new()
    
    # Schedule next cleanup
    Process.send_after(self(), :cleanup_cache, 60_000)
    
    {:noreply, %{state | cache: new_cache}}
  end
  
  @impl true
  def handle_info(:circuit_breaker_timeout, state) do
    # Try to transition from open to half-open
    case state.circuit_breaker.state do
      :open ->
        new_cb = %{state.circuit_breaker | state: :half_open}
        {:noreply, %{state | circuit_breaker: new_cb}}
        
      _ ->
        {:noreply, state}
    end
  end
  
  # Private Functions
  
  defp handle_remote_call(state, method, args, from) do
    case state.remote_node do
      nil ->
        # No remote node known yet
        {:reply, {:error, :remote_node_unknown}, state}
        
      node ->
        # Create pending call record
        call_ref = make_ref()
        pending_call = %{
          from: from,
          method: method,
          args: args,
          attempt: 1,
          start_time: DateTime.utc_now()
        }
        
        # Send remote request
        case send_remote_request(state, node, method, args, call_ref) do
          :ok ->
            new_state = put_in(state.pending_calls[call_ref], pending_call)
            new_stats = update_stats(state.stats, :call_initiated)
            
            # Set timeout for response
            Process.send_after(self(), {:call_timeout, call_ref}, state.config.timeout)
            
            {:noreply, %{new_state | stats: new_stats}}
            
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
  end
  
  defp handle_call_failure(state, call_ref, reason) do
    case Map.get(state.pending_calls, call_ref) do
      %{attempt: attempt} = call when attempt < state.config.max_retries ->
        # Retry with exponential backoff
        backoff = :math.pow(2, attempt) * 100 |> round()
        Process.send_after(self(), {:retry_call, call_ref}, backoff)
        
        new_call = %{call | attempt: attempt + 1}
        new_state = put_in(state.pending_calls[call_ref], new_call)
        {:noreply, new_state}
        
      %{from: from} ->
        # Max retries exceeded
        GenServer.reply(from, {:error, reason})
        
        # Update stats and circuit breaker
        new_stats = update_stats(state.stats, :failure)
        new_cb = update_circuit_breaker(state.circuit_breaker, :failure)
        
        new_state = %{state | 
          stats: new_stats,
          circuit_breaker: new_cb,
          pending_calls: Map.delete(state.pending_calls, call_ref)
        }
        
        {:noreply, new_state}
        
      nil ->
        {:noreply, state}
    end
  end
  
  defp send_remote_request(state, node, method, args, call_ref) do
    message = NetworkProtocol.create_request(
      state.object_id,
      method,
      args,
      metadata: %{
        proxy_id: self() |> :erlang.pid_to_list() |> to_string(),
        call_ref: Base.encode64(:erlang.term_to_binary(call_ref))
      }
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
        
      {:error, reason} ->
        {:error, {:encoding_failed, reason}}
    end
  end
  
  defp send_remote_cast(state, node, method, args) do
    message = NetworkProtocol.create_cast(
      state.object_id,
      method,
      args
    )
    
    case NetworkProtocol.encode(message) do
      {:ok, encoded} ->
        NetworkTransport.send_to(node.address, node.port, encoded)
        :ok
        
      {:error, reason} ->
        Logger.error("Failed to encode cast: #{inspect(reason)}")
        :error
    end
  end
  
  defp lookup_remote_node(object_id) do
    case DistributedRegistry.lookup_object(object_id) do
      {:ok, object_info} ->
        # Extract node information from object metadata
        node_info = %{
          id: Map.get(object_info, :node_id),
          address: Map.get(object_info, :address),
          port: Map.get(object_info, :port),
          connection_id: nil
        }
        {:ok, node_info}
        
      error ->
        error
    end
  end
  
  defp ensure_connection(node_info) do
    case NetworkTransport.connect(node_info.address, node_info.port) do
      {:ok, conn_id} ->
        %{node_info | connection_id: conn_id}
        
      {:error, _reason} ->
        # Connection failed, but we can still try direct sends
        node_info
    end
  end
  
  defp prefetch_common_methods(state) do
    # Prefetch common read-only methods
    common_methods = ["get_state", "get_metadata", "get_type"]
    
    Enum.each(common_methods, fn method ->
      Task.start(fn ->
        send_remote_request(state, state.remote_node, method, [], make_ref())
      end)
    end)
    
    state
  end
  
  # Cache Management
  
  defp generate_cache_key(method, args) do
    :erlang.phash2({method, args}) |> Integer.to_string()
  end
  
  defp get_cached_result(cache, key) do
    case Map.get(cache, key) do
      nil ->
        :miss
        
      %{result: result, timestamp: timestamp, ttl: ttl} ->
        if DateTime.diff(DateTime.utc_now(), timestamp, :millisecond) <= ttl do
          {:ok, result}
        else
          :miss
        end
    end
  end
  
  defp put_cached_result(cache, key, result, ttl) do
    entry = %{
      result: result,
      timestamp: DateTime.utc_now(),
      ttl: ttl
    }
    
    Map.put(cache, key, entry)
  end
  
  # Circuit Breaker
  
  defp check_circuit_breaker(%{state: :closed}), do: :ok
  defp check_circuit_breaker(%{state: :half_open}), do: :ok
  defp check_circuit_breaker(%{state: :open}), do: {:error, :circuit_open}
  
  defp update_circuit_breaker(cb, :success) do
    case cb.state do
      :half_open ->
        # Transition back to closed
        %{cb | 
          state: :closed,
          failure_count: 0,
          success_count: cb.success_count + 1
        }
        
      _ ->
        %{cb | 
          failure_count: 0,
          success_count: cb.success_count + 1
        }
    end
  end
  
  defp update_circuit_breaker(cb, :failure) do
    new_failure_count = cb.failure_count + 1
    
    if new_failure_count >= @circuit_breaker_threshold do
      # Open the circuit
      Process.send_after(self(), :circuit_breaker_timeout, @circuit_breaker_timeout)
      
      %{cb | 
        state: :open,
        failure_count: new_failure_count,
        last_failure: DateTime.utc_now()
      }
    else
      %{cb | failure_count: new_failure_count}
    end
  end
  
  # Statistics
  
  defp update_stats(stats, :cache_hit) do
    %{stats | 
      cache_hits: stats.cache_hits + 1
    }
  end
  
  defp update_stats(stats, :call_initiated) do
    %{stats | 
      total_calls: stats.total_calls + 1
    }
  end
  
  defp update_stats(stats, :cast) do
    %{stats | 
      total_calls: stats.total_calls + 1,
      successful_calls: stats.successful_calls + 1
    }
  end
  
  defp update_stats(stats, :success, latency) do
    # Update moving average latency
    new_avg = (stats.avg_latency_ms * stats.successful_calls + latency) / 
              (stats.successful_calls + 1)
    
    %{stats | 
      successful_calls: stats.successful_calls + 1,
      avg_latency_ms: new_avg
    }
  end
  
  defp update_stats(stats, :failure) do
    %{stats | 
      failed_calls: stats.failed_calls + 1
    }
  end
end
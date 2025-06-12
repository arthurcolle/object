defmodule Object.MessageRouter do
  @moduledoc """
  High-performance message routing service using GenStage for backpressure.
  Handles message delivery between objects with priority queuing and load balancing.
  """
  
  use GenStage
  require Logger
  
  
  @priority_multipliers %{
    critical: 4,
    high: 3,
    medium: 2,
    low: 1
  }
  
  # Client API
  
  @doc """
  Processes a batch of messages through the router.
  
  ## Parameters
  - `router_state`: Current router state
  - `message_batch`: List of messages to process
  
  ## Returns
  `{updated_router_state, batch_results}` where batch_results contains delivery statistics
  """
  def process_message_batch(router_state, message_batch) do
    start_time = System.monotonic_time(:microsecond)
    
    results = Enum.flat_map(message_batch, fn message ->
      # Handle both single recipient (to) and multiple recipients (recipients)
      recipients = cond do
        Map.has_key?(message, :to) -> [message.to]
        Map.has_key?(message, :recipients) -> message.recipients
        true -> []
      end
      
      Enum.map(recipients, fn recipient ->
        result = case Registry.lookup(Object.Registry, recipient) do
          [{pid, _}] when is_pid(pid) ->
            try do
              GenServer.cast(pid, {:receive_message, message})
              :success
            catch
              :exit, reason -> {:error, {:target_process_died, reason}}
              error -> {:error, error}
            end
          
          [] ->
            {:error, :target_not_found}
        end
        
        {"#{message.id}->#{recipient}", result}
      end)
    end)
    
    # Count results
    batch_results = Enum.reduce(results, %{delivered: 0, failed: 0, dropped: 0}, fn
      {_id, :success}, acc -> 
        %{acc | delivered: acc.delivered + 1}
      {_id, {:error, _reason}}, acc -> 
        %{acc | failed: acc.failed + 1}
    end)
    
    # Update router stats
    end_time = System.monotonic_time(:microsecond)
    batch_latency = end_time - start_time
    
    updated_stats = %{
      router_state.performance_stats | 
      messages_routed: router_state.performance_stats.messages_routed + length(message_batch),
      average_latency: update_average_latency(
        router_state.performance_stats.average_latency,
        router_state.performance_stats.messages_routed,
        batch_latency,
        length(message_batch)
      )
    }
    
    updated_router = %{router_state | performance_stats: updated_stats}
    
    {updated_router, batch_results}
  end

  @doc """
  Creates a new message router for the given objects and dyads.
  
  ## Parameters
  - `objects`: List of objects to route messages between
  - `initial_dyads`: Initial dyad connections
  
  ## Returns
  Router state with initialized routing state
  """
  def new(objects, initial_dyads \\ %{}) do
    router_state = %{
      objects: objects,
      dyads: initial_dyads,
      routing_table: build_routing_table(objects),
      message_queue: :queue.new(),
      performance_stats: %{
        messages_routed: 0,
        average_latency: 0.0,
        dropped_messages: 0
      }
    }
    
    router_state
  end
  
  @doc """
  Starts the message router GenStage producer.
  
  ## Returns
  `{:ok, pid}` on successful startup
  """
  def start_link(_) do
    GenStage.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  @doc """
  Routes a message through the system with priority queuing.
  
  ## Parameters
  - `message`: Message struct with routing information
  
  ## Returns
  `:ok` - message is queued for delivery
  
  ## Examples
      iex> message = %{id: "msg1", from: "obj1", to: "obj2", ...}
      iex> Object.MessageRouter.route_message(message)
      :ok
  """
  def route_message(message) do
    GenStage.cast(__MODULE__, {:route_message, message})
  end
  
  @doc """
  Gets performance statistics for the message router.
  
  ## Returns
  Map with pending messages, delivery rate, success/failure counts
  """
  def get_routing_stats() do
    GenStage.call(__MODULE__, :get_stats)
  end
  
  @doc """
  Gets active dyads from router state.
  """
  def get_active_dyads(router_state) do
    Map.get(router_state, :dyads, %{})
  end
  
  @doc """
  Gets objects from router state.
  """
  def get_objects(router_state) do
    Map.get(router_state, :objects, [])
  end
  
  @doc """
  Updates the router topology with new dyads.
  """
  def update_topology(router_state, new_dyads) do
    Map.put(router_state, :dyads, new_dyads)
  end
  
  # Producer callbacks
  
  @impl true
  def init(:ok) do
    state = %{
      pending_messages: :queue.new(),
      delivered_count: 0,
      failed_count: 0,
      started_at: DateTime.utc_now()
    }
    
    Logger.info("Message Router started")
    {:producer, state}
  end

  # Helper function to dequeue messages based on demand
  defp dequeue_messages(queue, 0, acc), do: {Enum.reverse(acc), queue}
  defp dequeue_messages([], _demand, acc), do: {Enum.reverse(acc), []}
  defp dequeue_messages([msg | rest], demand, acc) when demand > 0 do
    dequeue_messages(rest, demand - 1, [msg | acc])
  end
  
  @impl true
  def handle_cast({:route_message, message}, state) do
    # Validate message TTL
    case is_message_expired?(message) do
      true ->
        Logger.debug("Message #{message.id} expired, dropping")
        {:noreply, [], state}
      
      false ->
        # Add priority score for ordering
        scored_message = add_priority_score(message)
        updated_queue = :queue.in(scored_message, state.pending_messages)
        
        {:noreply, [], %{state | pending_messages: updated_queue}}
    end
  end
  
  @impl true
  def handle_call(:get_stats, _from, state) do
    uptime = DateTime.diff(DateTime.utc_now(), state.started_at, :second)
    
    stats = %{
      pending_messages: :queue.len(state.pending_messages),
      delivered_count: state.delivered_count,
      failed_count: state.failed_count,
      delivery_rate: state.delivered_count / max(1, uptime),
      uptime_seconds: uptime
    }
    
    {:reply, stats, [], state}
  end
  
  @impl true
  def handle_demand(demand, state) when demand > 0 do
    {messages, updated_queue} = dequeue_messages(state.pending_messages, demand, [])
    
    updated_state = %{state | pending_messages: updated_queue}
    
    {:noreply, messages, updated_state}
  end
  
  @impl true
  def handle_info({:delivery_result, message_id, :success}, state) do
    Logger.debug("Message #{message_id} delivered successfully")
    {:noreply, [], %{state | delivered_count: state.delivered_count + 1}}
  end
  
  @impl true
  def handle_info({:delivery_result, message_id, {:error, reason}}, state) do
    Logger.warning("Message #{message_id} delivery failed: #{inspect(reason)}")
    {:noreply, [], %{state | failed_count: state.failed_count + 1}}
  end
  
  # Private functions
  
  defp build_routing_table(objects) do
    # Build a routing table mapping object IDs to their addresses/PIDs
    Enum.reduce(objects, %{}, fn object, acc ->
      object_id = case object do
        %{id: id} -> id
        id when is_binary(id) -> id
        _ -> "unknown"
      end
      
      Map.put(acc, object_id, object)
    end)
  end
  
  defp is_message_expired?(message) do
    expiry_time = DateTime.add(message.timestamp, message.ttl, :second)
    DateTime.compare(DateTime.utc_now(), expiry_time) == :gt
  end
  
  defp add_priority_score(message) do
    base_score = @priority_multipliers[message.priority] || 1
    
    # Age factor - older messages get higher priority
    age_seconds = DateTime.diff(DateTime.utc_now(), message.timestamp, :second)
    age_factor = min(age_seconds / 60.0, 10.0)  # Max 10x boost after 10 minutes
    
    score = base_score + age_factor
    
    Map.put(message, :priority_score, score)
  end
  
  defp update_average_latency(current_avg, current_count, new_latency, new_count) do
    if current_count == 0 do
      new_latency / new_count
    else
      total_current = current_avg * current_count
      total_new = new_latency
      (total_current + total_new) / (current_count + new_count)
    end
  end
end

defmodule Object.MessageConsumer do
  @moduledoc """
  GenStage consumer for processing messages from the MessageRouter.
  Handles actual message delivery to target objects.
  """
  
  use GenStage
  require Logger
  
  def start_link(consumer_id) do
    GenStage.start_link(__MODULE__, consumer_id, name: :"#{__MODULE__}_#{consumer_id}")
  end
  
  @impl true
  def init(consumer_id) do
    Logger.info("Message Consumer #{consumer_id} started")
    
    state = %{
      consumer_id: consumer_id,
      processed_count: 0
    }
    
    {:consumer, state, subscribe_to: [Object.MessageRouter]}
  end
  
  @impl true
  def handle_events(messages, _from, state) do
    # Process messages concurrently using Task.async_stream
    results = Task.async_stream(messages, &deliver_message/1, 
      max_concurrency: 10,
      timeout: 5_000,
      on_timeout: :kill_task
    )
    
    # Report delivery results back to router
    Enum.each(results, fn
      {:ok, {message_id, result}} ->
        send(Object.MessageRouter, {:delivery_result, message_id, result})
      
      {:exit, reason} ->
        Logger.error("Message delivery task crashed: #{inspect(reason)}")
    end)
    
    updated_state = %{state | 
      processed_count: state.processed_count + length(messages)
    }
    
    {:noreply, [], updated_state}
  end
  
  # Private functions
  
  defp deliver_message(message) do
    result = case Registry.lookup(Object.Registry, message.to) do
      [{pid, _}] when is_pid(pid) ->
        try do
          GenServer.cast(pid, {:receive_message, message})
          :success
        catch
          :exit, reason -> {:error, {:target_process_died, reason}}
          error -> {:error, error}
        end
      
      [] ->
        {:error, :target_not_found}
    end
    
    {message.id, result}
  end
end

defmodule Object.MessageRouterSupervisor do
  @moduledoc """
  Supervisor for the message routing system with multiple consumers.
  """
  
  use Supervisor
  
  def start_link(_) do
    Supervisor.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  @impl true
  def init(:ok) do
    # Number of consumers based on system cores
    num_consumers = max(2, System.schedulers_online())
    
    children = [
      Object.MessageRouter
    ] ++ 
    for i <- 1..num_consumers do
      %{
        id: {Object.MessageConsumer, i},
        start: {Object.MessageConsumer, :start_link, [i]}
      }
    end
    
    Supervisor.init(children, strategy: :one_for_one)
  end
end
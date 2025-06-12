defmodule Object.SchemaRegistry do
  @moduledoc """
  ETS-based registry for tracking object schemas and evolution.
  Provides fast lookup and atomic updates for the object schema space.
  """
  
  use GenServer
  require Logger
  
  @table_name :object_schema_registry
  @evolution_table :schema_evolution_history
  @compatibility_cache :compatibility_cache
  
  # Client API
  
  @doc """
  Starts the schema registry GenServer.
  
  Creates ETS tables for fast schema lookup and evolution tracking.
  
  ## Returns
  
  - `{:ok, pid}` - Successfully started schema registry
  """
  def start_link(_) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  @doc """
  Registers an object's schema in the registry.
  
  ## Parameters
  
  - `object` - Object struct to register
  
  ## Returns
  
  - `:ok` - Object schema registered successfully
  """
  def register_object(object) do
    GenServer.call(__MODULE__, {:register_object, object})
  end
  
  @doc """
  Unregisters an object from the schema registry.
  
  ## Parameters
  
  - `object_id` - ID of the object to unregister
  
  ## Returns
  
  - `:ok` - Object unregistered successfully
  """
  def unregister_object(object_id) do
    GenServer.call(__MODULE__, {:unregister_object, object_id})
  end
  
  @doc """
  Gets the schema for a specific object.
  
  ## Parameters
  
  - `object_id` - ID of the object
  
  ## Returns
  
  - `{:ok, schema}` - Object schema found
  - `{:error, :not_found}` - Object not found
  """
  def get_object_schema(object_id) do
    case :ets.lookup(@table_name, object_id) do
      [{^object_id, schema}] -> {:ok, schema}
      [] -> {:error, :not_found}
    end
  end
  
  @doc """
  Lists all registered objects and their schemas.
  
  ## Returns
  
  List of tuples with object IDs and schemas
  """
  def list_objects() do
    :ets.tab2list(@table_name)
  end
  
  @doc """
  Lists objects filtered by subtype.
  
  ## Parameters
  
  - `object_type` - Object subtype to filter by
  
  ## Returns
  
  List of objects matching the specified type
  """
  def list_objects_by_type(object_type) do
    :ets.match_object(@table_name, {:_, %{subtype: object_type, _: :_}})
  end
  
  @doc """
  Updates an object's schema and records the evolution.
  
  ## Parameters
  
  - `object_id` - ID of the object to update
  - `schema_updates` - Schema changes to apply
  
  ## Returns
  
  - `:ok` - Schema updated successfully
  - `{:error, :object_not_found}` - Object not found
  """
  def update_object_schema(object_id, schema_updates) do
    GenServer.call(__MODULE__, {:update_schema, object_id, schema_updates})
  end
  
  @doc """
  Gets the schema evolution history for an object.
  
  ## Parameters
  
  - `object_id` - ID of the object
  
  ## Returns
  
  - `{:ok, history}` - Evolution history list
  """
  def get_schema_evolution_history(object_id) do
    case :ets.lookup(@evolution_table, object_id) do
      [{^object_id, history}] -> {:ok, history}
      [] -> {:ok, []}
    end
  end
  
  @doc """
  Gets all object schemas as a map.
  
  ## Returns
  
  Map of object IDs to schemas
  """
  def get_all_schemas() do
    :ets.tab2list(@table_name)
    |> Enum.into(%{})
  end
  
  @doc """
  Finds objects compatible with the specified object.
  
  ## Parameters
  
  - `object_id` - ID of the reference object
  - `compatibility_threshold` - Minimum compatibility score (default: 0.7)
  
  ## Returns
  
  - `{:ok, compatible_objects}` - List of compatible objects with scores
  - `{:error, reason}` - Object not found or other error
  """
  def find_compatible_objects(object_id, compatibility_threshold \\ 0.7) do
    case get_object_schema(object_id) do
      {:ok, target_schema} ->
        # Check cache first for recent calculations
        cache_key = {object_id, compatibility_threshold}
        
        case :ets.lookup(@compatibility_cache, cache_key) do
          [{^cache_key, {result, timestamp}}] ->
            # Use cached result if less than 5 minutes old
            if System.monotonic_time(:second) - timestamp < 300 do
              {:ok, result}
            else
              perform_compatibility_search(object_id, target_schema, compatibility_threshold, cache_key)
            end
          
          [] ->
            perform_compatibility_search(object_id, target_schema, compatibility_threshold, cache_key)
        end
      
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  # Server callbacks
  
  @impl true
  def init(:ok) do
    # Create ETS tables with optimized settings
    :ets.new(@table_name, [:named_table, :public, :set, {:read_concurrency, true}])
    :ets.new(@evolution_table, [:named_table, :public, :set, {:read_concurrency, true}])
    :ets.new(@compatibility_cache, [:named_table, :public, :set, {:read_concurrency, true}, {:write_concurrency, true}])
    
    Logger.info("Object Schema Registry started with compatibility caching")
    {:ok, %{}}
  end
  
  @impl true
  def handle_call({:register_object, object}, _from, state) do
    schema = extract_schema(object)
    :ets.insert(@table_name, {object.id, schema})
    
    # Initialize evolution history
    :ets.insert(@evolution_table, {object.id, []})
    
    Logger.debug("Registered object schema for #{object.id}")
    {:reply, :ok, state}
  end
  
  @impl true
  def handle_call({:unregister_object, object_id}, _from, state) do
    :ets.delete(@table_name, object_id)
    :ets.delete(@evolution_table, object_id)
    
    Logger.debug("Unregistered object schema for #{object_id}")
    {:reply, :ok, state}
  end
  
  @impl true
  def handle_call({:update_schema, object_id, schema_updates}, _from, state) do
    case :ets.lookup(@table_name, object_id) do
      [{^object_id, current_schema}] ->
        # Record evolution step
        evolution_entry = %{
          timestamp: DateTime.utc_now(),
          from_schema: current_schema,
          updates: schema_updates,
          evolution_type: determine_evolution_type(schema_updates)
        }
        
        # Update evolution history
        [{^object_id, history}] = :ets.lookup(@evolution_table, object_id)
        updated_history = [evolution_entry | history] |> Enum.take(100)  # Keep last 100 changes
        :ets.insert(@evolution_table, {object_id, updated_history})
        
        # Update schema
        updated_schema = Map.merge(current_schema, schema_updates)
        :ets.insert(@table_name, {object_id, updated_schema})
        
        Logger.debug("Updated schema for object #{object_id}")
        {:reply, :ok, state}
      
      [] ->
        {:reply, {:error, :object_not_found}, state}
    end
  end
  
  # Private functions
  
  defp extract_schema(object) do
    %{
      id: object.id,
      subtype: object.subtype,
      methods: object.methods,
      goal_type: extract_goal_type(object.goal),
      world_model_structure: Map.keys(object.world_model),
      meta_dsl_constructs: object.meta_dsl.constructs,
      state_dimensions: Map.keys(object.state),
      created_at: object.created_at,
      last_updated: object.updated_at
    }
  end
  
  defp extract_goal_type(goal_fn) when is_function(goal_fn) do
    # Try to determine goal type from function info
    case Function.info(goal_fn) do
      info when is_list(info) -> 
        case Keyword.get(info, :module) do
          nil -> :anonymous_function
          module -> "#{module}.#{Keyword.get(info, :name, "unknown")}"
        end
      _ -> :custom_function
    end
  end
  
  defp extract_goal_type(_), do: :unknown
  
  defp determine_evolution_type(schema_updates) do
    cond do
      Map.has_key?(schema_updates, :methods) -> :method_evolution
      Map.has_key?(schema_updates, :goal_type) -> :goal_evolution
      Map.has_key?(schema_updates, :state_dimensions) -> :state_evolution
      Map.has_key?(schema_updates, :meta_dsl_constructs) -> :meta_dsl_evolution
      true -> :general_evolution
    end
  end
  
  defp calculate_compatibility(schema1, schema2) do
    # Multi-dimensional compatibility calculation
    method_compatibility = calculate_method_compatibility(schema1.methods, schema2.methods)
    goal_compatibility = calculate_goal_compatibility(schema1.goal_type, schema2.goal_type)
    state_compatibility = calculate_state_compatibility(schema1.state_dimensions, schema2.state_dimensions)
    meta_dsl_compatibility = calculate_meta_dsl_compatibility(schema1.meta_dsl_constructs, schema2.meta_dsl_constructs)
    
    # Weighted average
    0.3 * method_compatibility + 
    0.3 * goal_compatibility + 
    0.2 * state_compatibility + 
    0.2 * meta_dsl_compatibility
  end
  
  defp calculate_method_compatibility(methods1, methods2) do
    set1 = MapSet.new(methods1)
    set2 = MapSet.new(methods2)
    
    intersection = MapSet.intersection(set1, set2)
    union = MapSet.union(set1, set2)
    
    if MapSet.size(union) == 0 do
      1.0
    else
      MapSet.size(intersection) / MapSet.size(union)
    end
  end
  
  defp calculate_goal_compatibility(goal1, goal2) do
    if goal1 == goal2, do: 1.0, else: 0.5
  end
  
  defp calculate_state_compatibility(state_dims1, state_dims2) do
    set1 = MapSet.new(state_dims1)
    set2 = MapSet.new(state_dims2)
    
    intersection = MapSet.intersection(set1, set2)
    union = MapSet.union(set1, set2)
    
    if MapSet.size(union) == 0 do
      1.0
    else
      MapSet.size(intersection) / MapSet.size(union)
    end
  end
  
  defp calculate_meta_dsl_compatibility(constructs1, constructs2) when is_list(constructs1) and is_list(constructs2) do
    set1 = MapSet.new(constructs1)
    set2 = MapSet.new(constructs2)
    
    intersection = MapSet.intersection(set1, set2)
    union = MapSet.union(set1, set2)
    
    if MapSet.size(union) == 0 do
      1.0
    else
      MapSet.size(intersection) / MapSet.size(union)
    end
  end
  
  defp calculate_meta_dsl_compatibility(_, _), do: 0.5

  defp perform_compatibility_search(object_id, target_schema, compatibility_threshold, cache_key) do
    result = :ets.tab2list(@table_name)
             |> Enum.reject(fn {id, _schema} -> id == object_id end)
             |> Enum.map(fn {id, schema} -> 
               {id, calculate_compatibility(target_schema, schema)}
             end)
             |> Enum.filter(fn {_id, compatibility} -> 
               compatibility >= compatibility_threshold
             end)
             |> Enum.sort_by(fn {_id, compatibility} -> compatibility end, :desc)
    
    # Cache the result
    :ets.insert(@compatibility_cache, {cache_key, {result, System.monotonic_time(:second)}})
    
    {:ok, result}
  end
end
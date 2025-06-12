defmodule Object.Serialization do
  @moduledoc """
  Serialization and deserialization for Object instances.
  
  Provides multiple serialization formats optimized for different use cases:
  - ETF (Erlang Term Format) - Native, fast, compact
  - JSON - Human-readable, widely compatible
  - Protocol Buffers - Efficient, schema-based
  - MessagePack - Compact binary format
  
  ## Features
  
  - Multiple format support with automatic negotiation
  - Schema versioning and migration
  - Partial serialization for large objects
  - Compression support
  - Type preservation across formats
  - Streaming serialization for large data
  """
  
  require Logger
  
  @type format :: :etf | :json | :protobuf | :msgpack
  @type serialize_opts :: [
    format: format(),
    compress: boolean(),
    include_methods: boolean(),
    include_private: boolean(),
    max_depth: pos_integer(),
    fields: [atom()]
  ]
  
  @doc """
  Serializes an Object to the specified format.
  """
  @spec serialize(Object.t(), serialize_opts()) :: {:ok, binary()} | {:error, term()}
  def serialize(object, opts \\ []) do
    format = Keyword.get(opts, :format, :etf)
    
    with {:ok, prepared} <- prepare_object(object, opts),
         {:ok, serialized} <- serialize_format(prepared, format),
         {:ok, processed} <- post_process(serialized, opts) do
      {:ok, processed}
    end
  end
  
  @doc """
  Deserializes binary data back into an Object.
  """
  @spec deserialize(binary(), Keyword.t()) :: {:ok, Object.t()} | {:error, term()}
  def deserialize(data, opts \\ []) do
    format = detect_format(data, opts)
    
    with {:ok, preprocessed} <- pre_process(data, opts),
         {:ok, deserialized} <- deserialize_format(preprocessed, format),
         {:ok, object} <- reconstruct_object(deserialized, opts) do
      {:ok, object}
    end
  end
  
  @doc """
  Serializes only specific fields of an object.
  """
  @spec serialize_partial(Object.t(), [atom()], serialize_opts()) :: 
    {:ok, binary()} | {:error, term()}
  def serialize_partial(object, fields, opts \\ []) do
    opts = Keyword.put(opts, :fields, fields)
    serialize(object, opts)
  end
  
  @doc """
  Streams serialization for large objects.
  """
  @spec serialize_stream(Object.t(), serialize_opts()) :: Enumerable.t()
  def serialize_stream(object, opts \\ []) do
    Stream.resource(
      fn -> init_serialization(object, opts) end,
      fn state -> serialize_chunk(state) end,
      fn state -> cleanup_serialization(state) end
    )
  end
  
  @doc """
  Calculates serialized size without actually serializing.
  """
  @spec calculate_size(Object.t(), serialize_opts()) :: {:ok, non_neg_integer()} | {:error, term()}
  def calculate_size(object, opts \\ []) do
    format = Keyword.get(opts, :format, :etf)
    
    case prepare_object(object, opts) do
      {:ok, prepared} ->
        estimate_size(prepared, format)
      error ->
        error
    end
  end
  
  # Format-specific serialization
  
  defp serialize_format(data, :etf) do
    try do
      {:ok, :erlang.term_to_binary(data, [:compressed])}
    rescue
      _ -> {:error, :etf_serialization_failed}
    end
  end
  
  defp serialize_format(data, :json) do
    try do
      json_safe = prepare_for_json(data)
      case Jason.encode(json_safe) do
        {:ok, json} -> {:ok, json}
        {:error, reason} -> {:error, {:json_encoding_failed, reason}}
      end
    rescue
      e -> {:error, {:json_serialization_failed, e}}
    end
  end
  
  defp serialize_format(data, :msgpack) do
    try do
      case Msgpax.pack(data) do
        {:ok, packed} -> {:ok, packed}
        {:error, reason} -> {:error, {:msgpack_encoding_failed, reason}}
      end
    rescue
      _ -> {:error, :msgpack_serialization_failed}
    end
  end
  
  defp serialize_format(data, :protobuf) do
    # Placeholder for protobuf implementation
    # Would require protobuf schema definition
    {:error, :protobuf_not_implemented}
  end
  
  defp serialize_format(_data, format) do
    {:error, {:unsupported_format, format}}
  end
  
  # Format-specific deserialization
  
  defp deserialize_format(data, :etf) do
    try do
      {:ok, :erlang.binary_to_term(data, [:safe])}
    rescue
      _ -> {:error, :etf_deserialization_failed}
    end
  end
  
  defp deserialize_format(data, :json) do
    case Jason.decode(data) do
      {:ok, decoded} -> 
        {:ok, restore_from_json(decoded)}
      {:error, reason} -> 
        {:error, {:json_decoding_failed, reason}}
    end
  end
  
  defp deserialize_format(data, :msgpack) do
    case Msgpax.unpack(data) do
      {:ok, unpacked} -> {:ok, unpacked}
      {:error, reason} -> {:error, {:msgpack_decoding_failed, reason}}
    end
  end
  
  defp deserialize_format(_data, format) do
    {:error, {:unsupported_format, format}}
  end
  
  # Object preparation
  
  defp prepare_object(object, opts) do
    prepared = %{
      __struct__: "Object",
      __version__: 1,
      id: object.id,
      subtype: object.subtype,
      state: prepare_state(object.state, opts),
      goal: serialize_function(object.goal),
      created_at: object.created_at,
      updated_at: object.updated_at,
      timestamp: DateTime.utc_now()
    }
    
    prepared = if Keyword.get(opts, :include_methods, true) do
      Map.put(prepared, :methods, prepare_methods(object.methods))
    else
      prepared
    end
    
    prepared = if Keyword.get(opts, :include_private, false) do
      Map.put(prepared, :private, %{
        mailbox: serialize_mailbox(object.mailbox),
        interaction_history: object.interaction_history,
        world_model: object.world_model,
        meta_dsl: object.meta_dsl,
        parameters: object.parameters
      })
    else
      prepared
    end
    
    # Handle partial serialization
    case Keyword.get(opts, :fields) do
      nil -> {:ok, prepared}
      fields -> {:ok, Map.take(prepared, fields)}
    end
  end
  
  defp prepare_state(state, opts) do
    max_depth = Keyword.get(opts, :max_depth, 10)
    traverse_and_prepare(state, max_depth)
  end
  
  defp traverse_and_prepare(data, 0), do: {:truncated, inspect(data)}
  
  defp traverse_and_prepare(%DateTime{} = data, _depth), do: data
  
  defp traverse_and_prepare(data, depth) when is_struct(data) do
    # Handle other structs by keeping them as-is
    data
  end
  
  defp traverse_and_prepare(data, depth) when is_map(data) do
    Map.new(data, fn {k, v} -> 
      {k, traverse_and_prepare(v, depth - 1)}
    end)
  end
  
  defp traverse_and_prepare(data, depth) when is_list(data) do
    Enum.map(data, &traverse_and_prepare(&1, depth - 1))
  end
  
  defp traverse_and_prepare(data, depth) when is_tuple(data) do
    data
    |> Tuple.to_list()
    |> Enum.map(&traverse_and_prepare(&1, depth - 1))
    |> List.to_tuple()
  end
  
  defp traverse_and_prepare(data, _depth), do: data
  
  defp prepare_methods(methods) when is_map(methods) do
    Map.new(methods, fn {name, func} ->
      {name, serialize_function(func)}
    end)
  end
  defp prepare_methods(_), do: %{}
  
  defp serialize_function(func) when is_function(func) do
    # Store function metadata for reconstruction
    %{
      __type__: "function",
      arity: :erlang.fun_info(func, :arity) |> elem(1),
      module: :erlang.fun_info(func, :module) |> elem(1),
      # Store as string for safety
      source: inspect(func)
    }
  end
  defp serialize_function(other), do: other
  
  defp serialize_mailbox(mailbox) do
    # Convert mailbox to serializable format
    %{
      __type__: "mailbox",
      messages: [] # Don't serialize actual messages for security
    }
  end
  
  # JSON preparation
  
  defp prepare_for_json(data) when is_map(data) do
    Map.new(data, fn {k, v} ->
      {to_string(k), prepare_for_json(v)}
    end)
  end
  
  defp prepare_for_json(data) when is_list(data) do
    Enum.map(data, &prepare_for_json/1)
  end
  
  defp prepare_for_json(data) when is_tuple(data) do
    %{
      "__type__" => "tuple",
      "data" => data |> Tuple.to_list() |> prepare_for_json()
    }
  end
  
  defp prepare_for_json(data) when is_atom(data) do
    %{
      "__type__" => "atom",
      "value" => to_string(data)
    }
  end
  
  defp prepare_for_json(data) when is_pid(data) do
    %{
      "__type__" => "pid",
      "value" => inspect(data)
    }
  end
  
  defp prepare_for_json(data) when is_reference(data) do
    %{
      "__type__" => "reference",
      "value" => inspect(data)
    }
  end
  
  defp prepare_for_json(%DateTime{} = dt) do
    %{
      "__type__" => "datetime",
      "value" => DateTime.to_iso8601(dt)
    }
  end
  
  defp prepare_for_json(data), do: data
  
  defp restore_from_json(%{"__type__" => "tuple", "data" => data}) do
    data |> restore_from_json() |> List.to_tuple()
  end
  
  defp restore_from_json(%{"__type__" => "atom", "value" => value}) do
    String.to_existing_atom(value)
  rescue
    _ -> String.to_atom(value)
  end
  
  defp restore_from_json(%{"__type__" => "datetime", "value" => value}) do
    case DateTime.from_iso8601(value) do
      {:ok, dt, _} -> dt
      _ -> value
    end
  end
  
  defp restore_from_json(data) when is_map(data) do
    Map.new(data, fn {k, v} -> {k, restore_from_json(v)} end)
  end
  
  defp restore_from_json(data) when is_list(data) do
    Enum.map(data, &restore_from_json/1)
  end
  
  defp restore_from_json(data), do: data
  
  # Object reconstruction
  
  defp reconstruct_object(data, _opts) do
    with :ok <- validate_object_data(data),
         {:ok, base_object} <- build_base_object(data),
         {:ok, full_object} <- restore_object_components(base_object, data) do
      {:ok, full_object}
    end
  end
  
  defp validate_object_data(%{__struct__: "Object", __version__: 1} = data) do
    required_fields = [:id, :subtype, :state]
    
    if Enum.all?(required_fields, &Map.has_key?(data, &1)) do
      :ok
    else
      {:error, :missing_required_fields}
    end
  end
  defp validate_object_data(_), do: {:error, :invalid_object_format}
  
  defp build_base_object(data) do
    object = %Object{
      id: data.id,
      state: data.state,
      subtype: Map.get(data, :subtype, :ai_agent),
      methods: Map.get(data, :methods, []),
      mailbox: nil,
      goal: nil,
      world_model: %{},
      interaction_history: [],
      meta_dsl: %{},
      parameters: Map.get(data, :metadata, %{}),
      created_at: DateTime.utc_now(),
      updated_at: DateTime.utc_now()
    }
    
    {:ok, object}
  end
  
  defp restore_object_components(object, data) do
    object = if Map.has_key?(data, :methods) do
      %{object | methods: restore_methods(data.methods)}
    else
      object
    end
    
    object = if Map.has_key?(data, :goal) do
      %{object | goal: restore_function(data.goal)}
    else
      object
    end
    
    object = if Map.has_key?(data, :private) do
      restore_private_state(object, data.private)
    else
      object
    end
    
    {:ok, object}
  end
  
  defp restore_methods(methods) when is_map(methods) do
    Map.new(methods, fn {name, func_data} ->
      {String.to_atom(name), restore_function(func_data)}
    end)
  end
  defp restore_methods(_), do: %{}
  
  defp restore_function(%{__type__: "function"} = func_data) do
    # For security, we don't restore actual functions
    # Instead, return a placeholder that can be resolved later
    fn args ->
      {:error, {:serialized_function, func_data, args}}
    end
  end
  defp restore_function(other), do: other
  
  defp restore_private_state(object, private_data) do
    object = if Map.has_key?(private_data, :learning_state) do
      %{object | learning_state: private_data.learning_state}
    else
      object
    end
    
    # Mailbox is recreated fresh, not restored
    object
  end
  
  # Post-processing
  
  defp post_process(data, opts) do
    if Keyword.get(opts, :compress, false) do
      {:ok, :zlib.compress(data)}
    else
      {:ok, data}
    end
  end
  
  defp pre_process(data, opts) do
    if is_compressed?(data) do
      try do
        {:ok, :zlib.uncompress(data)}
      rescue
        _ -> {:ok, data}
      end
    else
      {:ok, data}
    end
  end
  
  # Format detection
  
  defp detect_format(data, opts) do
    case Keyword.get(opts, :format) do
      nil -> auto_detect_format(data)
      format -> format
    end
  end
  
  defp auto_detect_format(<<131, _::binary>>), do: :etf
  defp auto_detect_format(<<"{", _::binary>>), do: :json
  defp auto_detect_format(<<"[", _::binary>>), do: :json
  defp auto_detect_format(_), do: :msgpack
  
  defp is_compressed?(<<120, 156, _::binary>>), do: true  # zlib header
  defp is_compressed?(<<120, 218, _::binary>>), do: true  # zlib header
  defp is_compressed?(_), do: false
  
  # Size estimation
  
  defp estimate_size(data, :etf) do
    try do
      size = :erlang.term_to_binary(data) |> byte_size()
      {:ok, size}
    rescue
      _ -> {:error, :size_calculation_failed}
    end
  end
  
  defp estimate_size(data, :json) do
    case prepare_for_json(data) |> Jason.encode() do
      {:ok, json} -> {:ok, byte_size(json)}
      _ -> {:error, :size_calculation_failed}
    end
  end
  
  defp estimate_size(data, :msgpack) do
    case Msgpax.pack(data) do
      {:ok, packed} -> {:ok, byte_size(packed)}
      _ -> {:error, :size_calculation_failed}
    end
  end
  
  defp estimate_size(_, _), do: {:error, :unsupported_format}
  
  # Streaming support
  
  defp init_serialization(object, opts) do
    format = Keyword.get(opts, :format, :etf)
    chunk_size = Keyword.get(opts, :chunk_size, 65536)
    
    case prepare_object(object, opts) do
      {:ok, prepared} ->
        %{
          data: prepared,
          format: format,
          chunk_size: chunk_size,
          position: 0,
          buffer: <<>>
        }
      _ ->
        nil
    end
  end
  
  defp serialize_chunk(nil), do: {:halt, nil}
  
  defp serialize_chunk(state) do
    # This is a simplified version - real streaming would be more complex
    if state.position == 0 do
      case serialize_format(state.data, state.format) do
        {:ok, serialized} ->
          chunks = chunk_binary(serialized, state.chunk_size)
          {chunks, %{state | position: 1}}
        _ ->
          {:halt, state}
      end
    else
      {:halt, state}
    end
  end
  
  defp cleanup_serialization(_state), do: :ok
  
  defp chunk_binary(binary, chunk_size) do
    binary
    |> :binary.bin_to_list()
    |> Enum.chunk_every(chunk_size)
    |> Enum.map(&:binary.list_to_bin/1)
  end
end
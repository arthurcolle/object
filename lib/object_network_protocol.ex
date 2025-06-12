defmodule Object.NetworkProtocol do
  use Bitwise
  @moduledoc """
  Binary protocol for efficient Object communication over network.
  
  Implements a high-performance binary protocol with support for:
  - Message framing and fragmentation
  - Request/response and streaming patterns
  - Compression and encryption
  - Version negotiation
  - Heartbeat and health checking
  
  ## Protocol Format
  
  ```
  |  1 byte  |  1 byte  |  2 bytes  |  4 bytes  |  Variable  |
  | Version  |   Type   |   Flags   |  Length   |   Payload  |
  ```
  
  ## Message Types
  
  - 0x01: HELLO - Initial handshake
  - 0x02: HELLO_ACK - Handshake acknowledgment
  - 0x10: REQUEST - Request message
  - 0x11: RESPONSE - Response message
  - 0x12: STREAM_START - Start streaming
  - 0x13: STREAM_DATA - Stream chunk
  - 0x14: STREAM_END - End streaming
  - 0x20: CAST - One-way message
  - 0x30: HEARTBEAT - Keepalive
  - 0x31: HEARTBEAT_ACK - Keepalive response
  - 0xFF: ERROR - Error message
  """
  
  require Logger
  
  @protocol_version 1
  @max_message_size 16_777_216  # 16MB
  
  # Message types
  @msg_hello 0x01
  @msg_hello_ack 0x02
  @msg_request 0x10
  @msg_response 0x11
  @msg_stream_start 0x12
  @msg_stream_data 0x13
  @msg_stream_end 0x14
  @msg_cast 0x20
  @msg_heartbeat 0x30
  @msg_heartbeat_ack 0x31
  @msg_error 0xFF
  
  # Flags
  @flag_compressed 0x0001
  @flag_encrypted 0x0002
  @flag_fragmented 0x0004
  @flag_priority_high 0x0008
  @flag_requires_ack 0x0010
  
  @type message_type :: :hello | :hello_ack | :request | :response | 
                       :stream_start | :stream_data | :stream_end |
                       :cast | :heartbeat | :heartbeat_ack | :error
  
  @type message :: %{
    version: non_neg_integer(),
    type: message_type(),
    flags: non_neg_integer(),
    id: binary(),
    correlation_id: binary() | nil,
    payload: term(),
    metadata: map()
  }
  
  @type encode_opts :: [
    compress: boolean(),
    encrypt: boolean(),
    encryption_key: binary()
  ]
  
  # Encoding/Decoding API
  
  @doc """
  Encodes a message into binary protocol format.
  """
  @spec encode(message(), encode_opts()) :: {:ok, binary()} | {:error, term()}
  def encode(message, opts \\ []) do
    with {:ok, payload_binary} <- encode_payload(message.payload),
         {:ok, processed_payload} <- process_payload(payload_binary, opts),
         {:ok, header} <- build_header(message, processed_payload, opts) do
      {:ok, header <> processed_payload}
    end
  end
  
  @doc """
  Decodes a binary message into structured format.
  """
  @spec decode(binary(), Keyword.t()) :: {:ok, message()} | {:error, term()}
  def decode(data, opts \\ []) when byte_size(data) >= 8 do
    <<version::8, type::8, flags::16, length::32, payload::binary>> = data
    
    if length > @max_message_size do
      {:error, :message_too_large}
    else
      with {:ok, message_type} <- decode_message_type(type),
           {:ok, processed_payload} <- process_incoming_payload(payload, flags, opts),
           {:ok, decoded_payload} <- decode_payload(processed_payload) do
        
        message = %{
          version: version,
          type: message_type,
          flags: flags,
          payload: decoded_payload,
          id: extract_message_id(decoded_payload),
          correlation_id: extract_correlation_id(decoded_payload),
          metadata: extract_metadata(decoded_payload)
        }
        
        {:ok, message}
      end
    end
  end
  
  def decode(_data, _opts), do: {:error, :invalid_message}
  
  @doc """
  Creates a request message.
  """
  @spec create_request(String.t(), String.t(), list(), Keyword.t()) :: message()
  def create_request(object_id, method, args, opts \\ []) do
    %{
      version: @protocol_version,
      type: :request,
      flags: build_flags(opts),
      id: generate_message_id(),
      correlation_id: nil,
      payload: %{
        object_id: object_id,
        method: method,
        args: args,
        timeout: Keyword.get(opts, :timeout, 5000)
      },
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end
  
  @doc """
  Creates a response message.
  """
  @spec create_response(binary(), term(), Keyword.t()) :: message()
  def create_response(correlation_id, result, opts \\ []) do
    %{
      version: @protocol_version,
      type: :response,
      flags: build_flags(opts),
      id: generate_message_id(),
      correlation_id: correlation_id,
      payload: %{
        result: result,
        error: Keyword.get(opts, :error),
        metadata: Keyword.get(opts, :metadata, %{})
      },
      metadata: %{}
    }
  end
  
  @doc """
  Creates a cast (one-way) message.
  """
  @spec create_cast(String.t(), String.t(), list(), Keyword.t()) :: message()
  def create_cast(object_id, method, args, opts \\ []) do
    %{
      version: @protocol_version,
      type: :cast,
      flags: build_flags(opts),
      id: generate_message_id(),
      correlation_id: nil,
      payload: %{
        object_id: object_id,
        method: method,
        args: args
      },
      metadata: Keyword.get(opts, :metadata, %{})
    }
  end
  
  @doc """
  Creates a heartbeat message.
  """
  @spec create_heartbeat() :: message()
  def create_heartbeat do
    %{
      version: @protocol_version,
      type: :heartbeat,
      flags: 0,
      id: generate_message_id(),
      correlation_id: nil,
      payload: %{timestamp: System.system_time(:millisecond)},
      metadata: %{}
    }
  end
  
  @doc """
  Creates a heartbeat acknowledgment.
  """
  @spec create_heartbeat_ack(binary()) :: message()
  def create_heartbeat_ack(correlation_id) do
    %{
      version: @protocol_version,
      type: :heartbeat_ack,
      flags: 0,
      id: generate_message_id(),
      correlation_id: correlation_id,
      payload: %{timestamp: System.system_time(:millisecond)},
      metadata: %{}
    }
  end
  
  @doc """
  Fragments a large message into smaller chunks.
  """
  @spec fragment_message(binary(), pos_integer()) :: [binary()]
  def fragment_message(data, chunk_size \\ 65536) do
    total_chunks = ceil(byte_size(data) / chunk_size)
    fragment_id = generate_message_id()
    
    data
    |> :binary.bin_to_list()
    |> Enum.chunk_every(chunk_size)
    |> Enum.with_index()
    |> Enum.map(fn {chunk_data, index} ->
      build_fragment(fragment_id, index, total_chunks, chunk_data)
    end)
  end
  
  @doc """
  Reassembles fragmented messages.
  """
  @spec reassemble_fragments([binary()]) :: {:ok, binary()} | {:error, term()}
  def reassemble_fragments(fragments) do
    # Sort by fragment index and concatenate
    sorted = Enum.sort_by(fragments, &extract_fragment_index/1)
    
    if valid_fragment_sequence?(sorted) do
      data = Enum.map(sorted, &extract_fragment_data/1) |> Enum.join()
      {:ok, data}
    else
      {:error, :invalid_fragment_sequence}
    end
  end
  
  # Private Functions
  
  defp encode_payload(payload) do
    try do
      {:ok, :erlang.term_to_binary(payload)}
    rescue
      _ -> {:error, :encoding_failed}
    end
  end
  
  defp decode_payload(binary) do
    try do
      {:ok, :erlang.binary_to_term(binary, [:safe])}
    rescue
      _ -> {:error, :decoding_failed}
    end
  end
  
  defp process_payload(payload, opts) do
    payload
    |> maybe_compress(Keyword.get(opts, :compress, false))
    |> maybe_encrypt(Keyword.get(opts, :encrypt, false), 
                     Keyword.get(opts, :encryption_key))
  end
  
  defp process_incoming_payload(payload, flags, opts) do
    payload
    |> maybe_decrypt(has_flag?(flags, @flag_encrypted), 
                     Keyword.get(opts, :decryption_key))
    |> maybe_decompress(has_flag?(flags, @flag_compressed))
  end
  
  defp maybe_compress({:ok, data}, true) do
    {:ok, :zlib.compress(data)}
  end
  defp maybe_compress(result, _), do: result
  
  defp maybe_compress(data, true) when is_binary(data) do
    :zlib.compress(data)
  end
  defp maybe_compress(data, _), do: data
  
  defp maybe_decompress({:ok, data}, true) do
    try do
      {:ok, :zlib.uncompress(data)}
    rescue
      _ -> {:error, :decompression_failed}
    end
  end
  defp maybe_decompress(result, _), do: result
  
  defp maybe_encrypt({:ok, data}, true, key) when is_binary(key) do
    iv = :crypto.strong_rand_bytes(16)
    encrypted = :crypto.crypto_one_time(:aes_256_cbc, key, iv, data, true)
    {:ok, iv <> encrypted}
  end
  defp maybe_encrypt(result, _, _), do: result
  
  defp maybe_decrypt({:ok, <<iv::binary-16, encrypted::binary>>}, true, key) 
       when is_binary(key) do
    try do
      decrypted = :crypto.crypto_one_time(:aes_256_cbc, key, iv, encrypted, false)
      {:ok, decrypted}
    rescue
      _ -> {:error, :decryption_failed}
    end
  end
  defp maybe_decrypt(result, _, _), do: result
  
  defp build_header(message, payload, opts) do
    type_byte = message_type_to_byte(message.type)
    flags = build_flags(opts)
    length = byte_size(payload)
    
    if length > @max_message_size do
      {:error, :message_too_large}
    else
      header = <<@protocol_version::8, type_byte::8, flags::16, length::32>>
      {:ok, header}
    end
  end
  
  defp build_flags(opts) do
    flags = 0
    flags = if Keyword.get(opts, :compress, false), do: flags ||| @flag_compressed, else: flags
    flags = if Keyword.get(opts, :encrypt, false), do: flags ||| @flag_encrypted, else: flags
    flags = if Keyword.get(opts, :priority, :normal) == :high, do: flags ||| @flag_priority_high, else: flags
    flags = if Keyword.get(opts, :requires_ack, false), do: flags ||| @flag_requires_ack, else: flags
    flags
  end
  
  defp has_flag?(flags, flag), do: (flags &&& flag) != 0
  
  defp message_type_to_byte(:hello), do: @msg_hello
  defp message_type_to_byte(:hello_ack), do: @msg_hello_ack
  defp message_type_to_byte(:request), do: @msg_request
  defp message_type_to_byte(:response), do: @msg_response
  defp message_type_to_byte(:stream_start), do: @msg_stream_start
  defp message_type_to_byte(:stream_data), do: @msg_stream_data
  defp message_type_to_byte(:stream_end), do: @msg_stream_end
  defp message_type_to_byte(:cast), do: @msg_cast
  defp message_type_to_byte(:heartbeat), do: @msg_heartbeat
  defp message_type_to_byte(:heartbeat_ack), do: @msg_heartbeat_ack
  defp message_type_to_byte(:error), do: @msg_error
  
  defp decode_message_type(@msg_hello), do: {:ok, :hello}
  defp decode_message_type(@msg_hello_ack), do: {:ok, :hello_ack}
  defp decode_message_type(@msg_request), do: {:ok, :request}
  defp decode_message_type(@msg_response), do: {:ok, :response}
  defp decode_message_type(@msg_stream_start), do: {:ok, :stream_start}
  defp decode_message_type(@msg_stream_data), do: {:ok, :stream_data}
  defp decode_message_type(@msg_stream_end), do: {:ok, :stream_end}
  defp decode_message_type(@msg_cast), do: {:ok, :cast}
  defp decode_message_type(@msg_heartbeat), do: {:ok, :heartbeat}
  defp decode_message_type(@msg_heartbeat_ack), do: {:ok, :heartbeat_ack}
  defp decode_message_type(@msg_error), do: {:ok, :error}
  defp decode_message_type(_), do: {:error, :unknown_message_type}
  
  defp generate_message_id do
    :crypto.strong_rand_bytes(16)
  end
  
  defp extract_message_id(%{id: id}), do: id
  defp extract_message_id(_), do: nil
  
  defp extract_correlation_id(%{correlation_id: id}), do: id
  defp extract_correlation_id(_), do: nil
  
  defp extract_metadata(%{metadata: metadata}), do: metadata
  defp extract_metadata(_), do: %{}
  
  defp build_fragment(fragment_id, index, total, data) do
    flags = @flag_fragmented
    payload = %{
      fragment_id: fragment_id,
      index: index,
      total: total,
      data: data
    }
    
    {:ok, encoded} = encode_payload(payload)
    type_byte = message_type_to_byte(:stream_data)
    length = byte_size(encoded)
    
    <<@protocol_version::8, type_byte::8, flags::16, length::32, encoded::binary>>
  end
  
  defp extract_fragment_index(fragment) do
    case decode(fragment) do
      {:ok, %{payload: %{index: index}}} -> index
      _ -> -1
    end
  end
  
  defp extract_fragment_data(fragment) do
    case decode(fragment) do
      {:ok, %{payload: %{data: data}}} -> data
      _ -> <<>>
    end
  end
  
  defp valid_fragment_sequence?(fragments) do
    indices = Enum.map(fragments, &extract_fragment_index/1)
    expected = Enum.to_list(0..(length(fragments) - 1))
    indices == expected
  end
end
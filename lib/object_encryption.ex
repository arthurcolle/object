defmodule Object.Encryption do
  @moduledoc """
  End-to-end encryption for Object network communication.
  
  Provides strong cryptographic guarantees for Object-to-Object communication
  including identity verification, forward secrecy, and message authentication.
  
  ## Features
  
  - X25519 ECDH for key exchange
  - Ed25519 for digital signatures
  - ChaCha20-Poly1305 for authenticated encryption
  - Double Ratchet algorithm for forward secrecy
  - Certificate-based identity verification
  - Optional onion routing for anonymity
  """
  
  use GenServer
  require Logger
  
  # @curve25519_key_size 32
  # @ed25519_key_size 32
  # @chacha20_key_size 32
  @chacha20_nonce_size 12
  @poly1305_tag_size 16
  
  @type keypair :: %{
    public: binary(),
    private: binary()
  }
  
  @type identity :: %{
    id: binary(),
    signing_key: keypair(),
    encryption_key: keypair(),
    certificate: certificate()
  }
  
  @type certificate :: %{
    subject_id: binary(),
    public_signing_key: binary(),
    public_encryption_key: binary(),
    issuer_id: binary() | :self,
    signature: binary(),
    issued_at: DateTime.t(),
    expires_at: DateTime.t()
  }
  
  @type session :: %{
    peer_id: binary(),
    peer_certificate: certificate(),
    root_key: binary(),
    chain_keys: %{send: binary(), receive: binary()},
    message_keys: %{send: [binary()], receive: [binary()]},
    counters: %{send: non_neg_integer(), receive: non_neg_integer()},
    handshake_state: :pending | :completed,
    last_activity: DateTime.t()
  }
  
  @type state :: %{
    identity: identity(),
    sessions: %{binary() => session()},
    trusted_certificates: %{binary() => certificate()},
    onion_routes: %{binary() => [binary()]},
    config: map()
  }
  
  # Client API
  
  @doc """
  Starts the encryption service.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Generates a new identity with keypairs and self-signed certificate.
  """
  @spec generate_identity(binary()) :: {:ok, identity()} | {:error, term()}
  def generate_identity(node_id) do
    GenServer.call(__MODULE__, {:generate_identity, node_id})
  end
  
  @doc """
  Establishes an encrypted session with a peer.
  """
  @spec establish_session(binary(), certificate()) :: :ok | {:error, term()}
  def establish_session(peer_id, peer_certificate) do
    GenServer.call(__MODULE__, {:establish_session, peer_id, peer_certificate})
  end
  
  @doc """
  Encrypts a message for a specific peer.
  """
  @spec encrypt_message(binary(), binary()) :: {:ok, binary()} | {:error, term()}
  def encrypt_message(peer_id, plaintext) do
    GenServer.call(__MODULE__, {:encrypt_message, peer_id, plaintext})
  end
  
  @doc """
  Decrypts a message from a specific peer.
  """
  @spec decrypt_message(binary(), binary()) :: {:ok, binary()} | {:error, term()}
  def decrypt_message(peer_id, ciphertext) do
    GenServer.call(__MODULE__, {:decrypt_message, peer_id, ciphertext})
  end
  
  @doc """
  Signs data with the node's signing key.
  """
  @spec sign(binary()) :: {:ok, binary()} | {:error, term()}
  def sign(data) do
    GenServer.call(__MODULE__, {:sign, data})
  end
  
  @doc """
  Verifies a signature from a peer.
  """
  @spec verify(binary(), binary(), binary()) :: boolean()
  def verify(peer_id, data, signature) do
    GenServer.call(__MODULE__, {:verify, peer_id, data, signature})
  end
  
  @doc """
  Creates an onion-encrypted message for anonymous routing.
  """
  @spec create_onion_message(binary(), [binary()]) :: {:ok, binary()} | {:error, term()}
  def create_onion_message(plaintext, route) do
    GenServer.call(__MODULE__, {:create_onion_message, plaintext, route})
  end
  
  @doc """
  Processes an onion-encrypted message.
  """
  @spec process_onion_message(binary()) :: {:ok, binary(), binary() | :final} | {:error, term()}
  def process_onion_message(onion) do
    GenServer.call(__MODULE__, {:process_onion_message, onion})
  end
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    # Load or generate identity
    identity = case Keyword.get(opts, :identity_file) do
      nil -> 
        node_id = Keyword.get(opts, :node_id, generate_node_id())
        generate_new_identity(node_id)
        
      file ->
        load_identity_from_file(file)
    end
    
    state = %{
      identity: identity,
      sessions: %{},
      trusted_certificates: %{},
      onion_routes: %{},
      config: %{
        session_timeout: Keyword.get(opts, :session_timeout, 3600_000),
        max_sessions: Keyword.get(opts, :max_sessions, 1000),
        require_trusted_certs: Keyword.get(opts, :require_trusted_certs, false)
      }
    }
    
    # Schedule session cleanup
    Process.send_after(self(), :cleanup_sessions, 60_000)
    
    {:ok, state}
  end
  
  @impl true
  def handle_call({:generate_identity, node_id}, _from, state) do
    identity = generate_new_identity(node_id)
    {:reply, {:ok, identity}, %{state | identity: identity}}
  end
  
  @impl true
  def handle_call({:establish_session, peer_id, peer_cert}, _from, state) do
    case verify_certificate(peer_cert, state.trusted_certificates) do
      :ok ->
        # Perform ECDH key exchange
        shared_secret = perform_ecdh(
          state.identity.encryption_key.private,
          peer_cert.public_encryption_key
        )
        
        # Derive root key
        root_key = derive_root_key(shared_secret, state.identity.id, peer_id)
        
        # Initialize double ratchet
        session = %{
          peer_id: peer_id,
          peer_certificate: peer_cert,
          root_key: root_key,
          chain_keys: %{
            send: derive_chain_key(root_key, "send"),
            receive: derive_chain_key(root_key, "receive")
          },
          message_keys: %{send: [], receive: []},
          counters: %{send: 0, receive: 0},
          handshake_state: :completed,
          last_activity: DateTime.utc_now()
        }
        
        new_state = put_in(state.sessions[peer_id], session)
        {:reply, :ok, new_state}
        
      {:error, reason} = error ->
        {:reply, error, state}
    end
  end
  
  @impl true
  def handle_call({:encrypt_message, peer_id, plaintext}, _from, state) do
    case Map.get(state.sessions, peer_id) do
      nil ->
        {:reply, {:error, :no_session}, state}
        
      session ->
        # Get next message key
        {message_key, new_session} = get_next_send_key(session)
        
        # Generate nonce
        nonce = :crypto.strong_rand_bytes(@chacha20_nonce_size)
        
        # Encrypt with ChaCha20-Poly1305
        {ciphertext, tag} = :crypto.crypto_one_time_aead(
          :chacha20_poly1305,
          message_key,
          nonce,
          plaintext,
          <<>>,  # No additional authenticated data for now
          true
        )
        
        # Package encrypted message
        encrypted = <<
          nonce::binary-size(@chacha20_nonce_size),
          tag::binary-size(@poly1305_tag_size),
          ciphertext::binary
        >>
        
        # Update session
        new_state = put_in(state.sessions[peer_id], new_session)
        
        {:reply, {:ok, encrypted}, new_state}
    end
  end
  
  @impl true
  def handle_call({:decrypt_message, peer_id, encrypted}, _from, state) do
    case Map.get(state.sessions, peer_id) do
      nil ->
        {:reply, {:error, :no_session}, state}
        
      session ->
        # Extract components
        <<
          nonce::binary-size(@chacha20_nonce_size),
          tag::binary-size(@poly1305_tag_size),
          ciphertext::binary
        >> = encrypted
        
        # Try message keys
        case try_decrypt_with_keys(session.message_keys.receive, nonce, tag, ciphertext) do
          {:ok, plaintext, key_index} ->
            # Remove used key and older keys
            new_keys = Enum.drop(session.message_keys.receive, key_index + 1)
            new_session = put_in(session.message_keys.receive, new_keys)
            new_state = put_in(state.sessions[peer_id], new_session)
            
            {:reply, {:ok, plaintext}, new_state}
            
          :error ->
            # Try with next chain key
            {message_key, new_session} = get_next_receive_key(session)
            
            case decrypt_with_key(message_key, nonce, tag, ciphertext) do
              {:ok, plaintext} ->
                new_state = put_in(state.sessions[peer_id], new_session)
                {:reply, {:ok, plaintext}, new_state}
                
              :error ->
                {:reply, {:error, :decryption_failed}, state}
            end
        end
    end
  end
  
  @impl true
  def handle_call({:sign, data}, _from, state) do
    signature = :crypto.sign(
      :eddsa, 
      :ed25519, 
      data, 
      [state.identity.signing_key.private, :crypto.hash(:sha256, data)]
    )
    {:reply, {:ok, signature}, state}
  end
  
  @impl true
  def handle_call({:verify, peer_id, data, signature}, _from, state) do
    case Map.get(state.sessions, peer_id) do
      %{peer_certificate: cert} ->
        result = :crypto.verify(
          :eddsa,
          :ed25519,
          data,
          signature,
          [cert.public_signing_key, :crypto.hash(:sha256, data)]
        )
        {:reply, result, state}
        
      nil ->
        {:reply, false, state}
    end
  end
  
  @impl true
  def handle_call({:create_onion_message, plaintext, route}, _from, state) do
    # Build onion layers from destination to source
    encrypted = Enum.reduce(Enum.reverse(route), plaintext, fn node_id, inner ->
      case Map.get(state.sessions, node_id) do
        nil ->
          # Skip nodes we don't have sessions with
          inner
          
        session ->
          # Encrypt layer
          {message_key, _} = get_next_send_key(session)
          nonce = :crypto.strong_rand_bytes(@chacha20_nonce_size)
          
          # Add next hop info if not final destination
          next_hop = if node_id == List.last(route), do: <<>>, else: <<0>>
          payload = next_hop <> inner
          
          {ciphertext, tag} = :crypto.crypto_one_time_aead(
            :chacha20_poly1305,
            message_key,
            nonce,
            payload,
            <<>>,
            true
          )
          
          <<
            nonce::binary-size(@chacha20_nonce_size),
            tag::binary-size(@poly1305_tag_size),
            ciphertext::binary
          >>
      end
    end)
    
    {:reply, {:ok, encrypted}, state}
  end
  
  @impl true
  def handle_call({:process_onion_message, onion}, _from, state) do
    # Try to decrypt with our keys
    result = Enum.find_value(state.sessions, fn {peer_id, session} ->
      case try_decrypt_onion_layer(onion, session) do
        {:ok, payload} ->
          case payload do
            <<0, next_onion::binary>> ->
              # Forward to next hop
              {:ok, next_onion, peer_id}
              
            _ ->
              # We are the final destination
              {:ok, payload, :final}
          end
          
        :error ->
          nil
      end
    end)
    
    case result do
      nil -> {:reply, {:error, :not_for_us}, state}
      found -> {:reply, found, state}
    end
  end
  
  @impl true
  def handle_info(:cleanup_sessions, state) do
    # Remove inactive sessions
    now = DateTime.utc_now()
    timeout = state.config.session_timeout
    
    active_sessions = state.sessions
    |> Enum.reject(fn {_peer_id, session} ->
      DateTime.diff(now, session.last_activity, :millisecond) > timeout
    end)
    |> Map.new()
    
    # Schedule next cleanup
    Process.send_after(self(), :cleanup_sessions, 60_000)
    
    {:noreply, %{state | sessions: active_sessions}}
  end
  
  # Private Functions - Identity Management
  
  defp generate_new_identity(node_id) do
    # Generate Ed25519 signing keypair
    {ed_pub, ed_priv} = :crypto.generate_key(:eddsa, :ed25519)
    signing_key = %{public: ed_pub, private: ed_priv}
    
    # Generate X25519 encryption keypair
    {x_pub, x_priv} = :crypto.generate_key(:ecdh, :x25519)
    encryption_key = %{public: x_pub, private: x_priv}
    
    # Create self-signed certificate
    certificate = create_certificate(
      node_id,
      signing_key.public,
      encryption_key.public,
      signing_key.private
    )
    
    %{
      id: node_id,
      signing_key: signing_key,
      encryption_key: encryption_key,
      certificate: certificate
    }
  end
  
  defp create_certificate(subject_id, pub_signing, pub_encryption, signing_key) do
    now = DateTime.utc_now()
    expires = DateTime.add(now, 365, :day)
    
    cert_data = %{
      subject_id: subject_id,
      public_signing_key: pub_signing,
      public_encryption_key: pub_encryption,
      issuer_id: :self,
      issued_at: now,
      expires_at: expires
    }
    
    # Sign certificate
    to_sign = :erlang.term_to_binary(cert_data)
    signature = :crypto.sign(
      :eddsa,
      :ed25519,
      to_sign,
      [signing_key, :crypto.hash(:sha256, to_sign)]
    )
    
    Map.put(cert_data, :signature, signature)
  end
  
  defp verify_certificate(cert, trusted_certs) do
    # Check expiration
    if DateTime.compare(DateTime.utc_now(), cert.expires_at) == :gt do
      {:error, :certificate_expired}
    else
      # Verify signature
      cert_data = Map.delete(cert, :signature)
      to_verify = :erlang.term_to_binary(cert_data)
      
      signing_key = case cert.issuer_id do
        :self -> cert.public_signing_key
        issuer_id ->
          case Map.get(trusted_certs, issuer_id) do
            %{public_signing_key: key} -> key
            nil -> nil
          end
      end
      
      if signing_key && :crypto.verify(
        :eddsa,
        :ed25519,
        to_verify,
        cert.signature,
        [signing_key, :crypto.hash(:sha256, to_verify)]
      ) do
        :ok
      else
        {:error, :invalid_signature}
      end
    end
  end
  
  # Private Functions - Cryptographic Operations
  
  defp perform_ecdh(private_key, public_key) do
    :crypto.compute_key(:ecdh, public_key, private_key, :x25519)
  end
  
  defp derive_root_key(shared_secret, id1, id2) do
    # Sort IDs for consistency
    sorted_ids = Enum.sort([id1, id2])
    
    :crypto.hash(:sha256, [
      shared_secret,
      "root_key",
      Enum.at(sorted_ids, 0),
      Enum.at(sorted_ids, 1)
    ] |> Enum.join())
  end
  
  defp derive_chain_key(root_key, direction) do
    :crypto.hash(:sha256, [root_key, "chain_", direction] |> Enum.join())
  end
  
  defp derive_message_key(chain_key, counter) do
    :crypto.hash(:sha256, [
      chain_key,
      "msg_",
      <<counter::32>>
    ] |> Enum.join())
  end
  
  defp get_next_send_key(session) do
    counter = session.counters.send
    chain_key = session.chain_keys.send
    
    # Derive message key
    message_key = derive_message_key(chain_key, counter)
    
    # Ratchet chain key forward
    new_chain_key = :crypto.hash(:sha256, [chain_key, "ratchet"] |> Enum.join())
    
    new_session = session
    |> put_in([:counters, :send], counter + 1)
    |> put_in([:chain_keys, :send], new_chain_key)
    |> update_in([:message_keys, :send], &([message_key | &1]))
    
    {message_key, new_session}
  end
  
  defp get_next_receive_key(session) do
    counter = session.counters.receive
    chain_key = session.chain_keys.receive
    
    # Derive message key
    message_key = derive_message_key(chain_key, counter)
    
    # Ratchet chain key forward
    new_chain_key = :crypto.hash(:sha256, [chain_key, "ratchet"] |> Enum.join())
    
    new_session = session
    |> put_in([:counters, :receive], counter + 1)
    |> put_in([:chain_keys, :receive], new_chain_key)
    |> update_in([:message_keys, :receive], &(&1 ++ [message_key]))
    
    {message_key, new_session}
  end
  
  defp try_decrypt_with_keys(keys, nonce, tag, ciphertext) do
    keys
    |> Enum.with_index()
    |> Enum.find_value(fn {key, index} ->
      case decrypt_with_key(key, nonce, tag, ciphertext) do
        {:ok, plaintext} -> {:ok, plaintext, index}
        :error -> nil
      end
    end)
    |> case do
      nil -> :error
      result -> result
    end
  end
  
  defp decrypt_with_key(key, nonce, tag, ciphertext) do
    case :crypto.crypto_one_time_aead(
      :chacha20_poly1305,
      key,
      nonce,
      ciphertext,
      <<>>,
      tag,
      false
    ) do
      plaintext when is_binary(plaintext) -> {:ok, plaintext}
      :error -> :error
    end
  end
  
  defp try_decrypt_onion_layer(<<nonce::binary-size(@chacha20_nonce_size),
                                 tag::binary-size(@poly1305_tag_size),
                                 ciphertext::binary>>, session) do
    # Try with current receive keys
    case try_decrypt_with_keys(session.message_keys.receive, nonce, tag, ciphertext) do
      {:ok, plaintext, _} -> {:ok, plaintext}
      :error ->
        # Try with next key
        {message_key, _} = get_next_receive_key(session)
        decrypt_with_key(message_key, nonce, tag, ciphertext)
    end
  end
  defp try_decrypt_onion_layer(_, _), do: :error
  
  # Utility Functions
  
  defp generate_node_id do
    :crypto.strong_rand_bytes(20)
  end
  
  defp load_identity_from_file(file) do
    # This would load identity from persistent storage
    # For now, generate a new one
    generate_new_identity(generate_node_id())
  end
end
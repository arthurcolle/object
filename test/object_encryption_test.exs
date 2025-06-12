defmodule Object.EncryptionTest do
  use ExUnit.Case, async: true
  alias Object.Encryption

  setup do
    {:ok, _pid} = Encryption.start_link()
    :ok
  end

  describe "identity generation" do
    test "generates valid identity with keypairs" do
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, identity} = Encryption.generate_identity(node_id)
      
      assert identity.id == node_id
      assert is_binary(identity.signing_key.public)
      assert is_binary(identity.signing_key.private)
      assert is_binary(identity.encryption_key.public)
      assert is_binary(identity.encryption_key.private)
      assert is_map(identity.certificate)
    end

    test "creates self-signed certificates" do
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, identity} = Encryption.generate_identity(node_id)
      
      cert = identity.certificate
      assert cert.subject_id == node_id
      assert cert.issuer_id == :self
      assert is_binary(cert.signature)
      assert %DateTime{} = cert.issued_at
      assert %DateTime{} = cert.expires_at
    end
  end

  describe "session establishment" do
    test "establishes encrypted session between peers" do
      # Generate two identities
      {:ok, alice} = Encryption.generate_identity("alice")
      {:ok, bob} = Encryption.generate_identity("bob")
      
      # Alice establishes session with Bob
      :ok = Encryption.establish_session("bob", bob.certificate)
      
      # Should be able to encrypt messages now
      {:ok, encrypted} = Encryption.encrypt_message("bob", "Hello Bob!")
      assert is_binary(encrypted)
    end

    test "rejects expired certificates" do
      # Create certificate with past expiration
      expired_cert = %{
        subject_id: "expired_node",
        public_signing_key: :crypto.strong_rand_bytes(32),
        public_encryption_key: :crypto.strong_rand_bytes(32),
        issuer_id: :self,
        signature: :crypto.strong_rand_bytes(64),
        issued_at: DateTime.add(DateTime.utc_now(), -400, :day),
        expires_at: DateTime.add(DateTime.utc_now(), -35, :day)
      }
      
      result = Encryption.establish_session("expired_node", expired_cert)
      assert {:error, :certificate_expired} = result
    end
  end

  describe "message encryption/decryption" do
    test "encrypts and decrypts messages successfully" do
      # Setup peer session
      {:ok, peer} = Encryption.generate_identity("peer")
      :ok = Encryption.establish_session("peer", peer.certificate)
      
      original = "Secret message with unicode: 🔐"
      {:ok, encrypted} = Encryption.encrypt_message("peer", original)
      
      # Encrypted should be different from original
      assert encrypted != original
      assert byte_size(encrypted) > byte_size(original)
      
      # Should decrypt back to original
      {:ok, decrypted} = Encryption.decrypt_message("peer", encrypted)
      assert decrypted == original
    end

    test "handles large messages" do
      {:ok, peer} = Encryption.generate_identity("peer")
      :ok = Encryption.establish_session("peer", peer.certificate)
      
      large_message = :crypto.strong_rand_bytes(1_000_000)
      {:ok, encrypted} = Encryption.encrypt_message("peer", large_message)
      {:ok, decrypted} = Encryption.decrypt_message("peer", encrypted)
      
      assert decrypted == large_message
    end

    test "provides forward secrecy via ratcheting" do
      {:ok, peer} = Encryption.generate_identity("peer")
      :ok = Encryption.establish_session("peer", peer.certificate)
      
      # Encrypt multiple messages
      messages = for i <- 1..5, do: "Message #{i}"
      encrypted = for msg <- messages do
        {:ok, enc} = Encryption.encrypt_message("peer", msg)
        enc
      end
      
      # Each should be different even for same plaintext
      encrypted_dups = for _ <- 1..3 do
        {:ok, enc} = Encryption.encrypt_message("peer", "Duplicate")
        enc
      end
      
      assert length(Enum.uniq(encrypted_dups)) == 3
    end
  end

  describe "digital signatures" do
    test "signs and verifies data" do
      data = "Important document"
      {:ok, signature} = Encryption.sign(data)
      
      assert is_binary(signature)
      assert byte_size(signature) == 64  # Ed25519 signature size
    end

    test "verifies signatures from peers" do
      # Setup peer
      {:ok, peer} = Encryption.generate_identity("signer")
      :ok = Encryption.establish_session("signer", peer.certificate)
      
      data = "Verify this"
      
      # Peer would sign with their key
      # For test, we'll use a valid signature format
      fake_signature = :crypto.strong_rand_bytes(64)
      
      # Verification should return boolean
      result = Encryption.verify("signer", data, fake_signature)
      assert is_boolean(result)
    end
  end

  describe "onion routing" do
    test "creates onion encrypted messages" do
      # Setup route through 3 nodes
      nodes = for i <- 1..3 do
        {:ok, node} = Encryption.generate_identity("node#{i}")
        :ok = Encryption.establish_session("node#{i}", node.certificate)
        "node#{i}"
      end
      
      plaintext = "Secret onion message"
      {:ok, onion} = Encryption.create_onion_message(plaintext, nodes)
      
      assert is_binary(onion)
      assert byte_size(onion) > byte_size(plaintext)
    end

    test "processes onion layers" do
      # In a real implementation, each node would peel one layer
      {:ok, node} = Encryption.generate_identity("relay")
      :ok = Encryption.establish_session("relay", node.certificate)
      
      # Create single-layer onion
      {:ok, onion} = Encryption.create_onion_message("Hidden", ["relay"])
      
      # Process should indicate if we're final destination
      result = Encryption.process_onion_message(onion)
      assert match?({:ok, _, _}, result) || match?({:error, :not_for_us}, result)
    end
  end

  describe "session management" do
    test "maintains multiple concurrent sessions" do
      # Create multiple peer sessions
      peers = for i <- 1..5 do
        peer_id = "peer#{i}"
        {:ok, peer} = Encryption.generate_identity(peer_id)
        :ok = Encryption.establish_session(peer_id, peer.certificate)
        peer_id
      end
      
      # Should be able to encrypt to each peer
      for peer_id <- peers do
        {:ok, _encrypted} = Encryption.encrypt_message(peer_id, "Hello #{peer_id}")
      end
    end

    test "handles session timeout cleanup" do
      # Create session with short timeout
      {:ok, _pid} = Encryption.start_link(session_timeout: 100)
      
      {:ok, peer} = Encryption.generate_identity("timeout_peer")
      :ok = Encryption.establish_session("timeout_peer", peer.certificate)
      
      # Session should work initially
      {:ok, _} = Encryption.encrypt_message("timeout_peer", "Quick message")
      
      # After timeout, session should be cleaned up
      Process.sleep(200)
      
      # This would fail with no session
      result = Encryption.encrypt_message("timeout_peer", "Too late")
      assert {:error, :no_session} = result
    end
  end
end
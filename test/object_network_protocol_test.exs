defmodule Object.NetworkProtocolTest do
  use ExUnit.Case, async: true
  alias Object.NetworkProtocol

  describe "message encoding/decoding" do
    test "encodes and decodes request messages" do
      message = NetworkProtocol.create_request("obj123", "increment", [1, 2, 3])
      
      {:ok, encoded} = NetworkProtocol.encode(message)
      assert is_binary(encoded)
      
      {:ok, decoded} = NetworkProtocol.decode(encoded)
      assert decoded.type == :request
      assert decoded.payload.object_id == "obj123"
      assert decoded.payload.method == "increment"
      assert decoded.payload.args == [1, 2, 3]
    end

    test "encodes and decodes response messages" do
      correlation_id = :crypto.strong_rand_bytes(16)
      message = NetworkProtocol.create_response(correlation_id, {:ok, 42})
      
      {:ok, encoded} = NetworkProtocol.encode(message)
      {:ok, decoded} = NetworkProtocol.decode(encoded)
      
      assert decoded.type == :response
      assert decoded.correlation_id == correlation_id
      assert decoded.payload.result == {:ok, 42}
    end

    test "handles compression" do
      large_data = String.duplicate("Hello World! ", 1000)
      message = NetworkProtocol.create_cast("obj123", "process", [large_data])
      
      {:ok, uncompressed} = NetworkProtocol.encode(message, compress: false)
      {:ok, compressed} = NetworkProtocol.encode(message, compress: true)
      
      # Compressed should be smaller
      assert byte_size(compressed) < byte_size(uncompressed)
      
      # Both should decode to same message
      {:ok, decoded1} = NetworkProtocol.decode(uncompressed)
      {:ok, decoded2} = NetworkProtocol.decode(compressed)
      
      assert decoded1.payload == decoded2.payload
    end

    test "handles encryption" do
      key = :crypto.strong_rand_bytes(32)
      message = NetworkProtocol.create_cast("obj123", "secret", ["classified"])
      
      {:ok, encrypted} = NetworkProtocol.encode(message, 
        encrypt: true, 
        encryption_key: key
      )
      
      # Should fail without key
      {:error, _} = NetworkProtocol.decode(encrypted)
      
      # Should succeed with key
      {:ok, decrypted} = NetworkProtocol.decode(encrypted, decryption_key: key)
      assert decrypted.payload.args == ["classified"]
    end
  end

  describe "message types" do
    test "creates heartbeat messages" do
      heartbeat = NetworkProtocol.create_heartbeat()
      assert heartbeat.type == :heartbeat
      assert is_integer(heartbeat.payload.timestamp)
    end

    test "creates heartbeat acknowledgments" do
      correlation_id = :crypto.strong_rand_bytes(16)
      ack = NetworkProtocol.create_heartbeat_ack(correlation_id)
      
      assert ack.type == :heartbeat_ack
      assert ack.correlation_id == correlation_id
    end
  end

  describe "message fragmentation" do
    test "fragments and reassembles large messages" do
      # Create large data that exceeds chunk size
      large_data = :crypto.strong_rand_bytes(200_000)
      
      fragments = NetworkProtocol.fragment_message(large_data, 65536)
      assert length(fragments) > 1
      
      # Each fragment should be valid
      for fragment <- fragments do
        assert byte_size(fragment) <= 65536 + 100  # Some overhead
      end
      
      # Reassemble
      {:ok, reassembled} = NetworkProtocol.reassemble_fragments(fragments)
      assert reassembled == large_data
    end

    test "detects invalid fragment sequences" do
      large_data = :crypto.strong_rand_bytes(200_000)
      fragments = NetworkProtocol.fragment_message(large_data, 65536)
      
      # Drop a fragment
      incomplete = Enum.drop(fragments, 1)
      
      result = NetworkProtocol.reassemble_fragments(incomplete)
      assert {:error, :invalid_fragment_sequence} = result
    end
  end

  describe "protocol versioning" do
    test "includes version in encoded messages" do
      message = NetworkProtocol.create_cast("obj123", "test", [])
      {:ok, encoded} = NetworkProtocol.encode(message)
      
      # Check version byte
      <<version::8, _::binary>> = encoded
      assert version == 1
    end

    test "rejects messages with invalid size" do
      # Create a message that claims to be huge
      invalid = <<1::8, 0x20::8, 0::16, 0xFFFFFFFF::32, "small">>
      result = NetworkProtocol.decode(invalid)
      assert {:error, :message_too_large} = result
    end
  end
end
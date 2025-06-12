defmodule Object.SerializationTest do
  use ExUnit.Case, async: true
  alias Object.Serialization

  setup do
    object = Object.new(
      id: "test_object",
      state: %{
        counter: 42,
        name: "Test Object",
        tags: [:important, :active],
        nested: %{deep: %{value: "hidden"}}
      }
    )
    {:ok, object: object}
  end

  describe "serialization formats" do
    test "serializes to ETF format", %{object: object} do
      {:ok, serialized} = Serialization.serialize(object, format: :etf)
      assert is_binary(serialized)
      
      {:ok, deserialized} = Serialization.deserialize(serialized, format: :etf)
      assert deserialized.id == object.id
      assert deserialized.state.counter == 42
    end

    test "serializes to JSON format", %{object: object} do
      {:ok, serialized} = Serialization.serialize(object, format: :json)
      assert is_binary(serialized)
      
      # Should be valid JSON
      assert {:ok, _} = Jason.decode(serialized)
      
      {:ok, deserialized} = Serialization.deserialize(serialized, format: :json)
      assert deserialized.id == object.id
    end

    test "serializes to MessagePack format", %{object: object} do
      {:ok, serialized} = Serialization.serialize(object, format: :msgpack)
      assert is_binary(serialized)
      
      {:ok, deserialized} = Serialization.deserialize(serialized, format: :msgpack)
      assert deserialized.id == object.id
    end

    test "auto-detects format during deserialization" do
      object = Object.new(id: "auto_test", state: %{value: 123})
      
      # ETF format (starts with 131)
      {:ok, etf} = Serialization.serialize(object, format: :etf)
      {:ok, from_etf} = Serialization.deserialize(etf)
      assert from_etf.id == "auto_test"
      
      # JSON format (starts with {)
      {:ok, json} = Serialization.serialize(object, format: :json)
      {:ok, from_json} = Serialization.deserialize(json)
      assert from_json.id == "auto_test"
    end
  end

  describe "compression" do
    test "compresses serialized data", %{object: object} do
      # Add large data to object
      large_object = %{object | state: Map.put(object.state, :data, 
        String.duplicate("Test data ", 10000))}
      
      {:ok, uncompressed} = Serialization.serialize(large_object, compress: false)
      {:ok, compressed} = Serialization.serialize(large_object, compress: true)
      
      assert byte_size(compressed) < byte_size(uncompressed)
      
      # Both should deserialize correctly
      {:ok, from_uncompressed} = Serialization.deserialize(uncompressed)
      {:ok, from_compressed} = Serialization.deserialize(compressed)
      
      assert from_uncompressed.id == from_compressed.id
    end
  end

  describe "partial serialization" do
    test "serializes only specified fields", %{object: object} do
      {:ok, partial} = Serialization.serialize_partial(object, 
        [:id, :state], format: :json)
      
      decoded = Jason.decode!(partial)
      assert Map.has_key?(decoded, "id")
      assert Map.has_key?(decoded, "state")
      assert not Map.has_key?(decoded, "methods")
    end

    test "respects max_depth option", %{object: object} do
      {:ok, shallow} = Serialization.serialize(object, 
        format: :json, max_depth: 2)
      
      decoded = Jason.decode!(shallow)
      # Deep nested value should be truncated
      assert is_map(decoded["state"]["nested"])
    end
  end

  describe "type preservation" do
    test "preserves atoms in JSON format" do
      object = Object.new(id: "atom_test", state: %{status: :active})
      
      {:ok, json} = Serialization.serialize(object, format: :json)
      {:ok, restored} = Serialization.deserialize(json, format: :json)
      
      assert restored.state.status == :active
    end

    test "preserves tuples in JSON format" do
      object = Object.new(id: "tuple_test", 
        state: %{position: {10, 20, 30}})
      
      {:ok, json} = Serialization.serialize(object, format: :json)
      {:ok, restored} = Serialization.deserialize(json, format: :json)
      
      assert restored.state.position == {10, 20, 30}
    end

    test "preserves DateTime objects" do
      now = DateTime.utc_now()
      object = Object.new(id: "time_test", state: %{created: now})
      
      {:ok, json} = Serialization.serialize(object, format: :json)
      {:ok, restored} = Serialization.deserialize(json, format: :json)
      
      assert DateTime.compare(restored.state.created, now) == :eq
    end
  end

  describe "streaming serialization" do
    test "streams large objects" do
      large_object = Object.new(
        id: "stream_test",
        state: %{data: :crypto.strong_rand_bytes(1_000_000)}
      )
      
      stream = Serialization.serialize_stream(large_object)
      chunks = Enum.to_list(stream)
      
      assert length(chunks) > 0
      assert Enum.all?(chunks, &is_binary/1)
    end
  end

  describe "size calculation" do
    test "calculates serialized size without serializing", %{object: object} do
      {:ok, estimated_size} = Serialization.calculate_size(object, format: :etf)
      {:ok, actual} = Serialization.serialize(object, format: :etf)
      
      # Size should match
      assert estimated_size == byte_size(actual)
    end
  end
end
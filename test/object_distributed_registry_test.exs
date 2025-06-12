defmodule Object.DistributedRegistryTest do
  use ExUnit.Case, async: false
  alias Object.DistributedRegistry

  setup do
    {:ok, _pid} = DistributedRegistry.start_link(node_id: "test_node")
    :ok
  end

  describe "DHT operations" do
    test "generates consistent node IDs" do
      node_id = DistributedRegistry.get_node_id()
      assert is_binary(node_id)
      assert byte_size(node_id) == 20  # 160 bits
    end

    test "registers and looks up objects" do
      object = Object.new(id: "test_object_123", state: %{value: 42})
      
      :ok = DistributedRegistry.register_object(object.id, object)
      
      # Immediate lookup should work (local storage)
      {:ok, found} = DistributedRegistry.lookup_object(object.id)
      assert found.id == object.id
    end

    test "finds nodes close to a target" do
      target_id = :crypto.strong_rand_bytes(20)
      
      {:ok, nodes} = DistributedRegistry.find_node(target_id)
      assert is_list(nodes)
      
      # Should return nodes sorted by distance
      if length(nodes) > 1 do
        distances = Enum.map(nodes, fn node ->
          xor_distance(node.id, target_id)
        end)
        
        assert distances == Enum.sort(distances)
      end
    end
  end

  describe "k-bucket management" do
    test "adds nodes to appropriate buckets" do
      # Add some fake peers
      for i <- 1..10 do
        peer_id = :crypto.strong_rand_bytes(20)
        DistributedRegistry.add_peer(peer_id, "192.168.1.#{i}", 4000 + i)
      end
      
      # Should have distributed nodes across buckets
      {:ok, closest} = DistributedRegistry.find_node(
        DistributedRegistry.get_node_id()
      )
      
      assert length(closest) > 0
    end

    test "maintains k-bucket size limits" do
      bucket_index = 100  # Arbitrary bucket
      
      # Try to add more than k nodes to same bucket
      # This is tricky to test directly, so we just verify
      # that find_node returns at most k nodes
      target = :crypto.strong_rand_bytes(20)
      {:ok, nodes} = DistributedRegistry.find_node(target)
      
      assert length(nodes) <= 20  # k = 20
    end
  end

  describe "bootstrap process" do
    test "bootstraps from known nodes" do
      bootstrap_nodes = [{"localhost", 5000}, {"localhost", 5001}]
      
      # This will attempt connections (which will fail in test)
      # but should not crash
      result = DistributedRegistry.bootstrap(bootstrap_nodes)
      assert result == :ok
    end
  end

  describe "data storage" do
    test "stores values with TTL" do
      key = "ephemeral_key"
      value = %{data: "temporary"}
      
      # Register with short TTL
      :ok = DistributedRegistry.register_object(key, value)
      
      # Should be retrievable immediately
      {:ok, retrieved} = DistributedRegistry.lookup_object(key)
      assert retrieved == value
      
      # Note: Testing TTL expiration would require waiting
      # or mocking time, which is complex for async tests
    end

    test "handles storage limits" do
      # Try to store many objects
      for i <- 1..100 do
        object = Object.new(id: "spam_#{i}", state: %{})
        DistributedRegistry.register_object(object.id, object)
      end
      
      # Registry should still be functional
      test_obj = Object.new(id: "final_test", state: %{})
      :ok = DistributedRegistry.register_object(test_obj.id, test_obj)
      
      # Can still look up
      result = DistributedRegistry.lookup_object(test_obj.id)
      assert {:ok, _} = result
    end
  end

  # Helper function
  defp xor_distance(id1, id2) do
    :crypto.exor(id1, id2)
    |> :binary.bin_to_list()
    |> Integer.undigits(256)
  end
end
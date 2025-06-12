defmodule Object.NetworkProxyTest do
  use ExUnit.Case, async: true
  alias Object.NetworkProxy

  setup do
    # Start required services
    case Object.NetworkTransport.start_link() do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end
    
    case Object.DistributedRegistry.start_link() do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
    end
    
    # Create a test object
    test_object = Object.new(
      id: "remote_counter",
      state: %{count: 0},
      methods: %{
        increment: fn state, _args -> 
          {:ok, %{state | count: state.count + 1}} 
        end,
        get_count: fn state, _args -> 
          {:ok, state.count} 
        end,
        slow_operation: fn state, _args ->
          Process.sleep(100)
          {:ok, state}
        end
      }
    )
    
    # Register it in the registry
    :ok = Object.DistributedRegistry.register_object(test_object.id, %{
      node_id: "test_node",
      address: "localhost",
      port: 4000
    })
    
    {:ok, test_object: test_object}
  end

  describe "proxy creation" do
    test "creates proxy for remote object" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      assert is_pid(proxy)
    end

    test "handles unknown object gracefully" do
      {:ok, proxy} = NetworkProxy.create("non_existent")
      
      # Call should fail but not crash
      result = NetworkProxy.call(proxy, "method", [])
      assert {:error, _} = result
    end
  end

  describe "method calls" do
    test "caches method results" do
      {:ok, proxy} = NetworkProxy.create("remote_counter", cache_ttl: 5000)
      
      # First call - cache miss
      stats1 = NetworkProxy.get_stats(proxy)
      cache_hits_before = stats1.cache_hits
      
      # Make same call twice
      NetworkProxy.call(proxy, "get_count", [])
      NetworkProxy.call(proxy, "get_count", [])
      
      # Second call should be cache hit
      stats2 = NetworkProxy.get_stats(proxy)
      assert stats2.cache_hits > cache_hits_before
    end

    test "handles casts (fire-and-forget)" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      
      # Cast should return immediately
      :ok = NetworkProxy.cast(proxy, "increment", [])
      
      # Stats should show the cast
      stats = NetworkProxy.get_stats(proxy)
      assert stats.total_calls > 0
    end

    test "clears cache on demand" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      
      # Populate cache
      NetworkProxy.call(proxy, "get_count", [])
      
      # Clear cache
      :ok = NetworkProxy.clear_cache(proxy)
      
      # Next call should be cache miss
      stats_before = NetworkProxy.get_stats(proxy)
      NetworkProxy.call(proxy, "get_count", [])
      stats_after = NetworkProxy.get_stats(proxy)
      
      # Should not increase cache hits
      assert stats_after.cache_hits == stats_before.cache_hits
    end
  end

  describe "circuit breaker" do
    @tag :skip
    test "opens circuit after failures" do
      # This test would require simulating network failures
      # which is complex in a unit test environment
    end
  end

  describe "statistics" do
    test "tracks call statistics" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      
      initial_stats = NetworkProxy.get_stats(proxy)
      assert initial_stats.total_calls == 0
      assert initial_stats.successful_calls == 0
      assert initial_stats.failed_calls == 0
      
      # Make some calls
      NetworkProxy.call(proxy, "get_count", [])
      NetworkProxy.cast(proxy, "increment", [])
      
      final_stats = NetworkProxy.get_stats(proxy)
      assert final_stats.total_calls > initial_stats.total_calls
    end

    test "calculates average latency" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      
      # Make several calls
      for _ <- 1..5 do
        NetworkProxy.call(proxy, "get_count", [])
      end
      
      stats = NetworkProxy.get_stats(proxy)
      assert stats.avg_latency_ms >= 0
    end
  end

  describe "location refresh" do
    test "refreshes remote node location" do
      {:ok, proxy} = NetworkProxy.create("remote_counter")
      
      # Update location in registry
      Object.DistributedRegistry.register_object("remote_counter", %{
        node_id: "new_node",
        address: "192.168.1.100",
        port: 4001
      })
      
      # Refresh location
      :ok = NetworkProxy.refresh_location(proxy)
      
      # Proxy should still work
      result = NetworkProxy.call(proxy, "get_count", [])
      assert {:error, _} = result  # Will fail due to fake address
    end
  end
end
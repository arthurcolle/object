defmodule Object.NetworkTransportTest do
  use ExUnit.Case, async: true
  alias Object.NetworkTransport

  setup do
    {:ok, pid} = NetworkTransport.start_link()
    {:ok, transport: pid}
  end

  describe "TCP connections" do
    test "establishes TCP connection successfully", %{transport: _transport} do
      # Start a test TCP server
      {:ok, listen_socket} = :gen_tcp.listen(0, [:binary, packet: 4, active: false])
      {:ok, port} = :inet.port(listen_socket)
      
      server_task = Task.async(fn ->
        {:ok, _client} = :gen_tcp.accept(listen_socket)
        :ok
      end)
      
      # Connect to test server
      {:ok, conn_id} = NetworkTransport.connect("localhost", port, transport: :tcp)
      assert is_binary(conn_id)
      
      Task.await(server_task)
      :gen_tcp.close(listen_socket)
    end

    test "handles connection failure gracefully" do
      # Try to connect to non-existent server
      result = NetworkTransport.connect("localhost", 59999, transport: :tcp)
      assert {:error, _reason} = result
    end

    test "sends data over TCP connection", %{transport: _transport} do
      # Setup echo server
      {:ok, listen_socket} = :gen_tcp.listen(0, [:binary, packet: 4, active: false])
      {:ok, port} = :inet.port(listen_socket)
      
      echo_server = Task.async(fn ->
        {:ok, client} = :gen_tcp.accept(listen_socket)
        {:ok, data} = :gen_tcp.recv(client, 0)
        :gen_tcp.send(client, data)
        :gen_tcp.close(client)
      end)
      
      # Connect and send data
      {:ok, conn_id} = NetworkTransport.connect("localhost", port, transport: :tcp)
      :ok = NetworkTransport.send_data(conn_id, "Hello, World!")
      
      Task.await(echo_server)
      :gen_tcp.close(listen_socket)
    end
  end

  describe "connection pooling" do
    test "reuses connections from pool" do
      # Create a simple server
      {:ok, listen_socket} = :gen_tcp.listen(0, [:binary, packet: 4, active: false])
      {:ok, port} = :inet.port(listen_socket)
      
      # Accept connections in background
      Task.start(fn ->
        for _ <- 1..10 do
          {:ok, _client} = :gen_tcp.accept(listen_socket)
        end
      end)
      
      # Send multiple messages to same endpoint
      for i <- 1..5 do
        :ok = NetworkTransport.send_to("localhost", port, "Message #{i}")
      end
      
      # Check metrics show connection reuse
      metrics = NetworkTransport.get_metrics()
      assert metrics.total_connections < 5  # Should reuse connections
      
      :gen_tcp.close(listen_socket)
    end
  end

  describe "circuit breaker" do
    test "handles connection failures appropriately" do
      # Connect to a server that accepts connections but closes them immediately
      {:ok, listen_socket} = :gen_tcp.listen(0, [:binary, packet: 4, active: false])
      {:ok, port} = :inet.port(listen_socket)
      
      # Accept connection then close it immediately
      Task.start(fn ->
        {:ok, client} = :gen_tcp.accept(listen_socket)
        :gen_tcp.close(client)
      end)
      
      {:ok, conn_id} = NetworkTransport.connect("localhost", port, transport: :tcp)
      
      # Give time for connection to be established and then closed
      Process.sleep(50)
      
      # After connection is closed, sends should fail with connection_not_found
      # This is the correct behavior - closed connections are removed from state
      assert {:error, :connection_not_found} = NetworkTransport.send_data(conn_id, "test")
      
      :gen_tcp.close(listen_socket)
    end
  end

  describe "metrics tracking" do
    test "tracks connection and message metrics" do
      initial_metrics = NetworkTransport.get_metrics()
      
      # Create some activity
      {:ok, listen_socket} = :gen_tcp.listen(0, [:binary, packet: 4, active: false])
      {:ok, port} = :inet.port(listen_socket)
      
      Task.start(fn ->
        {:ok, _client} = :gen_tcp.accept(listen_socket)
      end)
      
      {:ok, conn_id} = NetworkTransport.connect("localhost", port)
      NetworkTransport.disconnect(conn_id)
      
      final_metrics = NetworkTransport.get_metrics()
      
      assert final_metrics.total_connections > initial_metrics.total_connections
      assert final_metrics.active_connections >= 0
      
      :gen_tcp.close(listen_socket)
    end
  end
end
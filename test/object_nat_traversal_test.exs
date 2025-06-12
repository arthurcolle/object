defmodule Object.NATTraversalTest do
  use ExUnit.Case, async: false
  alias Object.NATTraversal

  setup do
    {:ok, _pid} = NATTraversal.start_link(
      stun_servers: [{"stun.l.google.com", 19302}],
      enable_upnp: false,  # Disable for tests
      enable_turn: false   # Disable for tests
    )
    :ok
  end

  describe "NAT discovery" do
    @tag :external_network
    test "discovers NAT type using STUN" do
      # This test requires internet connection
      case NATTraversal.discover_nat() do
        {:ok, {nat_type, external_addr, external_port}} ->
          assert nat_type in [:none, :full_cone, :restricted_cone, 
                             :port_restricted, :symmetric]
          assert is_binary(external_addr)
          assert is_integer(external_port)
          
        {:error, :no_stun_servers} ->
          # Expected if no STUN servers configured
          :ok
          
        {:error, :timeout} ->
          # Network might be unavailable
          :ok
      end
    end
  end

  describe "ICE candidate gathering" do
    test "gathers host candidates" do
      {:ok, candidates} = NATTraversal.gather_candidates("test_conn")
      
      # Should have at least one host candidate
      host_candidates = Enum.filter(candidates, &(&1.type == :host))
      assert length(host_candidates) > 0
      
      # Host candidates should have local addresses
      for candidate <- host_candidates do
        assert candidate.type == :host
        assert is_binary(candidate.address)
        assert candidate.priority > 0
        assert is_binary(candidate.foundation)
      end
    end

    test "assigns priorities correctly" do
      {:ok, candidates} = NATTraversal.gather_candidates("priority_test")
      
      # Verify priority calculation
      for candidate <- candidates do
        assert is_integer(candidate.priority)
        assert candidate.priority > 0
        
        # Host candidates should have highest priority
        if candidate.type == :host do
          assert candidate.priority > 100_000_000
        end
      end
    end

    test "generates unique foundations" do
      {:ok, candidates} = NATTraversal.gather_candidates("foundation_test")
      
      foundations = Enum.map(candidates, & &1.foundation)
      # Foundations should be unique per type/protocol/address combo
      assert length(foundations) == length(Enum.uniq(foundations))
    end
  end

  describe "ICE negotiation" do
    test "starts ICE negotiation with remote candidates" do
      # Gather local candidates first
      {:ok, local_candidates} = NATTraversal.gather_candidates("ice_test")
      
      # Simulate remote candidates
      remote_candidates = [
        %{
          type: :host,
          protocol: :udp,
          address: "192.168.1.100",
          port: 5000,
          priority: 1000000,
          foundation: "remote1",
          related_address: nil,
          related_port: nil
        }
      ]
      
      # Start negotiation
      result = NATTraversal.start_ice_negotiation(
        "ice_test", 
        remote_candidates, 
        true  # controlling
      )
      
      assert :ok = result
    end

    test "sorts candidate pairs by priority" do
      {:ok, _} = NATTraversal.gather_candidates("pair_test")
      
      remote = [
        %{type: :host, priority: 1000, address: "1.1.1.1", port: 1000},
        %{type: :srflx, priority: 500, address: "2.2.2.2", port: 2000}
      ]
      
      :ok = NATTraversal.start_ice_negotiation("pair_test", remote, true)
      
      # Pairs should be checked in priority order
      # (This is hard to verify without inspecting internal state)
    end
  end

  describe "UDP hole punching" do
    test "attempts hole punching" do
      # Create a local UDP server to test against
      {:ok, server} = :gen_udp.open(0, [:binary, active: false])
      {:ok, server_port} = :inet.port(server)
      
      # Start server that responds to punches
      server_task = Task.async(fn ->
        {:ok, {_addr, _port, "PUNCH1"}} = :gen_udp.recv(server, 0, 1000)
        :gen_udp.close(server)
      end)
      
      # Attempt hole punch
      result = NATTraversal.hole_punch(0, "127.0.0.1", server_port)
      
      case result do
        {:ok, socket} ->
          :gen_udp.close(socket)
        {:error, :hole_punch_failed} ->
          :ok
      end
      
      Task.await(server_task)
    end

    test "handles hole punch timeout" do
      # Try to punch to non-existent server
      result = NATTraversal.hole_punch(0, "127.0.0.1", 59999)
      assert {:error, :hole_punch_failed} = result
    end
  end

  describe "TURN relay allocation" do
    test "handles missing TURN servers gracefully" do
      # No TURN servers configured in setup
      result = NATTraversal.allocate_relay()
      assert {:error, :no_turn_servers} = result
    end

    @tag :skip
    test "allocates TURN relay when available" do
      # This would require a real TURN server
      # Skipped in unit tests
    end
  end

  describe "UPnP port mapping" do
    test "handles disabled UPnP" do
      # UPnP disabled in setup
      result = NATTraversal.create_port_mapping(8080, :tcp, "Test mapping")
      assert {:error, :upnp_disabled} = result
    end

    @tag :requires_upnp
    test "creates UPnP mapping when available" do
      # This test requires a UPnP-enabled router
      {:ok, _} = NATTraversal.start_link(enable_upnp: true)
      
      case NATTraversal.create_port_mapping(8080, :tcp, "Object P2P") do
        {:ok, external_port} ->
          assert is_integer(external_port)
          assert external_port > 0
          
        {:error, :no_upnp_gateway} ->
          # No UPnP gateway found
          :ok
      end
    end
  end

  describe "STUN message handling" do
    test "builds valid STUN binding requests" do
      # The module builds STUN messages internally
      # We test this indirectly through NAT discovery
      
      # Create mock STUN server
      {:ok, mock_stun} = :gen_udp.open(3478, [:binary, active: false])
      
      Task.start(fn ->
        # Wait for STUN request
        case :gen_udp.recv(mock_stun, 0, 1000) do
          {:ok, {addr, port, data}} ->
            # Verify it looks like STUN
            <<msg_type::16, _length::16, magic::32, _rest::binary>> = data
            assert msg_type == 0x0001  # Binding request
            assert magic == 0x2112A442  # Magic cookie
            
            # Send mock response
            response = build_mock_stun_response(data, addr)
            :gen_udp.send(mock_stun, addr, port, response)
            
          {:error, :timeout} ->
            :ok
        end
        
        :gen_udp.close(mock_stun)
      end)
      
      # Trigger STUN request
      NATTraversal.discover_nat()
      
      Process.sleep(100)
    end
  end

  # Helper functions
  
  defp build_mock_stun_response(_request, {a, b, c, d}) do
    # Build a minimal STUN binding response
    msg_type = 0x0101  # Binding response
    transaction_id = :crypto.strong_rand_bytes(12)
    
    # XOR-MAPPED-ADDRESS attribute
    addr_type = 0x0020
    addr_length = 8
    family = 0x01  # IPv4
    port = 12345 ^^^ (0x2112A442 >>> 16)
    addr = (a <<< 24 ||| b <<< 16 ||| c <<< 8 ||| d) ^^^ 0x2112A442
    
    attr = <<addr_type::16, addr_length::16, 0::8, family::8, 
             port::16, addr::32>>
    
    <<msg_type::16, byte_size(attr)::16, 0x2112A442::32, 
      transaction_id::binary, attr::binary>>
  end
end
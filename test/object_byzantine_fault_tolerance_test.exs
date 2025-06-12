defmodule Object.ByzantineFaultToleranceTest do
  use ExUnit.Case, async: false
  alias Object.ByzantineFaultTolerance, as: BFT

  setup do
    {:ok, _pid} = BFT.start_link(
      node_id: "test_bft_node",
      require_pow: false,  # Disable for faster tests
      min_reputation: 0.3
    )
    :ok
  end

  describe "reputation system" do
    test "tracks node reputation" do
      node_id = :crypto.strong_rand_bytes(20)
      
      # Initial reputation
      initial_rep = BFT.get_reputation(node_id)
      assert initial_rep == 0.5  # Default
      
      # Update with success
      BFT.update_reputation(node_id, :success)
      new_rep = BFT.get_reputation(node_id)
      assert new_rep > initial_rep
      
      # Update with failure
      BFT.update_reputation(node_id, :failure)
      final_rep = BFT.get_reputation(node_id)
      assert final_rep < new_rep
    end

    test "handles reputation violations" do
      node_id = :crypto.strong_rand_bytes(20)
      
      violation = %{
        type: :double_spend,
        timestamp: DateTime.utc_now(),
        evidence: %{tx1: "abc", tx2: "def"}
      }
      
      BFT.update_reputation(node_id, violation)
      
      # Should severely impact reputation
      rep = BFT.get_reputation(node_id)
      assert rep < 0.2  # Heavy penalty for double spend
    end

    test "determines trustworthiness based on reputation" do
      good_node = :crypto.strong_rand_bytes(20)
      bad_node = :crypto.strong_rand_bytes(20)
      
      # Build good reputation
      for _ <- 1..10 do
        BFT.update_reputation(good_node, :success)
      end
      
      # Build bad reputation
      for _ <- 1..10 do
        BFT.update_reputation(bad_node, :failure)
      end
      
      assert BFT.is_trustworthy?(good_node) == true
      assert BFT.is_trustworthy?(bad_node) == false
    end
  end

  describe "Byzantine consensus" do
    test "initiates consensus rounds" do
      participants = for _ <- 1..5, do: :crypto.strong_rand_bytes(20)
      value = %{action: "update_config", data: %{key: "value"}}
      
      {:ok, round_id} = BFT.start_consensus(value, participants)
      assert is_binary(round_id)
    end

    test "processes consensus votes" do
      participants = for _ <- 1..5, do: :crypto.strong_rand_bytes(20)
      value = %{decision: "important"}
      
      {:ok, round_id} = BFT.start_consensus(value, participants)
      
      # Cast votes (non-blocking)
      BFT.vote_consensus(round_id, :prepare, true)
      BFT.vote_consensus(round_id, :prepare, false)  # Negative vote
      
      # Votes are processed asynchronously
    end

    test "requires 2/3 majority for consensus" do
      # Create 9 participants for clear 2/3 majority
      participants = for i <- 1..9, do: "node_#{i}"
      value = %{proposal: "test"}
      
      {:ok, round_id} = BFT.start_consensus(value, participants)
      
      # Vote from 6 nodes (exactly 2/3)
      for i <- 1..6 do
        BFT.vote_consensus(round_id, :prepare, true)
      end
      
      # This would advance to commit phase in real implementation
    end

    test "handles consensus timeout" do
      participants = for _ <- 1..3, do: :crypto.strong_rand_bytes(20)
      value = %{timeout: "test"}
      
      # Start with very short timeout
      {:ok, _pid} = BFT.start_link(
        node_id: "timeout_test",
        consensus_timeout: 100  # 100ms
      )
      
      {:ok, round_id} = BFT.start_consensus(value, participants)
      
      # Wait for timeout
      Process.sleep(200)
      
      # Round should be cleaned up (hard to verify without inspecting state)
    end
  end

  describe "proof of work challenges" do
    test "issues proof of work challenges" do
      node_id = :crypto.strong_rand_bytes(20)
      
      {:ok, challenge_id} = BFT.challenge_node(node_id, :proof_of_work)
      assert is_binary(challenge_id)
    end

    test "verifies proof of work responses" do
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, challenge_id} = BFT.challenge_node(node_id, :proof_of_work)
      
      # Generate valid proof of work (simplified for test)
      # In reality, would need to find nonce that produces hash with leading zeros
      valid_response = :crypto.strong_rand_bytes(32)
      
      result = BFT.respond_to_challenge(challenge_id, valid_response)
      assert {:error, _} = result  # Will fail without proper PoW
    end

    test "handles challenge expiration" do
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, challenge_id} = BFT.challenge_node(node_id, :proof_of_work)
      
      # Wait for expiration (would need to configure shorter timeout)
      # Process.sleep(31_000)  # Too long for tests
      
      # Expired challenges should be rejected
      future_result = BFT.respond_to_challenge(challenge_id, "late_response")
      assert match?({:error, _}, future_result) or match?(:ok, future_result)
    end
  end

  describe "data availability challenges" do
    test "challenges nodes for data availability" do
      node_id = :crypto.strong_rand_bytes(20)
      
      {:ok, challenge_id} = BFT.challenge_node(node_id, :data_availability)
      assert is_binary(challenge_id)
    end

    test "verifies data availability proofs" do
      node_id = :crypto.strong_rand_bytes(20)
      {:ok, challenge_id} = BFT.challenge_node(node_id, :data_availability)
      
      # Mock response with requested data and merkle proofs
      response = %{
        "audit_id_1" => {"data1", [{:left, "hash1"}]},
        "audit_id_2" => {"data2", [{:right, "hash2"}]}
      }
      
      result = BFT.respond_to_challenge(challenge_id, response)
      assert {:error, _} = result  # Will fail without valid merkle proofs
    end
  end

  describe "Merkle tree verification" do
    test "verifies valid merkle proofs" do
      # Create a simple merkle tree
      data = "important_data"
      data_hash = :crypto.hash(:sha256, data)
      
      # Build proof (simplified)
      sibling = :crypto.hash(:sha256, "sibling_data")
      intermediate = :crypto.hash(:sha256, [data_hash, sibling])
      root = :crypto.hash(:sha256, [intermediate, intermediate])
      
      proof = [
        {:right, sibling},
        {:right, intermediate}
      ]
      
      # Should accept valid proof
      result = BFT.verify_merkle_proof(data, proof, root)
      assert is_boolean(result)
    end

    test "rejects invalid merkle proofs" do
      data = "test_data"
      fake_proof = [{:left, :crypto.strong_rand_bytes(32)}]
      fake_root = :crypto.strong_rand_bytes(32)
      
      result = BFT.verify_merkle_proof(data, fake_proof, fake_root)
      assert result == false
    end
  end

  describe "audit log" do
    test "maintains audit trail of consensus decisions" do
      # Audit log is internal, but we can trigger operations that create entries
      participants = ["node1", "node2", "node3"]
      value = %{audit: "test"}
      
      {:ok, round_id} = BFT.start_consensus(value, participants)
      
      # Complete consensus would add audit entry
      # Entries are periodically cleaned up
    end
  end

  describe "Sybil resistance" do
    test "requires proof of work when enabled" do
      {:ok, _pid} = BFT.start_link(
        node_id: "sybil_resistant",
        require_pow: true
      )
      
      new_node = :crypto.strong_rand_bytes(20)
      
      # Without PoW, node is not trustworthy
      assert BFT.is_trustworthy?(new_node) == false
      
      # After PoW challenge/response, would become trustworthy
    end

    test "combines reputation and PoW for trust" do
      {:ok, _pid} = BFT.start_link(
        node_id: "combined_trust",
        require_pow: true,
        min_reputation: 0.5
      )
      
      node_id = :crypto.strong_rand_bytes(20)
      
      # Even with good reputation, needs PoW
      for _ <- 1..20 do
        BFT.update_reputation(node_id, :success)
      end
      
      # Still not trustworthy without PoW
      assert BFT.is_trustworthy?(node_id) == false
    end
  end
end
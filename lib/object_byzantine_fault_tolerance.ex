defmodule Object.ByzantineFaultTolerance do
  @moduledoc """
  Byzantine fault tolerance mechanisms for the Object P2P network.
  
  Implements various algorithms and strategies to handle malicious nodes
  including Sybil attack resistance, consensus protocols, reputation systems,
  and proof-of-work challenges.
  
  ## Features
  
  - PBFT-inspired consensus for critical operations
  - Reputation-based trust system
  - Proof-of-work for Sybil resistance
  - Merkle tree verification for data integrity
  - Threshold signatures for group decisions
  - Audit trails and accountability
  """
  
  use GenServer
  require Logger
  
  @min_reputation 0.1
  @max_reputation 1.0
  @initial_reputation 0.5
  @consensus_threshold 0.67  # 2/3 majority
  @pow_difficulty 20  # Number of leading zero bits
  @challenge_timeout 30_000
  @audit_retention_days 30
  
  @type node_reputation :: %{
    node_id: binary(),
    reputation: float(),
    interactions: non_neg_integer(),
    successful: non_neg_integer(),
    failed: non_neg_integer(),
    last_updated: DateTime.t(),
    proof_of_work: binary() | nil,
    violations: [violation()]
  }
  
  @type violation :: %{
    type: :double_spend | :invalid_signature | :protocol_violation | :dos_attack,
    timestamp: DateTime.t(),
    evidence: term()
  }
  
  @type consensus_round :: %{
    id: binary(),
    proposer: binary(),
    value: term(),
    phase: :prepare | :commit | :complete | :aborted,
    votes: %{prepare: MapSet.t(), commit: MapSet.t()},
    participants: MapSet.t(),
    started_at: DateTime.t(),
    timeout_at: DateTime.t()
  }
  
  @type audit_entry :: %{
    id: binary(),
    operation: atom(),
    participants: [binary()],
    result: term(),
    signatures: %{binary() => binary()},
    timestamp: DateTime.t(),
    merkle_root: binary()
  }
  
  @type state :: %{
    node_id: binary(),
    reputations: %{binary() => node_reputation()},
    consensus_rounds: %{binary() => consensus_round()},
    pending_challenges: %{binary() => challenge()},
    audit_log: [audit_entry()],
    merkle_tree: merkle_tree(),
    config: map()
  }
  
  @type challenge :: %{
    challenger: binary(),
    challenged: binary(),
    type: :proof_of_work | :data_availability | :computation,
    challenge_data: term(),
    issued_at: DateTime.t(),
    expires_at: DateTime.t()
  }
  
  @type merkle_tree :: %{
    root: binary(),
    height: non_neg_integer(),
    nodes: %{non_neg_integer() => [binary()]}
  }
  
  # Client API
  
  @doc """
  Starts the Byzantine fault tolerance service.
  """
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  @doc """
  Checks if a node is trustworthy based on reputation.
  """
  @spec is_trustworthy?(binary()) :: boolean()
  def is_trustworthy?(node_id) do
    GenServer.call(__MODULE__, {:is_trustworthy, node_id})
  end
  
  @doc """
  Updates reputation based on interaction outcome.
  """
  @spec update_reputation(binary(), :success | :failure | violation()) :: :ok
  def update_reputation(node_id, outcome) do
    GenServer.cast(__MODULE__, {:update_reputation, node_id, outcome})
  end
  
  @doc """
  Initiates Byzantine consensus for a value.
  """
  @spec start_consensus(term(), [binary()]) :: {:ok, binary()} | {:error, term()}
  def start_consensus(value, participants) do
    GenServer.call(__MODULE__, {:start_consensus, value, participants}, 60_000)
  end
  
  @doc """
  Votes in a consensus round.
  """
  @spec vote_consensus(binary(), :prepare | :commit, boolean()) :: :ok
  def vote_consensus(round_id, phase, vote) do
    GenServer.cast(__MODULE__, {:vote_consensus, round_id, phase, vote})
  end
  
  @doc """
  Issues a proof-of-work challenge to a node.
  """
  @spec challenge_node(binary(), :proof_of_work | :data_availability) :: 
    {:ok, binary()} | {:error, term()}
  def challenge_node(node_id, challenge_type) do
    GenServer.call(__MODULE__, {:challenge_node, node_id, challenge_type})
  end
  
  @doc """
  Submits a challenge response.
  """
  @spec respond_to_challenge(binary(), term()) :: :ok | {:error, term()}
  def respond_to_challenge(challenge_id, response) do
    GenServer.call(__MODULE__, {:respond_to_challenge, challenge_id, response})
  end
  
  @doc """
  Verifies data integrity using Merkle proofs.
  """
  @spec verify_merkle_proof(binary(), [binary()], binary()) :: boolean()
  def verify_merkle_proof(data, proof, root) do
    GenServer.call(__MODULE__, {:verify_merkle_proof, data, proof, root})
  end
  
  @doc """
  Gets the current reputation of a node.
  """
  @spec get_reputation(binary()) :: float()
  def get_reputation(node_id) do
    GenServer.call(__MODULE__, {:get_reputation, node_id})
  end
  
  # Server Callbacks
  
  @impl true
  def init(opts) do
    node_id = Keyword.get(opts, :node_id, Object.DistributedRegistry.get_node_id())
    
    state = %{
      node_id: node_id,
      reputations: %{},
      consensus_rounds: %{},
      pending_challenges: %{},
      audit_log: [],
      merkle_tree: %{root: <<>>, height: 0, nodes: %{}},
      config: %{
        require_pow: Keyword.get(opts, :require_pow, true),
        min_reputation_threshold: Keyword.get(opts, :min_reputation, 0.3),
        consensus_timeout: Keyword.get(opts, :consensus_timeout, 30_000),
        audit_public_key: Keyword.get(opts, :audit_public_key)
      }
    }
    
    # Schedule periodic tasks
    schedule_reputation_decay()
    schedule_audit_cleanup()
    schedule_consensus_timeout_check()
    
    {:ok, state}
  end
  
  @impl true
  def handle_call({:is_trustworthy, node_id}, _from, state) do
    result = case Map.get(state.reputations, node_id) do
      nil -> 
        # Unknown node - require proof of work
        state.config.require_pow == false
        
      %{reputation: rep, proof_of_work: pow} ->
        rep >= state.config.min_reputation_threshold and
        (not state.config.require_pow or pow != nil)
    end
    
    {:reply, result, state}
  end
  
  @impl true
  def handle_call({:get_reputation, node_id}, _from, state) do
    reputation = case Map.get(state.reputations, node_id) do
      nil -> @initial_reputation
      %{reputation: rep} -> rep
    end
    
    {:reply, reputation, state}
  end
  
  @impl true
  def handle_call({:start_consensus, value, participants}, _from, state) do
    round_id = generate_round_id()
    
    round = %{
      id: round_id,
      proposer: state.node_id,
      value: value,
      phase: :prepare,
      votes: %{prepare: MapSet.new(), commit: MapSet.new()},
      participants: MapSet.new(participants),
      started_at: DateTime.utc_now(),
      timeout_at: DateTime.add(DateTime.utc_now(), 
                              state.config.consensus_timeout, :millisecond)
    }
    
    # Send prepare messages to participants
    send_prepare_messages(round, state)
    
    new_state = put_in(state.consensus_rounds[round_id], round)
    {:reply, {:ok, round_id}, new_state}
  end
  
  @impl true
  def handle_call({:challenge_node, node_id, challenge_type}, _from, state) do
    challenge_id = generate_challenge_id()
    
    challenge = case challenge_type do
      :proof_of_work ->
        %{
          challenger: state.node_id,
          challenged: node_id,
          type: :proof_of_work,
          challenge_data: generate_pow_challenge(),
          issued_at: DateTime.utc_now(),
          expires_at: DateTime.add(DateTime.utc_now(), @challenge_timeout, :millisecond)
        }
        
      :data_availability ->
        %{
          challenger: state.node_id,
          challenged: node_id,
          type: :data_availability,
          challenge_data: generate_data_challenge(state),
          issued_at: DateTime.utc_now(),
          expires_at: DateTime.add(DateTime.utc_now(), @challenge_timeout, :millisecond)
        }
    end
    
    new_state = put_in(state.pending_challenges[challenge_id], challenge)
    
    # Send challenge to node
    send_challenge(node_id, challenge_id, challenge)
    
    {:reply, {:ok, challenge_id}, new_state}
  end
  
  @impl true
  def handle_call({:respond_to_challenge, challenge_id, response}, _from, state) do
    case Map.get(state.pending_challenges, challenge_id) do
      nil ->
        {:reply, {:error, :invalid_challenge}, state}
        
      challenge ->
        if DateTime.compare(DateTime.utc_now(), challenge.expires_at) == :gt do
          {:reply, {:error, :challenge_expired}, state}
        else
          case verify_challenge_response(challenge, response) do
            :ok ->
              # Update reputation positively
              new_state = update_node_reputation(state, challenge.challenged, :success)
              
              # Store proof of work if applicable
              newer_state = if challenge.type == :proof_of_work do
                store_proof_of_work(new_state, challenge.challenged, response)
              else
                new_state
              end
              
              # Remove challenge
              final_state = Map.delete(newer_state.pending_challenges, challenge_id)
              
              {:reply, :ok, final_state}
              
            {:error, reason} ->
              # Update reputation negatively
              new_state = update_node_reputation(state, challenge.challenged, 
                {:failure, :invalid_challenge_response})
              
              {:reply, {:error, reason}, new_state}
          end
        end
    end
  end
  
  @impl true
  def handle_call({:verify_merkle_proof, data, proof, root}, _from, state) do
    result = verify_merkle_path(hash_data(data), proof, root)
    {:reply, result, state}
  end
  
  @impl true
  def handle_cast({:update_reputation, node_id, outcome}, state) do
    new_state = update_node_reputation(state, node_id, outcome)
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:vote_consensus, round_id, phase, vote}, state) do
    case Map.get(state.consensus_rounds, round_id) do
      nil ->
        {:noreply, state}
        
      round ->
        new_round = process_consensus_vote(round, state.node_id, phase, vote)
        
        # Check if consensus reached
        new_state = if consensus_reached?(new_round, phase) do
          advance_consensus_phase(state, round_id, new_round)
        else
          put_in(state.consensus_rounds[round_id], new_round)
        end
        
        {:noreply, new_state}
    end
  end
  
  @impl true
  def handle_info(:reputation_decay, state) do
    # Slowly decay reputation over time to handle node churn
    now = DateTime.utc_now()
    
    updated_reputations = state.reputations
    |> Enum.map(fn {node_id, rep} ->
      days_old = DateTime.diff(now, rep.last_updated, :day)
      decay_factor = :math.pow(0.99, days_old)
      
      new_rep = %{rep | 
        reputation: max(@min_reputation, rep.reputation * decay_factor)
      }
      
      {node_id, new_rep}
    end)
    |> Map.new()
    
    schedule_reputation_decay()
    {:noreply, %{state | reputations: updated_reputations}}
  end
  
  @impl true
  def handle_info(:audit_cleanup, state) do
    # Remove old audit entries
    cutoff = DateTime.add(DateTime.utc_now(), -@audit_retention_days, :day)
    
    recent_audits = Enum.filter(state.audit_log, fn entry ->
      DateTime.compare(entry.timestamp, cutoff) == :gt
    end)
    
    # Rebuild merkle tree with recent audits
    new_merkle = build_merkle_tree(recent_audits)
    
    schedule_audit_cleanup()
    {:noreply, %{state | audit_log: recent_audits, merkle_tree: new_merkle}}
  end
  
  @impl true
  def handle_info(:check_consensus_timeouts, state) do
    now = DateTime.utc_now()
    
    {active, timed_out} = state.consensus_rounds
    |> Enum.split_with(fn {_id, round} ->
      DateTime.compare(now, round.timeout_at) == :lt
    end)
    
    # Abort timed out rounds
    Enum.each(timed_out, fn {id, round} ->
      Logger.warning("Consensus round #{id} timed out")
      
      # Record failure in audit log
      audit_consensus_failure(state, round, :timeout)
    end)
    
    schedule_consensus_timeout_check()
    {:noreply, %{state | consensus_rounds: Map.new(active)}}
  end
  
  # Reputation Management
  
  defp update_node_reputation(state, node_id, outcome) do
    current = Map.get(state.reputations, node_id, %{
      node_id: node_id,
      reputation: @initial_reputation,
      interactions: 0,
      successful: 0,
      failed: 0,
      last_updated: DateTime.utc_now(),
      proof_of_work: nil,
      violations: []
    })
    
    updated = case outcome do
      :success ->
        %{current |
          reputation: min(@max_reputation, current.reputation + 0.01),
          interactions: current.interactions + 1,
          successful: current.successful + 1,
          last_updated: DateTime.utc_now()
        }
        
      :failure ->
        %{current |
          reputation: max(@min_reputation, current.reputation - 0.05),
          interactions: current.interactions + 1,
          failed: current.failed + 1,
          last_updated: DateTime.utc_now()
        }
        
      %{type: violation_type} = violation ->
        penalty = case violation_type do
          :double_spend -> 0.5
          :invalid_signature -> 0.3
          :protocol_violation -> 0.2
          :dos_attack -> 0.4
        end
        
        %{current |
          reputation: max(@min_reputation, current.reputation - penalty),
          interactions: current.interactions + 1,
          failed: current.failed + 1,
          violations: [violation | current.violations],
          last_updated: DateTime.utc_now()
        }
    end
    
    put_in(state.reputations[node_id], updated)
  end
  
  # Consensus Protocol (Simplified PBFT)
  
  defp send_prepare_messages(round, state) do
    message = %{
      type: :prepare,
      round_id: round.id,
      value: round.value,
      proposer: round.proposer
    }
    
    Enum.each(round.participants, fn participant ->
      send_consensus_message(participant, message, state)
    end)
  end
  
  defp process_consensus_vote(round, voter, :prepare, true) do
    update_in(round.votes.prepare, &MapSet.put(&1, voter))
  end
  
  defp process_consensus_vote(round, voter, :commit, true) do
    update_in(round.votes.commit, &MapSet.put(&1, voter))
  end
  
  defp process_consensus_vote(round, _voter, _phase, false) do
    # Negative vote - could track for accountability
    round
  end
  
  defp consensus_reached?(round, phase) do
    votes = case phase do
      :prepare -> round.votes.prepare
      :commit -> round.votes.commit
    end
    
    vote_count = MapSet.size(votes)
    participant_count = MapSet.size(round.participants)
    threshold = ceil(participant_count * @consensus_threshold)
    
    vote_count >= threshold
  end
  
  defp advance_consensus_phase(state, round_id, round) do
    case round.phase do
      :prepare ->
        # Move to commit phase
        new_round = %{round | phase: :commit}
        send_commit_messages(new_round, state)
        put_in(state.consensus_rounds[round_id], new_round)
        
      :commit ->
        # Consensus complete
        finalize_consensus(state, round_id, round)
    end
  end
  
  defp send_commit_messages(round, state) do
    message = %{
      type: :commit,
      round_id: round.id,
      value: round.value,
      prepare_votes: MapSet.to_list(round.votes.prepare)
    }
    
    Enum.each(round.participants, fn participant ->
      send_consensus_message(participant, message, state)
    end)
  end
  
  defp finalize_consensus(state, round_id, round) do
    # Create audit entry
    audit_entry = %{
      id: generate_audit_id(),
      operation: :consensus,
      participants: MapSet.to_list(round.participants),
      result: %{value: round.value, round_id: round_id},
      signatures: collect_signatures(round),
      timestamp: DateTime.utc_now(),
      merkle_root: <<>>  # Will be updated when rebuilding tree
    }
    
    # Update state
    state
    |> Map.delete([:consensus_rounds, round_id])
    |> update_in([:audit_log], &[audit_entry | &1])
    |> update_merkle_tree()
  end
  
  # Challenge System
  
  defp generate_pow_challenge do
    # Random data that must be combined with solution
    nonce = :crypto.strong_rand_bytes(32)
    %{
      nonce: nonce,
      difficulty: @pow_difficulty,
      algorithm: :sha256
    }
  end
  
  defp generate_data_challenge(state) do
    # Select random audit entries for availability check
    if length(state.audit_log) > 0 do
      selected = Enum.take_random(state.audit_log, 3)
      %{
        audit_ids: Enum.map(selected, & &1.id),
        merkle_root: state.merkle_tree.root
      }
    else
      %{audit_ids: [], merkle_root: <<>>}
    end
  end
  
  defp verify_challenge_response(%{type: :proof_of_work} = challenge, response) do
    %{nonce: nonce, difficulty: difficulty} = challenge.challenge_data
    
    # Verify proof of work
    hash = :crypto.hash(:sha256, [nonce, response])
    
    if has_leading_zeros?(hash, difficulty) do
      :ok
    else
      {:error, :invalid_proof_of_work}
    end
  end
  
  defp verify_challenge_response(%{type: :data_availability} = challenge, response) do
    # Verify the node has the requested data
    %{audit_ids: requested_ids} = challenge.challenge_data
    
    provided_ids = Map.keys(response)
    
    if MapSet.subset?(MapSet.new(requested_ids), MapSet.new(provided_ids)) do
      # Verify merkle proofs for each entry
      all_valid = Enum.all?(response, fn {_id, {data, proof}} ->
        verify_merkle_proof(data, proof, challenge.challenge_data.merkle_root)
      end)
      
      if all_valid do
        :ok
      else
        {:error, :invalid_merkle_proof}
      end
    else
      {:error, :missing_data}
    end
  end
  
  defp has_leading_zeros?(hash, required_zeros) do
    <<int::size(required_zeros), _::bitstring>> = hash
    int == 0
  end
  
  defp store_proof_of_work(state, node_id, proof) do
    update_in(state.reputations[node_id], fn rep ->
      %{rep | proof_of_work: proof}
    end)
  end
  
  # Merkle Tree Operations
  
  defp build_merkle_tree([]), do: %{root: <<>>, height: 0, nodes: %{}}
  defp build_merkle_tree(entries) do
    # Build leaves
    leaves = Enum.map(entries, fn entry ->
      hash_data(:erlang.term_to_binary(entry))
    end)
    
    build_merkle_levels(leaves, %{}, 0)
  end
  
  defp build_merkle_levels([single], nodes, height) do
    %{
      root: single,
      height: height,
      nodes: Map.put(nodes, height, [single])
    }
  end
  
  defp build_merkle_levels(level, nodes, height) do
    # Pad if odd number
    padded = if rem(length(level), 2) == 1 do
      level ++ [List.last(level)]
    else
      level
    end
    
    # Build next level
    next_level = padded
    |> Enum.chunk_every(2)
    |> Enum.map(fn [left, right] ->
      hash_data([left, right])
    end)
    
    new_nodes = Map.put(nodes, height, level)
    build_merkle_levels(next_level, new_nodes, height + 1)
  end
  
  defp verify_merkle_path(leaf_hash, proof, root) do
    final_hash = Enum.reduce(proof, leaf_hash, fn {side, sibling}, current ->
      case side do
        :left -> hash_data([sibling, current])
        :right -> hash_data([current, sibling])
      end
    end)
    
    final_hash == root
  end
  
  defp update_merkle_tree(state) do
    new_tree = build_merkle_tree(state.audit_log)
    %{state | merkle_tree: new_tree}
  end
  
  # Utilities
  
  defp hash_data(data) do
    :crypto.hash(:sha256, :erlang.term_to_binary(data))
  end
  
  defp generate_round_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
  
  defp generate_challenge_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
  
  defp generate_audit_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
  
  defp send_consensus_message(participant, _message, _state) do
    # Would use Object.NetworkProxy to send message
    Task.start(fn ->
      Logger.debug("Sending consensus message to #{Base.encode16(participant)}")
    end)
  end
  
  defp send_challenge(node_id, challenge_id, _challenge) do
    Task.start(fn ->
      Logger.debug("Sending challenge #{challenge_id} to #{Base.encode16(node_id)}")
    end)
  end
  
  defp collect_signatures(round) do
    # Would collect actual signatures from participants
    round.participants
    |> Enum.map(fn participant ->
      {participant, :crypto.strong_rand_bytes(64)}
    end)
    |> Map.new()
  end
  
  defp audit_consensus_failure(_state, round, reason) do
    Logger.warning("Consensus round #{round.id} failed: #{reason}")
  end
  
  defp schedule_reputation_decay do
    Process.send_after(self(), :reputation_decay, 86_400_000)  # Daily
  end
  
  defp schedule_audit_cleanup do
    Process.send_after(self(), :audit_cleanup, 86_400_000)  # Daily
  end
  
  defp schedule_consensus_timeout_check do
    Process.send_after(self(), :check_consensus_timeouts, 5_000)  # Every 5 seconds
  end
end
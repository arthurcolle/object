defmodule Object.TrustManager do
  @moduledoc """
  Manages trust relationships between objects, including reputation tracking,
  Byzantine fault detection, and trust-based decision making.
  """

  defstruct [
    :reputation_decay,
    :trust_threshold,
    :verification_probability,
    :byzantine_detection_enabled,
    reputation_scores: %{},
    interaction_history: %{},
    trust_network: %{},
    byzantine_suspects: MapSet.new(),
    verification_results: %{}
  ]

  def new(opts) do
    %__MODULE__{
      reputation_decay: opts[:reputation_decay] || 0.01,
      trust_threshold: opts[:trust_threshold] || 0.5,
      verification_probability: opts[:verification_probability] || 0.1,
      byzantine_detection_enabled: opts[:byzantine_detection_enabled] || false
    }
  end

  def update_reputation(trust_manager, object_id, interaction_result) do
    current_rep = Map.get(trust_manager.reputation_scores, object_id, 0.5)
    
    new_rep = case interaction_result do
      :success -> min(1.0, current_rep + 0.1)
      :failure -> max(0.0, current_rep - 0.1)
      :partial -> current_rep + 0.02
      _ -> current_rep
    end
    
    new_scores = Map.put(trust_manager.reputation_scores, object_id, new_rep)
    
    # Record interaction
    history = Map.get(trust_manager.interaction_history, object_id, [])
    new_history = Map.put(trust_manager.interaction_history, object_id, 
                         [interaction_result | Enum.take(history, 99)])
    
    %{trust_manager | 
      reputation_scores: new_scores,
      interaction_history: new_history
    }
  end

  def get_trust_score(trust_manager, object_id) do
    Map.get(trust_manager.reputation_scores, object_id, 0.5)
  end

  def is_trustworthy?(trust_manager, object_id) do
    get_trust_score(trust_manager, object_id) >= trust_manager.trust_threshold
  end

  def should_verify?(trust_manager, _object_id) do
    :rand.uniform() <= trust_manager.verification_probability
  end

  def detect_byzantine_behavior(trust_manager, object_id, claimed_state, actual_state) do
    if trust_manager.byzantine_detection_enabled do
      inconsistency = calculate_state_inconsistency(claimed_state, actual_state)
      
      if inconsistency > 0.7 do
        suspects = MapSet.put(trust_manager.byzantine_suspects, object_id)
        trust_manager = %{trust_manager | byzantine_suspects: suspects}
        
        # Heavily penalize reputation
        trust_manager = update_reputation(trust_manager, object_id, :byzantine_detected)
        
        {trust_manager, :byzantine_detected}
      else
        {trust_manager, :normal}
      end
    else
      {trust_manager, :detection_disabled}
    end
  end

  defp calculate_state_inconsistency(claimed, actual) when is_map(claimed) and is_map(actual) do
    all_keys = MapSet.union(MapSet.new(Map.keys(claimed)), MapSet.new(Map.keys(actual)))
    
    inconsistencies = Enum.count(all_keys, fn key ->
      Map.get(claimed, key) != Map.get(actual, key)
    end)
    
    inconsistencies / max(1, MapSet.size(all_keys))
  end
  defp calculate_state_inconsistency(claimed, actual) do
    if claimed == actual, do: 0.0, else: 1.0
  end

  def apply_reputation_decay(trust_manager) do
    decayed_scores = Enum.into(trust_manager.reputation_scores, %{}, fn {id, score} ->
      new_score = score * (1 - trust_manager.reputation_decay)
      {id, max(0.0, new_score)}
    end)
    
    %{trust_manager | reputation_scores: decayed_scores}
  end

  def get_trust_network_stats(trust_manager) do
    total_objects = map_size(trust_manager.reputation_scores)
    trustworthy_count = Enum.count(trust_manager.reputation_scores, fn {_id, score} ->
      score >= trust_manager.trust_threshold
    end)
    
    avg_reputation = if total_objects > 0 do
      Enum.sum(Map.values(trust_manager.reputation_scores)) / total_objects
    else
      0.0
    end
    
    %{
      total_objects: total_objects,
      trustworthy_count: trustworthy_count,
      trustworthy_ratio: if(total_objects > 0, do: trustworthy_count / total_objects, else: 0.0),
      average_reputation: avg_reputation,
      byzantine_suspects: MapSet.size(trust_manager.byzantine_suspects)
    }
  end

  def verify_reports(trust_manager, reports) do
    verified_reports = Enum.map(reports, fn report ->
      # Verify report truthfulness based on trust score
      trust_score = get_trust_score(trust_manager, report.object_id)
      _verification_threshold = trust_manager.verification_probability
      
      verified = if should_verify?(trust_manager, report.object_id) do
        # Simulate verification - in real system would cross-check with multiple sources
        deception_magnitude = Map.get(report, :deception_magnitude, 0.0)
        deception_magnitude < 0.3  # Detection threshold
      else
        # Trust the report based on trust score
        trust_score > trust_manager.trust_threshold
      end
      
      Map.put(report, :verified, verified)
    end)
    
    # Update trust manager based on verification results
    updated_trust_manager = Enum.reduce(verified_reports, trust_manager, fn report, acc ->
      if Map.get(report, :verified, true) do
        update_reputation(acc, report.object_id, :success)
      else
        # Mark as potentially byzantine
        suspects = MapSet.put(acc.byzantine_suspects, report.object_id)
        updated_acc = %{acc | byzantine_suspects: suspects}
        update_reputation(updated_acc, report.object_id, :byzantine_detected)
      end
    end)
    
    {verified_reports, updated_trust_manager}
  end

  def update_trust_scores(trust_manager, verified_reports) do
    Enum.map(verified_reports, fn report ->
      trust_score = get_trust_score(trust_manager, report.object_id)
      is_verified = Map.get(report, :verified, true)
      
      %{
        object_id: report.object_id,
        old_trust_score: trust_score,
        new_trust_score: if(is_verified, do: min(1.0, trust_score + 0.1), else: max(0.0, trust_score - 0.2)),
        verification_result: is_verified,
        action: if(is_verified, do: :trust_increased, else: :trust_decreased)
      }
    end)
  end
end
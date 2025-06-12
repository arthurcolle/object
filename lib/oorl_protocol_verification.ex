defmodule OORL.ProtocolVerification do
  @moduledoc """
  Provides protocol verification capabilities including LTS (Labeled Transition System)
  verification, reachability analysis, and protocol property checking.
  """

  def compute_reachable_states(protocol_lts) do
    initial_state = protocol_lts.initial_state
    transitions = protocol_lts.transitions
    
    reachable = MapSet.new([initial_state])
    
    # Fixed-point computation for reachability
    compute_reachable_fixpoint(reachable, transitions, MapSet.new())
  end

  defp compute_reachable_fixpoint(current_reachable, transitions, prev_reachable) do
    if MapSet.equal?(current_reachable, prev_reachable) do
      MapSet.to_list(current_reachable)
    else
      new_reachable = Enum.reduce(current_reachable, current_reachable, fn state, acc ->
        reachable_from_state = get_reachable_from_state(state, transitions)
        MapSet.union(acc, MapSet.new(reachable_from_state))
      end)
      
      compute_reachable_fixpoint(new_reachable, transitions, current_reachable)
    end
  end

  defp get_reachable_from_state(state, transitions) do
    Enum.reduce(transitions, [], fn {{from_state, _action}, to_state}, acc ->
      if from_state == state do
        [to_state | acc]
      else
        acc
      end
    end)
  end

  def verify_reachability_properties(protocol_lts, properties) do
    reachable_states = compute_reachable_states(protocol_lts)
    reachable_set = MapSet.new(reachable_states)
    
    Enum.map(properties, fn property ->
      case property do
        {:eventually, target_state} ->
          result = MapSet.member?(reachable_set, target_state)
          %{property: property, satisfied: result, type: :eventually}
          
        {:always_reachable, state} ->
          result = MapSet.member?(reachable_set, state)
          %{property: property, satisfied: result, type: :always_reachable}
          
        {:never_reach, forbidden_state} ->
          result = not MapSet.member?(reachable_set, forbidden_state)
          %{property: property, satisfied: result, type: :never_reach}
          
        _ ->
          %{property: property, satisfied: false, type: :unknown}
      end
    end)
  end

  def check_deadlock_freedom(protocol_lts) do
    reachable_states = compute_reachable_states(protocol_lts)
    transitions = protocol_lts.transitions
    
    deadlock_states = Enum.filter(reachable_states, fn state ->
      outgoing_transitions = Enum.filter(transitions, fn {{from_state, _action}, _to_state} ->
        from_state == state
      end)
      
      Enum.empty?(outgoing_transitions)
    end)
    
    %{
      deadlock_free: Enum.empty?(deadlock_states),
      deadlock_states: deadlock_states,
      total_reachable: length(reachable_states)
    }
  end

  def check_liveness_properties(protocol_lts, liveness_properties) do
    reachable_states = compute_reachable_states(protocol_lts)
    transitions = protocol_lts.transitions
    
    Enum.map(liveness_properties, fn property ->
      case property do
        {:progress, action} ->
          check_progress_property(reachable_states, transitions, action)
          
        {:fairness, actions} ->
          check_fairness_property(reachable_states, transitions, actions)
          
        _ ->
          %{property: property, satisfied: false, type: :unknown}
      end
    end)
  end

  defp check_progress_property(reachable_states, transitions, action) do
    # Check if the action can always eventually be taken
    states_with_action = Enum.filter(reachable_states, fn state ->
      Enum.any?(transitions, fn {{from_state, trans_action}, _to_state} ->
        from_state == state and trans_action == action
      end)
    end)
    
    progress_ratio = length(states_with_action) / max(1, length(reachable_states))
    
    %{
      property: {:progress, action},
      satisfied: progress_ratio > 0.5,
      progress_ratio: progress_ratio,
      type: :progress
    }
  end

  defp check_fairness_property(reachable_states, transitions, actions) do
    # Check if all actions in the set can be fairly executed
    action_availability = Enum.map(actions, fn action ->
      states_with_action = Enum.count(reachable_states, fn state ->
        Enum.any?(transitions, fn {{from_state, trans_action}, _to_state} ->
          from_state == state and trans_action == action
        end)
      end)
      
      {action, states_with_action / max(1, length(reachable_states))}
    end)
    
    min_availability = action_availability
                      |> Enum.map(fn {_action, ratio} -> ratio end)
                      |> Enum.min(fn -> 0.0 end)
    
    %{
      property: {:fairness, actions},
      satisfied: min_availability > 0.1,
      min_availability: min_availability,
      action_availability: action_availability,
      type: :fairness
    }
  end

  def build_state_graph(protocol_lts) do
    reachable_states = compute_reachable_states(protocol_lts)
    transitions = protocol_lts.transitions
    
    # Build adjacency list representation
    graph = Enum.reduce(reachable_states, %{}, fn state, acc ->
      Map.put(acc, state, [])
    end)
    
    Enum.reduce(transitions, graph, fn {{from_state, action}, to_state}, acc ->
      if Map.has_key?(acc, from_state) do
        current_edges = Map.get(acc, from_state, [])
        new_edges = [{action, to_state} | current_edges]
        Map.put(acc, from_state, new_edges)
      else
        acc
      end
    end)
  end

  def analyze_protocol_completeness(protocol_lts) do
    states = protocol_lts.states
    alphabet = protocol_lts.alphabet
    transitions = protocol_lts.transitions
    
    total_possible_transitions = length(states) * length(alphabet)
    actual_transitions = map_size(transitions)
    
    completeness_ratio = actual_transitions / max(1, total_possible_transitions)
    
    missing_transitions = for state <- states,
                             action <- alphabet,
                             not Map.has_key?(transitions, {state, action}) do
      {state, action}
    end
    
    %{
      completeness_ratio: completeness_ratio,
      total_possible: total_possible_transitions,
      actual_transitions: actual_transitions,
      missing_transitions: missing_transitions,
      is_complete: Enum.empty?(missing_transitions)
    }
  end
end
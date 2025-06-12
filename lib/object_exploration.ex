defmodule Object.Exploration do
  @moduledoc """
  Object-Oriented Exploration strategies for OORL framework.
  
  Implements novelty-based and uncertainty-based exploration as specified
  in AAOS section 10, enabling objects to discover novel interactions
  and configurations in their environment.
  
  Key exploration strategies:
  - Novelty-based exploration using state visitation counts
  - Uncertainty-based exploration using prediction confidence
  - Curiosity-driven exploration with information gain
  - Social exploration through interaction dyads
  - Meta-exploration for learning strategy optimization
  """

  defstruct [
    :object_id,
    :exploration_strategy,
    :novelty_tracker,
    :uncertainty_estimator,
    :curiosity_model,
    :exploration_history,
    :exploration_parameters,
    :social_exploration_state
  ]

  @type t :: %__MODULE__{
    object_id: String.t(),
    exploration_strategy: strategy_type(),
    novelty_tracker: novelty_tracker(),
    uncertainty_estimator: uncertainty_estimator(),
    curiosity_model: curiosity_model(),
    exploration_history: [exploration_record()],
    exploration_parameters: exploration_params(),
    social_exploration_state: social_exploration_state()
  }

  @type strategy_type :: :novelty_based | :uncertainty_based | :curiosity_driven | :hybrid | :social
  
  @type novelty_tracker :: %{
    state_visitation_counts: %{state_key() => integer()},
    novelty_threshold: float(),
    decay_factor: float(),
    novelty_bonus_scale: float()
  }

  @type uncertainty_estimator :: %{
    prediction_errors: [float()],
    confidence_threshold: float(),
    uncertainty_bonus_scale: float(),
    model_uncertainty: float()
  }

  @type curiosity_model :: %{
    information_gain_estimates: %{state_key() => float()},
    surprise_threshold: float(),
    curiosity_bonus_scale: float(),
    learning_progress: float()
  }

  @type exploration_record :: %{
    timestamp: DateTime.t(),
    state: any(),
    action: any(),
    novelty_score: float(),
    uncertainty_score: float(),
    curiosity_score: float(),
    total_exploration_bonus: float(),
    outcome: any()
  }

  @type exploration_params :: %{
    exploration_rate: float(),
    novelty_weight: float(),
    uncertainty_weight: float(),
    curiosity_weight: float(),
    social_weight: float(),
    decay_rate: float()
  }

  @type social_exploration_state :: %{
    interaction_novelty: %{object_id() => float()},
    dyad_exploration_targets: [object_id()],
    social_curiosity_scores: %{object_id() => float()},
    collaboration_history: [collaboration_record()]
  }

  @type collaboration_record :: %{
    partner_id: object_id(),
    exploration_outcome: float(),
    timestamp: DateTime.t()
  }

  @type state_key :: term()
  @type object_id :: String.t()

  @doc """
  Creates a new exploration system for an object.
  
  ## Parameters
  - `object_id`: ID of the object
  - `strategy`: Exploration strategy (`:novelty_based`, `:uncertainty_based`, `:curiosity_driven`, `:hybrid`, `:social`)
  - `opts`: Optional configuration parameters
  
  ## Returns
  New exploration system struct
  
  ## Examples
      iex> Object.Exploration.new("agent1", :hybrid)
      %Object.Exploration{object_id: "agent1", exploration_strategy: :hybrid, ...}
  """
  def new(object_id, strategy \\ :hybrid, opts \\ []) do
    %__MODULE__{
      object_id: object_id,
      exploration_strategy: strategy,
      novelty_tracker: initialize_novelty_tracker(opts),
      uncertainty_estimator: initialize_uncertainty_estimator(opts),
      curiosity_model: initialize_curiosity_model(opts),
      exploration_history: [],
      exploration_parameters: initialize_exploration_parameters(strategy, opts),
      social_exploration_state: initialize_social_exploration_state(opts)
    }
  end

  @doc """
  Computes exploration bonus for a given state-action pair based on the exploration strategy.
  
  ## Parameters
  - `explorer`: Exploration system struct
  - `state`: Current state
  - `action`: Action being considered
  
  ## Returns
  Exploration bonus value (higher values encourage exploration)
  """
  def compute_exploration_bonus(%__MODULE__{} = explorer, state, action) do
    case explorer.exploration_strategy do
      :novelty_based ->
        compute_novelty_bonus(explorer, state, action)
      
      :uncertainty_based ->
        compute_uncertainty_bonus(explorer, state, action)
      
      :curiosity_driven ->
        compute_curiosity_bonus(explorer, state, action)
      
      :social ->
        compute_social_exploration_bonus(explorer, state, action)
      
      :hybrid ->
        compute_hybrid_exploration_bonus(explorer, state, action)
    end
  end

  @doc """
  Selects an action based on exploration strategy and value estimates.
  
  ## Parameters
  - `explorer`: Exploration system struct
  - `available_actions`: List of possible actions
  - `value_estimates`: Map of action -> estimated value
  
  ## Returns
  Selected action that balances exploration and exploitation
  """
  def select_exploration_action(%__MODULE__{} = explorer, available_actions, value_estimates) do
    exploration_bonuses = for action <- available_actions do
      bonus = compute_exploration_bonus(explorer, nil, action)
      {action, bonus}
    end
    
    # Combine value estimates with exploration bonuses
    action_scores = for {action, exploration_bonus} <- exploration_bonuses do
      base_value = Map.get(value_estimates, action, 0.0)
      total_score = base_value + exploration_bonus
      {action, total_score}
    end
    
    # Select action using exploration strategy
    case explorer.exploration_strategy do
      strategy when strategy in [:novelty_based, :uncertainty_based, :curiosity_driven] ->
        epsilon_greedy_selection(action_scores, explorer.exploration_parameters.exploration_rate)
      
      :social ->
        social_exploration_selection(explorer, action_scores)
      
      :hybrid ->
        hybrid_exploration_selection(explorer, action_scores)
    end
  end

  @doc """
  Updates exploration state based on observed outcome.
  
  ## Parameters
  - `explorer`: Exploration system struct
  - `state`: State where action was taken
  - `action`: Action that was executed
  - `outcome`: Observed outcome/result
  
  ## Returns
  Updated exploration system with recorded experience
  """
  def update_exploration_state(%__MODULE__{} = explorer, state, action, outcome) do
    # Record exploration experience
    exploration_record = create_exploration_record(explorer, state, action, outcome)
    updated_history = [exploration_record | explorer.exploration_history]
    
    # Update novelty tracker
    updated_novelty_tracker = update_novelty_tracker(explorer.novelty_tracker, state)
    
    # Update uncertainty estimator
    updated_uncertainty_estimator = update_uncertainty_estimator(explorer.uncertainty_estimator, state, outcome)
    
    # Update curiosity model
    updated_curiosity_model = update_curiosity_model(explorer.curiosity_model, state, action, outcome)
    
    %{explorer |
      exploration_history: updated_history,
      novelty_tracker: updated_novelty_tracker,
      uncertainty_estimator: updated_uncertainty_estimator,
      curiosity_model: updated_curiosity_model
    }
  end

  @doc """
  Identifies novel interaction patterns and opportunities with other objects.
  
  ## Parameters
  - `explorer`: Exploration system struct
  - `available_partners`: List of potential interaction partners
  
  ## Returns
  Map with novel partners, recommendations, and expected information gain
  """
  def identify_novel_interactions(%__MODULE__{} = explorer, available_partners) do
    social_state = explorer.social_exploration_state
    
    novel_partners = for partner_id <- available_partners do
      novelty_score = Map.get(social_state.interaction_novelty, partner_id, 1.0)
      {partner_id, novelty_score}
    end
    |> Enum.filter(fn {_, score} -> score > 0.7 end)
    |> Enum.sort_by(&elem(&1, 1), :desc)
    
    %{
      novel_partners: novel_partners,
      exploration_recommendations: generate_interaction_recommendations(novel_partners),
      expected_information_gain: estimate_social_information_gain(novel_partners)
    }
  end

  @doc """
  Adapts exploration parameters based on performance feedback.
  
  ## Parameters
  - `explorer`: Exploration system struct
  - `performance_metrics`: Recent performance data
  
  ## Returns
  Updated exploration system with adapted parameters
  """
  def adapt_exploration_parameters(%__MODULE__{} = explorer, performance_metrics) do
    current_params = explorer.exploration_parameters
    
    adaptation_adjustments = case analyze_exploration_performance(explorer, performance_metrics) do
      {:increase_exploration} ->
        %{exploration_rate: min(1.0, current_params.exploration_rate * 1.1)}
      
      {:decrease_exploration} ->
        %{exploration_rate: max(0.01, current_params.exploration_rate * 0.9)}
      
      {:balance_strategies} ->
        balance_exploration_weights(current_params, performance_metrics)
      
      {:no_change} ->
        %{}
    end
    
    updated_params = Map.merge(current_params, adaptation_adjustments)
    %{explorer | exploration_parameters: updated_params}
  end

  @doc """
  Evaluates the effectiveness of the current exploration strategy.
  
  ## Parameters
  - `explorer`: Exploration system struct
  
  ## Returns
  Map with overall effectiveness score, detailed metrics, and recommendations
  """
  def evaluate_exploration_effectiveness(%__MODULE__{} = explorer) do
    recent_history = Enum.take(explorer.exploration_history, 100)
    
    metrics = %{
      novelty_discovery_rate: calculate_novelty_discovery_rate(recent_history),
      uncertainty_reduction_rate: calculate_uncertainty_reduction_rate(recent_history),
      information_gain_rate: calculate_information_gain_rate(recent_history),
      exploration_efficiency: calculate_exploration_efficiency(recent_history),
      social_exploration_success: calculate_social_exploration_success(explorer)
    }
    
    overall_effectiveness = aggregate_exploration_metrics(metrics)
    
    %{
      overall_effectiveness: overall_effectiveness,
      detailed_metrics: metrics,
      recommendations: generate_exploration_recommendations(metrics)
    }
  end

  # Private implementation functions

  defp initialize_novelty_tracker(opts) do
    %{
      state_visitation_counts: %{},
      novelty_threshold: Keyword.get(opts, :novelty_threshold, 0.1),
      decay_factor: Keyword.get(opts, :novelty_decay, 0.99),
      novelty_bonus_scale: Keyword.get(opts, :novelty_bonus_scale, 1.0)
    }
  end

  defp initialize_uncertainty_estimator(opts) do
    %{
      prediction_errors: [],
      confidence_threshold: Keyword.get(opts, :confidence_threshold, 0.8),
      uncertainty_bonus_scale: Keyword.get(opts, :uncertainty_bonus_scale, 1.0),
      model_uncertainty: Keyword.get(opts, :initial_uncertainty, 0.5)
    }
  end

  defp initialize_curiosity_model(opts) do
    %{
      information_gain_estimates: %{},
      surprise_threshold: Keyword.get(opts, :surprise_threshold, 0.3),
      curiosity_bonus_scale: Keyword.get(opts, :curiosity_bonus_scale, 1.0),
      learning_progress: Keyword.get(opts, :initial_learning_progress, 0.0)
    }
  end

  defp initialize_exploration_parameters(strategy, opts) do
    base_params = %{
      exploration_rate: Keyword.get(opts, :exploration_rate, 0.1),
      novelty_weight: Keyword.get(opts, :novelty_weight, 0.3),
      uncertainty_weight: Keyword.get(opts, :uncertainty_weight, 0.3),
      curiosity_weight: Keyword.get(opts, :curiosity_weight, 0.3),
      social_weight: Keyword.get(opts, :social_weight, 0.1),
      decay_rate: Keyword.get(opts, :decay_rate, 0.995)
    }
    
    case strategy do
      :novelty_based ->
        %{base_params | novelty_weight: 0.8, uncertainty_weight: 0.1, curiosity_weight: 0.1}
      
      :uncertainty_based ->
        %{base_params | novelty_weight: 0.1, uncertainty_weight: 0.8, curiosity_weight: 0.1}
      
      :curiosity_driven ->
        %{base_params | novelty_weight: 0.1, uncertainty_weight: 0.1, curiosity_weight: 0.8}
      
      :social ->
        %{base_params | social_weight: 0.6, novelty_weight: 0.2, uncertainty_weight: 0.1, curiosity_weight: 0.1}
      
      :hybrid ->
        base_params
    end
  end

  defp initialize_social_exploration_state(_opts) do
    %{
      interaction_novelty: %{},
      dyad_exploration_targets: [],
      social_curiosity_scores: %{},
      collaboration_history: []
    }
  end

  defp compute_novelty_bonus(explorer, state, _action) do
    state_key = encode_state(state)
    visit_count = Map.get(explorer.novelty_tracker.state_visitation_counts, state_key, 0)
    
    # Inverse visitation count bonus
    novelty_score = 1.0 / (1.0 + visit_count)
    novelty_score * explorer.exploration_parameters.novelty_weight
  end

  defp compute_uncertainty_bonus(explorer, _state, _action) do
    # Use model uncertainty as exploration bonus
    uncertainty_score = explorer.uncertainty_estimator.model_uncertainty
    uncertainty_score * explorer.exploration_parameters.uncertainty_weight
  end

  defp compute_curiosity_bonus(explorer, state, action) do
    state_key = encode_state(state)
    action_key = encode_action(action)
    
    # Estimate information gain for this state-action pair
    info_gain = Map.get(explorer.curiosity_model.information_gain_estimates, {state_key, action_key}, 0.5)
    info_gain * explorer.exploration_parameters.curiosity_weight
  end

  defp compute_social_exploration_bonus(explorer, _state, action) do
    # Bonus for exploring novel social interactions
    case extract_social_action(action) do
      {:interact, partner_id} ->
        novelty_score = Map.get(explorer.social_exploration_state.interaction_novelty, partner_id, 1.0)
        novelty_score * explorer.exploration_parameters.social_weight
      
      _ ->
        0.0
    end
  end

  defp compute_hybrid_exploration_bonus(explorer, state, action) do
    novelty_bonus = compute_novelty_bonus(explorer, state, action)
    uncertainty_bonus = compute_uncertainty_bonus(explorer, state, action)
    curiosity_bonus = compute_curiosity_bonus(explorer, state, action)
    social_bonus = compute_social_exploration_bonus(explorer, state, action)
    
    novelty_bonus + uncertainty_bonus + curiosity_bonus + social_bonus
  end

  defp epsilon_greedy_selection(action_scores, epsilon) do
    if :rand.uniform() < epsilon do
      # Random exploration
      {action, _} = Enum.random(action_scores)
      action
    else
      # Greedy selection
      {action, _} = Enum.max_by(action_scores, &elem(&1, 1))
      action
    end
  end

  defp social_exploration_selection(explorer, action_scores) do
    # Prioritize actions that lead to novel social interactions
    social_actions = Enum.filter(action_scores, fn {action, _} ->
      is_social_action?(action)
    end)
    
    if length(social_actions) > 0 do
      {action, _} = Enum.max_by(social_actions, &elem(&1, 1))
      action
    else
      epsilon_greedy_selection(action_scores, explorer.exploration_parameters.exploration_rate)
    end
  end

  defp hybrid_exploration_selection(explorer, action_scores) do
    # Weighted combination of different exploration criteria
    epsilon_greedy_selection(action_scores, explorer.exploration_parameters.exploration_rate)
  end

  defp create_exploration_record(explorer, state, action, outcome) do
    %{
      timestamp: DateTime.utc_now(),
      state: state,
      action: action,
      novelty_score: compute_novelty_bonus(explorer, state, action),
      uncertainty_score: compute_uncertainty_bonus(explorer, state, action),
      curiosity_score: compute_curiosity_bonus(explorer, state, action),
      total_exploration_bonus: compute_exploration_bonus(explorer, state, action),
      outcome: outcome
    }
  end

  defp update_novelty_tracker(tracker, state) do
    state_key = encode_state(state)
    current_count = Map.get(tracker.state_visitation_counts, state_key, 0)
    
    updated_counts = Map.put(tracker.state_visitation_counts, state_key, current_count + 1)
    
    # Apply decay to all counts
    decayed_counts = Map.new(updated_counts, fn {key, count} ->
      {key, count * tracker.decay_factor}
    end)
    
    %{tracker | state_visitation_counts: decayed_counts}
  end

  defp update_uncertainty_estimator(estimator, state, outcome) do
    # Update prediction errors and model uncertainty
    prediction_error = calculate_prediction_error(state, outcome)
    updated_errors = [prediction_error | Enum.take(estimator.prediction_errors, 99)]
    
    # Update model uncertainty based on recent prediction errors
    recent_errors = Enum.take(updated_errors, 20)
    new_uncertainty = if length(recent_errors) > 0 do
      Enum.sum(recent_errors) / length(recent_errors)
    else
      estimator.model_uncertainty
    end
    
    %{estimator |
      prediction_errors: updated_errors,
      model_uncertainty: new_uncertainty
    }
  end

  defp update_curiosity_model(model, state, action, outcome) do
    state_key = encode_state(state)
    action_key = encode_action(action)
    
    # Calculate information gain from this experience
    info_gain = calculate_information_gain(state, action, outcome)
    
    # Update information gain estimates
    updated_estimates = Map.put(model.information_gain_estimates, {state_key, action_key}, info_gain)
    
    # Update learning progress
    new_learning_progress = (model.learning_progress + info_gain) / 2.0
    
    %{model |
      information_gain_estimates: updated_estimates,
      learning_progress: new_learning_progress
    }
  end

  defp analyze_exploration_performance(explorer, performance_metrics) do
    recent_performance = Map.get(performance_metrics, :recent_performance, 0.5)
    exploration_rate = explorer.exploration_parameters.exploration_rate
    
    cond do
      recent_performance < 0.3 and exploration_rate < 0.2 ->
        {:increase_exploration}
      
      recent_performance > 0.8 and exploration_rate > 0.3 ->
        {:decrease_exploration}
      
      recent_performance < 0.6 ->
        {:balance_strategies}
      
      true ->
        {:no_change}
    end
  end

  defp balance_exploration_weights(_params, performance_metrics) do
    # Adjust weights based on which exploration strategies are most effective
    novelty_effectiveness = Map.get(performance_metrics, :novelty_effectiveness, 0.5)
    uncertainty_effectiveness = Map.get(performance_metrics, :uncertainty_effectiveness, 0.5)
    curiosity_effectiveness = Map.get(performance_metrics, :curiosity_effectiveness, 0.5)
    
    total_effectiveness = novelty_effectiveness + uncertainty_effectiveness + curiosity_effectiveness
    
    if total_effectiveness > 0 do
      %{
        novelty_weight: novelty_effectiveness / total_effectiveness,
        uncertainty_weight: uncertainty_effectiveness / total_effectiveness,
        curiosity_weight: curiosity_effectiveness / total_effectiveness
      }
    else
      %{}
    end
  end

  # Simplified helper functions for demo

  defp encode_state(state) do
    :erlang.phash2(state)
  end

  defp encode_action(action) do
    :erlang.phash2(action)
  end

  defp extract_social_action(action) do
    case action do
      {:interact, partner_id} -> {:interact, partner_id}
      _ -> :not_social
    end
  end

  defp is_social_action?(action) do
    case extract_social_action(action) do
      {:interact, _} -> true
      _ -> false
    end
  end

  defp calculate_prediction_error(_state, _outcome) do
    # Simplified prediction error calculation
    :rand.uniform() * 0.5
  end

  defp calculate_information_gain(_state, _action, _outcome) do
    # Simplified information gain calculation
    :rand.uniform() * 0.3
  end

  defp generate_interaction_recommendations(novel_partners) do
    for {partner_id, novelty_score} <- Enum.take(novel_partners, 3) do
      "Consider exploring interaction with #{partner_id} (novelty: #{Float.round(novelty_score, 2)})"
    end
  end

  defp estimate_social_information_gain(novel_partners) do
    novel_partners
    |> Enum.map(&elem(&1, 1))
    |> Enum.sum()
  end

  defp calculate_novelty_discovery_rate(history) do
    if length(history) > 0 do
      novel_experiences = Enum.count(history, &(&1.novelty_score > 0.5))
      novel_experiences / length(history)
    else
      0.0
    end
  end

  defp calculate_uncertainty_reduction_rate(history) do
    if length(history) > 1 do
      uncertainty_scores = Enum.map(history, & &1.uncertainty_score)
      first_half = Enum.take(uncertainty_scores, div(length(uncertainty_scores), 2))
      second_half = Enum.drop(uncertainty_scores, div(length(uncertainty_scores), 2))
      
      avg_first = if length(first_half) > 0, do: Enum.sum(first_half) / length(first_half), else: 0.5
      avg_second = if length(second_half) > 0, do: Enum.sum(second_half) / length(second_half), else: 0.5
      
      max(0.0, avg_first - avg_second)
    else
      0.0
    end
  end

  defp calculate_information_gain_rate(history) do
    if length(history) > 0 do
      Enum.map(history, & &1.curiosity_score)
      |> Enum.sum()
      |> Kernel./(length(history))
    else
      0.0
    end
  end

  defp calculate_exploration_efficiency(history) do
    if length(history) > 0 do
      total_bonus = Enum.map(history, & &1.total_exploration_bonus) |> Enum.sum()
      total_bonus / length(history)
    else
      0.0
    end
  end

  defp calculate_social_exploration_success(explorer) do
    collaboration_history = explorer.social_exploration_state.collaboration_history
    
    if length(collaboration_history) > 0 do
      successful_collaborations = Enum.count(collaboration_history, &(&1.exploration_outcome > 0.5))
      successful_collaborations / length(collaboration_history)
    else
      0.0
    end
  end

  defp aggregate_exploration_metrics(metrics) do
    weights = %{
      novelty_discovery_rate: 0.25,
      uncertainty_reduction_rate: 0.2,
      information_gain_rate: 0.25,
      exploration_efficiency: 0.2,
      social_exploration_success: 0.1
    }
    
    Enum.reduce(metrics, 0.0, fn {metric, value}, acc ->
      weight = Map.get(weights, metric, 0.0)
      acc + (value * weight)
    end)
  end

  defp generate_exploration_recommendations(metrics) do
    recommendations = []
    
    recommendations = if metrics.novelty_discovery_rate < 0.3 do
      ["Increase novelty-based exploration" | recommendations]
    else
      recommendations
    end
    
    recommendations = if metrics.uncertainty_reduction_rate < 0.2 do
      ["Focus on uncertainty reduction strategies" | recommendations]
    else
      recommendations
    end
    
    recommendations = if metrics.social_exploration_success < 0.4 do
      ["Improve social exploration targeting" | recommendations]
    else
      recommendations
    end
    
    if length(recommendations) == 0 do
      ["Exploration strategy performing well"]
    else
      recommendations
    end
  end
end
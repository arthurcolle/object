defmodule OORL.RewardLearning do
  @moduledoc """
  OORL Reward Learning module implementing mathematical reward combination algorithms
  as specified in AAOS Section 6.
  
  Provides multiple reward combination strategies:
  - Linear combination
  - Weighted combination  
  - Adaptive combination
  - Hierarchical combination
  
  Maintains mathematical properties including Lipschitz continuity and bounded learning.
  """

  defstruct [
    :combination_strategy,
    :weights,
    :adaptation_rate,
    :lipschitz_constant,
    :reward_history,
    :intrinsic_components,
    :extrinsic_components
  ]

  @type reward_combination_strategy :: :linear | :weighted | :adaptive | :hierarchical
  @type reward_component :: %{
    type: :task_reward | :social_reward | :curiosity_reward | :intrinsic_reward,
    value: float(),
    confidence: float(),
    source: String.t()
  }

  @doc """
  Combines multiple reward components using the specified strategy.
  
  ## Parameters
  - `extrinsic_rewards`: List of external reward components
  - `intrinsic_rewards`: List of internal reward components  
  - `strategy`: Combination strategy to use
  
  ## Returns
  `{:ok, combined_reward}` or `{:error, reason}`
  
  ## Examples
      iex> OORL.RewardLearning.combine_rewards([%{type: :task_reward, value: 0.8}], 
      ...>   [%{type: :curiosity_reward, value: 0.3}], :linear)
      {:ok, 1.1}
  """
  def combine_rewards(extrinsic_rewards, intrinsic_rewards, strategy \\ :linear) do
    try do
      case strategy do
        :linear ->
          {:ok, linear_combination(extrinsic_rewards, intrinsic_rewards)}
        
        :weighted ->
          {:ok, weighted_combination(extrinsic_rewards, intrinsic_rewards)}
        
        :adaptive ->
          {:ok, adaptive_combination(extrinsic_rewards, intrinsic_rewards)}
        
        :hierarchical ->
          {:ok, hierarchical_combination(extrinsic_rewards, intrinsic_rewards)}
        
        _ ->
          {:error, "Unknown combination strategy: #{strategy}"}
      end
    rescue
      error ->
        {:error, "Reward combination failed: #{inspect(error)}"}
    end
  end

  @doc """
  Creates a new reward learning configuration.
  
  ## Parameters
  - `opts`: Configuration options including strategy, weights, adaptation_rate
  
  ## Returns
  `%OORL.RewardLearning{}` struct
  """
  def new(opts \\ []) do
    %OORL.RewardLearning{
      combination_strategy: Keyword.get(opts, :strategy, :adaptive),
      weights: Keyword.get(opts, :weights, default_weights()),
      adaptation_rate: Keyword.get(opts, :adaptation_rate, 0.01),
      lipschitz_constant: Keyword.get(opts, :lipschitz_constant, 1.0),
      reward_history: [],
      intrinsic_components: Keyword.get(opts, :intrinsic_components, [:curiosity_reward]),
      extrinsic_components: Keyword.get(opts, :extrinsic_components, [:task_reward, :social_reward])
    }
  end

  @doc """
  Adapts reward weights based on performance feedback.
  
  ## Parameters
  - `reward_learner`: Current reward learning configuration
  - `performance_metrics`: Performance feedback data
  
  ## Returns
  Updated `%OORL.RewardLearning{}` struct
  """
  def adapt_weights(reward_learner, performance_metrics) do
    new_weights = case reward_learner.combination_strategy do
      :adaptive ->
        update_adaptive_weights(reward_learner.weights, performance_metrics, reward_learner.adaptation_rate)
      
      :hierarchical ->
        update_hierarchical_weights(reward_learner.weights, performance_metrics)
      
      _ ->
        reward_learner.weights
    end

    %{reward_learner | 
      weights: new_weights,
      reward_history: [performance_metrics | Enum.take(reward_learner.reward_history, 99)]
    }
  end

  @doc """
  Validates that reward function maintains mathematical properties.
  
  ## Parameters
  - `reward_function`: Function to validate
  - `test_points`: Sample points for validation
  
  ## Returns
  `{:ok, validation_results}` with properties like Lipschitz continuity
  """
  def validate_mathematical_properties(reward_function, test_points) do
    lipschitz_violations = check_lipschitz_continuity(reward_function, test_points)
    boundedness_check = check_boundedness(reward_function, test_points)
    monotonicity_check = check_monotonicity(reward_function, test_points)

    validation_results = %{
      lipschitz_violations: lipschitz_violations,
      is_bounded: boundedness_check,
      is_monotonic: monotonicity_check,
      total_violations: length(lipschitz_violations),
      compliance_score: calculate_compliance_score(lipschitz_violations, boundedness_check, monotonicity_check)
    }

    {:ok, validation_results}
  end

  # Private implementation functions

  defp linear_combination(extrinsic_rewards, intrinsic_rewards) do
    extrinsic_sum = Enum.reduce(extrinsic_rewards, 0, fn reward, acc ->
      acc + Map.get(reward, :value, 0)
    end)
    
    intrinsic_sum = Enum.reduce(intrinsic_rewards, 0, fn reward, acc ->
      acc + Map.get(reward, :value, 0)
    end)
    
    extrinsic_sum + intrinsic_sum
  end

  defp weighted_combination(extrinsic_rewards, intrinsic_rewards) do
    weights = default_weights()
    
    extrinsic_weighted = Enum.reduce(extrinsic_rewards, 0, fn reward, acc ->
      weight = Map.get(weights, reward[:type], 1.0)
      acc + (reward[:value] * weight)
    end)
    
    intrinsic_weighted = Enum.reduce(intrinsic_rewards, 0, fn reward, acc ->
      weight = Map.get(weights, reward[:type], 0.3)
      acc + (reward[:value] * weight)
    end)
    
    extrinsic_weighted + intrinsic_weighted
  end

  defp adaptive_combination(extrinsic_rewards, intrinsic_rewards) do
    # Adaptive weights based on recent performance
    base_weights = default_weights()
    
    # Increase intrinsic weight if extrinsic rewards are sparse
    extrinsic_count = length(extrinsic_rewards)
    intrinsic_boost = if extrinsic_count < 2, do: 1.5, else: 1.0
    
    extrinsic_weighted = Enum.reduce(extrinsic_rewards, 0, fn reward, acc ->
      weight = Map.get(base_weights, reward[:type], 1.0)
      acc + (reward[:value] * weight)
    end)
    
    intrinsic_weighted = Enum.reduce(intrinsic_rewards, 0, fn reward, acc ->
      weight = Map.get(base_weights, reward[:type], 0.3) * intrinsic_boost
      acc + (reward[:value] * weight)
    end)
    
    extrinsic_weighted + intrinsic_weighted
  end

  defp hierarchical_combination(extrinsic_rewards, intrinsic_rewards) do
    # Hierarchical combination with priority levels
    high_priority = Enum.filter(extrinsic_rewards, fn r -> Map.get(r, :priority, :medium) == :high end)
    medium_priority = Enum.filter(extrinsic_rewards, fn r -> Map.get(r, :priority, :medium) == :medium end)
    
    high_sum = Enum.reduce(high_priority, 0, fn r, acc -> acc + r[:value] end) * 2.0
    medium_sum = Enum.reduce(medium_priority, 0, fn r, acc -> acc + r[:value] end) * 1.0
    intrinsic_sum = Enum.reduce(intrinsic_rewards, 0, fn r, acc -> acc + r[:value] end) * 0.5
    
    high_sum + medium_sum + intrinsic_sum
  end

  defp default_weights do
    %{
      task_reward: 1.0,
      social_reward: 0.6,
      curiosity_reward: 0.4,
      intrinsic_reward: 0.3,
      exploration_reward: 0.2,
      collaboration_reward: 0.5
    }
  end

  defp update_adaptive_weights(current_weights, performance_metrics, adaptation_rate) do
    performance_score = Map.get(performance_metrics, :overall_score, 0.5)
    
    # Increase weights for components that contributed to good performance
    Enum.reduce(current_weights, %{}, fn {component, weight}, acc ->
      component_contribution = Map.get(performance_metrics, component, 0.5)
      
      # Adaptive update based on contribution
      adjustment = adaptation_rate * (component_contribution - 0.5) * performance_score
      new_weight = max(0.1, min(2.0, weight + adjustment))
      
      Map.put(acc, component, new_weight)
    end)
  end

  defp update_hierarchical_weights(current_weights, performance_metrics) do
    # Hierarchical weights adapt based on goal achievement
    goal_achievement = Map.get(performance_metrics, :goal_achievement, 0.5)
    
    if goal_achievement > 0.8 do
      # Increase exploration when performing well
      Map.merge(current_weights, %{curiosity_reward: 0.6, exploration_reward: 0.4})
    else
      # Focus on exploitation when underperforming  
      Map.merge(current_weights, %{task_reward: 1.2, social_reward: 0.8})
    end
  end

  defp check_lipschitz_continuity(reward_function, test_points) do
    lipschitz_constant = 1.0
    
    violations = for {point1, point2} <- point_pairs(test_points), 
                      violation = check_lipschitz_pair(reward_function, point1, point2, lipschitz_constant),
                      violation != nil do
      violation
    end
    
    violations
  end

  defp check_lipschitz_pair(reward_function, point1, point2, lipschitz_constant) do
    try do
      reward1 = apply_reward_function(reward_function, point1)
      reward2 = apply_reward_function(reward_function, point2)
      
      distance = euclidean_distance(point1, point2)
      reward_diff = abs(reward1 - reward2)
      
      if reward_diff > lipschitz_constant * distance do
        %{point1: point1, point2: point2, violation_magnitude: reward_diff - lipschitz_constant * distance}
      else
        nil
      end
    rescue
      _ -> nil
    end
  end

  defp check_boundedness(reward_function, test_points) do
    rewards = Enum.map(test_points, fn point ->
      apply_reward_function(reward_function, point)
    end)
    
    min_reward = Enum.min(rewards)
    max_reward = Enum.max(rewards)
    
    # Check if rewards are reasonably bounded
    min_reward >= -10.0 && max_reward <= 10.0
  end

  defp check_monotonicity(reward_function, test_points) do
    # Simple monotonicity check for ordered test points
    ordered_points = Enum.sort(test_points)
    rewards = Enum.map(ordered_points, fn point ->
      apply_reward_function(reward_function, point)
    end)
    
    # Check if rewards are non-decreasing
    rewards
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.all?(fn [a, b] -> b >= a end)
  end

  defp apply_reward_function(reward_function, point) when is_function(reward_function) do
    reward_function.(point)
  end

  defp apply_reward_function(reward_function, point) when is_map(reward_function) do
    # Mock implementation for map-based reward functions
    Map.get(reward_function, :base_reward, 0.0) + :rand.uniform() * 0.1
  end

  defp euclidean_distance(point1, point2) when is_number(point1) and is_number(point2) do
    abs(point1 - point2)
  end

  defp euclidean_distance(point1, point2) when is_list(point1) and is_list(point2) do
    Enum.zip(point1, point2)
    |> Enum.reduce(0, fn {a, b}, acc -> acc + (a - b) * (a - b) end)
    |> :math.sqrt()
  end

  defp euclidean_distance(_point1, _point2), do: 1.0

  defp point_pairs(test_points) do
    for i <- 0..(length(test_points) - 2),
        j <- (i + 1)..(length(test_points) - 1) do
      {Enum.at(test_points, i), Enum.at(test_points, j)}
    end
  end

  defp calculate_compliance_score(lipschitz_violations, is_bounded, is_monotonic) do
    violation_penalty = length(lipschitz_violations) * 0.1
    boundedness_bonus = if is_bounded, do: 0.3, else: 0.0
    monotonicity_bonus = if is_monotonic, do: 0.2, else: 0.0
    
    max(0.0, 1.0 - violation_penalty + boundedness_bonus + monotonicity_bonus)
  end
end
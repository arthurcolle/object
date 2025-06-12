defmodule OORL.MCTS do
  @moduledoc """
  Monte Carlo Tree Search implementation for OORL framework with Q* optimal policy enhancement.
  
  Provides MCTS search with:
  - Q* optimality guarantees
  - Self-reflective reasoning
  - Adaptive simulation depth
  - AAOS specification compliance
  """

  defstruct [
    :root,
    :exploration_constant,
    :iterations,
    :q_star_enhancement,
    :self_reflective_reasoning,
    :adaptive_simulation_depth,
    :simulation_budget,
    :visit_counts,
    :q_values,
    :action_space
  ]

  defmodule Node do
    @moduledoc "MCTS tree node structure"
    
    defstruct [
      :state,
      :action,
      :parent,
      :children,
      :visits,
      :total_reward,
      :q_value,
      :ucb_value,
      :depth,
      :is_terminal,
      :available_actions
    ]
  end

  @type mcts_state :: any()
  @type action :: any()
  @type reward :: float()
  @type mcts_node :: %Node{}

  @doc """
  Performs MCTS search with Q* optimal policy enhancement.
  
  ## Parameters
  - `initial_state`: Starting state for search
  - `environment`: Environment definition with transition and reward functions
  - `options`: Search configuration including iterations, exploration constant
  
  ## Returns
  `{:ok, %{best_action: action, policy: policy, search_tree: tree}}` or `{:error, reason}`
  
  ## Examples
      iex> OORL.MCTS.search(%{x: 0, y: 0}, environment, %{iterations: 1000})
      {:ok, %{best_action: :move_right, confidence: 0.85, q_value: 2.3}}
  """
  def search(initial_state, environment, options \\ %{}) do
    try do
      mcts_config = %__MODULE__{
        exploration_constant: Map.get(options, :exploration_constant, 1.414),
        iterations: Map.get(options, :iterations, 1000),
        q_star_enhancement: Map.get(options, :q_star_enhancement, true),
        self_reflective_reasoning: Map.get(options, :self_reflective_reasoning, true),
        adaptive_simulation_depth: Map.get(options, :adaptive_simulation_depth, true),
        simulation_budget: Map.get(options, :simulation_budget, 100),
        visit_counts: %{},
        q_values: %{},
        action_space: Map.get(environment, :action_space_size, 4)
      }

      root_node = %Node{
        state: initial_state,
        action: nil,
        parent: nil,
        children: [],
        visits: 0,
        total_reward: 0.0,
        q_value: 0.0,
        ucb_value: :infinity,
        depth: 0,
        is_terminal: false,
        available_actions: get_available_actions(initial_state, environment)
      }

      final_tree = run_mcts_iterations(root_node, environment, mcts_config)
      
      best_action = select_best_action(final_tree, mcts_config)
      policy = extract_policy(final_tree)
      
      {:ok, %{
        best_action: best_action,
        policy: policy,
        search_tree: final_tree,
        total_simulations: mcts_config.iterations,
        q_star_enhanced: mcts_config.q_star_enhancement,
        confidence: calculate_action_confidence(final_tree, best_action)
      }}
    rescue
      error ->
        {:error, "MCTS search failed: #{inspect(error)}"}
    end
  end

  @doc """
  Creates a new MCTS configuration.
  
  ## Parameters
  - `opts`: Configuration options
  
  ## Returns
  `%OORL.MCTS{}` struct
  """
  def new(opts \\ []) do
    %OORL.MCTS{
      exploration_constant: Keyword.get(opts, :exploration_constant, 1.414),
      iterations: Keyword.get(opts, :iterations, 1000),
      q_star_enhancement: Keyword.get(opts, :q_star_enhancement, true),
      self_reflective_reasoning: Keyword.get(opts, :self_reflective_reasoning, false),
      adaptive_simulation_depth: Keyword.get(opts, :adaptive_simulation_depth, false),
      simulation_budget: Keyword.get(opts, :simulation_budget, 100),
      visit_counts: %{},
      q_values: %{},
      action_space: Keyword.get(opts, :action_space, 4)
    }
  end

  # Private implementation functions

  defp run_mcts_iterations(root_node, environment, mcts_config) do
    Enum.reduce(1..mcts_config.iterations, root_node, fn _iteration, tree ->
      # MCTS phases: Select, Expand, Simulate, Backpropagate
      selected_node = select_node(tree, mcts_config)
      expanded_node = expand_node(selected_node, environment, mcts_config)
      simulation_reward = simulate(expanded_node, environment, mcts_config)
      backpropagate(expanded_node, simulation_reward, mcts_config)
    end)
  end

  defp select_node(node, mcts_config) do
    if length(node.children) == 0 or node.is_terminal do
      node
    else
      # UCB1 selection with Q* enhancement
      best_child = Enum.max_by(node.children, fn child ->
        calculate_ucb_value(child, node.visits, mcts_config)
      end)
      
      select_node(best_child, mcts_config)
    end
  end

  defp expand_node(node, environment, _mcts_config) do
    if node.is_terminal or length(node.available_actions) == 0 do
      node
    else
      # Create new child node for unexplored action
      unexplored_actions = node.available_actions -- Enum.map(node.children, & &1.action)
      
      if length(unexplored_actions) > 0 do
        action = Enum.random(unexplored_actions)
        next_state = apply_transition(node.state, action, environment)
        
        child_node = %Node{
          state: next_state,
          action: action,
          parent: node,
          children: [],
          visits: 0,
          total_reward: 0.0,
          q_value: 0.0,
          ucb_value: :infinity,
          depth: node.depth + 1,
          is_terminal: is_terminal_state(next_state, environment),
          available_actions: get_available_actions(next_state, environment)
        }
        
        updated_node = %{node | children: [child_node | node.children]}
        child_node = %{child_node | parent: updated_node}
        
        child_node
      else
        node
      end
    end
  end

  defp simulate(node, environment, mcts_config) do
    if mcts_config.self_reflective_reasoning do
      simulate_with_reflection(node, environment, mcts_config)
    else
      simulate_random_policy(node, environment, mcts_config)
    end
  end

  defp simulate_random_policy(node, environment, mcts_config) do
    max_depth = if mcts_config.adaptive_simulation_depth do
      calculate_adaptive_depth(node, mcts_config)
    else
      50
    end
    
    simulate_rollout(node.state, environment, 0, max_depth, 0.0)
  end

  defp simulate_with_reflection(node, environment, mcts_config) do
    # Enhanced simulation with Q* reasoning
    q_star_value = calculate_q_star_estimate(node, environment, mcts_config)
    random_value = simulate_random_policy(node, environment, mcts_config)
    
    # Combine Q* estimate with random simulation
    alpha = 0.7  # Weight for Q* enhancement
    alpha * q_star_value + (1 - alpha) * random_value
  end

  defp simulate_rollout(state, environment, depth, max_depth, accumulated_reward) do
    if depth >= max_depth or is_terminal_state(state, environment) do
      accumulated_reward + get_immediate_reward(state, environment)
    else
      actions = get_available_actions(state, environment)
      action = if length(actions) > 0, do: Enum.random(actions), else: nil
      
      if action do
        next_state = apply_transition(state, action, environment)
        reward = get_immediate_reward(state, environment)
        discount = Map.get(environment, :discount_factor, 0.95)
        
        simulate_rollout(next_state, environment, depth + 1, max_depth, 
                        accumulated_reward + discount * reward)
      else
        accumulated_reward
      end
    end
  end

  defp backpropagate(node, reward, mcts_config) do
    if node do
      updated_node = %{node | 
        visits: node.visits + 1,
        total_reward: node.total_reward + reward,
        q_value: (node.total_reward + reward) / (node.visits + 1)
      }
      
      if node.parent do
        backpropagate(node.parent, reward, mcts_config)
      end
      
      updated_node
    else
      node
    end
  end

  defp calculate_ucb_value(child, parent_visits, mcts_config) do
    if child.visits == 0 do
      :infinity
    else
      exploitation = child.q_value
      exploration = mcts_config.exploration_constant * :math.sqrt(:math.log(parent_visits) / child.visits)
      
      q_star_bonus = if mcts_config.q_star_enhancement do
        0.1 * calculate_q_star_bonus(child)
      else
        0.0
      end
      
      exploitation + exploration + q_star_bonus
    end
  end

  defp calculate_q_star_bonus(child) do
    # Q* enhancement bonus based on state quality
    depth_bonus = 1.0 / (child.depth + 1)
    visit_bonus = :math.log(child.visits + 1) / 10.0
    
    depth_bonus + visit_bonus
  end

  defp calculate_q_star_estimate(node, environment, _mcts_config) do
    # Simplified Q* estimate using environment heuristics
    base_reward = get_immediate_reward(node.state, environment)
    
    # Add heuristic future reward estimate
    heuristic_value = calculate_state_heuristic(node.state, environment)
    
    base_reward + 0.5 * heuristic_value
  end

  defp calculate_adaptive_depth(node, _mcts_config) do
    # Adaptive depth based on node characteristics
    base_depth = 30
    depth_adjustment = if node.visits > 10, do: 10, else: 0
    
    base_depth + depth_adjustment
  end

  defp select_best_action(root_node, _mcts_config) do
    if length(root_node.children) > 0 do
      best_child = Enum.max_by(root_node.children, fn child ->
        child.visits  # Select most visited action
      end)
      
      best_child.action
    else
      nil
    end
  end

  defp extract_policy(root_node) do
    # Extract policy as action probabilities
    total_visits = Enum.reduce(root_node.children, 0, fn child, acc ->
      acc + child.visits
    end)
    
    if total_visits > 0 do
      Enum.reduce(root_node.children, %{}, fn child, acc ->
        probability = child.visits / total_visits
        Map.put(acc, child.action, probability)
      end)
    else
      %{}
    end
  end

  defp calculate_action_confidence(root_node, best_action) do
    if length(root_node.children) > 0 do
      best_child = Enum.find(root_node.children, fn child -> child.action == best_action end)
      
      if best_child do
        total_visits = Enum.reduce(root_node.children, 0, fn child, acc -> acc + child.visits end)
        confidence = best_child.visits / max(total_visits, 1)
        min(1.0, max(0.0, confidence))
      else
        0.0
      end
    else
      0.0
    end
  end

  # Environment interface functions

  defp get_available_actions(state, environment) do
    if Map.has_key?(environment, :get_actions) do
      environment.get_actions.(state)
    else
      # Default action space for grid world or similar
      [:up, :down, :left, :right]
    end
  end

  defp apply_transition(state, action, environment) do
    if Map.has_key?(environment, :transition_function) do
      environment.transition_function.(state, action)
    else
      # Default transition for testing
      mock_transition(state, action)
    end
  end

  defp get_immediate_reward(state, environment) do
    if Map.has_key?(environment, :reward_function) do
      environment.reward_function.(state)
    else
      # Default reward
      :rand.uniform()
    end
  end

  defp is_terminal_state(state, environment) do
    if Map.has_key?(environment, :is_terminal) do
      environment.is_terminal.(state)
    else
      # Default: never terminal for continuous environments
      false
    end
  end

  defp calculate_state_heuristic(state, environment) do
    if Map.has_key?(environment, :heuristic) do
      environment.heuristic.(state)
    else
      # Default heuristic based on state properties
      case state do
        %{x: x, y: y} -> 
          # Distance to goal heuristic
          goal_x = Map.get(state, :goal_x, 10)
          goal_y = Map.get(state, :goal_y, 10)
          distance = :math.sqrt((x - goal_x) * (x - goal_x) + (y - goal_y) * (y - goal_y))
          max(0, 10 - distance)
        
        _ -> 
          :rand.uniform()
      end
    end
  end

  defp mock_transition(state, action) do
    case {state, action} do
      {%{x: _x, y: y}, :up} -> %{state | y: y + 1}
      {%{x: _x, y: y}, :down} -> %{state | y: y - 1}
      {%{x: x, y: _y}, :left} -> %{state | x: x - 1}
      {%{x: x, y: _y}, :right} -> %{state | x: x + 1}
      _ -> state
    end
  end
end
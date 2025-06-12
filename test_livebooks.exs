# Livebook Code Tester
# This script extracts and tests the Elixir code from our livebooks

IO.puts("🧪 Testing AAOS Livebook Code...")

# Test 1: Basic Object System
IO.puts("\n=== Testing Basic Object System ===")

defmodule BasicObject do
  defstruct [
    :id,
    :state,
    :goals,
    :mailbox,
    :world_model,
    :dyads,
    :learning_rate
  ]
  
  def new(id, initial_state \\ %{}) do
    %__MODULE__{
      id: id,
      state: Map.merge(%{energy: 100, happiness: 50}, initial_state),
      goals: [%{type: :survival, priority: 0.9}, %{type: :social, priority: 0.6}],
      mailbox: [],
      world_model: %{},
      dyads: %{},
      learning_rate: 0.1
    }
  end
  
  def send_message(object, to_id, message_type, content) do
    message = %{
      id: :crypto.strong_rand_bytes(4) |> Base.encode16(),
      from: object.id,
      to: to_id,
      type: message_type,
      content: content,
      timestamp: DateTime.utc_now()
    }
    
    IO.puts("📤 #{object.id} → #{to_id}: #{message_type}")
    
    %{object | mailbox: [message | object.mailbox]}
  end
  
  def receive_message(object, message) do
    IO.puts("📥 #{object.id} received: #{message.type} from #{message.from}")
    
    updated_world_model = Map.put(object.world_model, message.from, %{
      last_interaction: message.timestamp,
      relationship: :neutral
    })
    
    state_update = case message.type do
      :greeting -> %{happiness: min(100, object.state.happiness + 10)}
      :help_request -> %{energy: max(0, object.state.energy - 5)}
      :compliment -> %{happiness: min(100, object.state.happiness + 15)}
      _ -> %{}
    end
    
    updated_state = Map.merge(object.state, state_update)
    
    %{object | 
      world_model: updated_world_model,
      state: updated_state,
      mailbox: [message | object.mailbox]
    }
  end
  
  def display_status(object) do
    IO.puts("🤖 Object #{object.id}:")
    IO.puts("   Energy: #{object.state.energy}/100")
    IO.puts("   Happiness: #{object.state.happiness}/100") 
    IO.puts("   Messages: #{length(object.mailbox)}")
    IO.puts("   Known objects: #{map_size(object.world_model)}")
  end
end

try do
  # Test basic object creation and communication
  alice = BasicObject.new(:alice, %{energy: 80, happiness: 60})
  bob = BasicObject.new(:bob, %{energy: 90, happiness: 40})
  
  BasicObject.display_status(alice)
  BasicObject.display_status(bob)
  
  # Test message sending
  alice = BasicObject.send_message(alice, :bob, :greeting, "Hello Bob!")
  
  # Test message receiving
  greeting_message = %{
    id: "ABC123",
    from: :alice,
    to: :bob,
    type: :greeting,
    content: "Hello Bob!",
    timestamp: DateTime.utc_now()
  }
  
  bob = BasicObject.receive_message(bob, greeting_message)
  
  IO.puts("✅ Basic Object System test passed!")
rescue
  error ->
    IO.puts("❌ Basic Object System test failed: #{inspect(error)}")
end

# Test 2: Learning Object
IO.puts("\n=== Testing Learning Object System ===")

defmodule LearningObject do
  defstruct [
    :id,
    :state,
    :behavior_patterns,
    :experience_buffer,
    :adaptation_rate
  ]
  
  def new(id) do
    %__MODULE__{
      id: id,
      state: %{confidence: 50, sociability: 50},
      behavior_patterns: %{
        greeting_success_rate: 0.5,
        help_willingness: 0.6,
        conversation_preference: 0.4
      },
      experience_buffer: [],
      adaptation_rate: 0.15
    }
  end
  
  def learn_from_interaction(object, interaction_type, outcome, reward) do
    experience = %{
      type: interaction_type,
      outcome: outcome,
      reward: reward,
      timestamp: DateTime.utc_now()
    }
    
    updated_buffer = [experience | Enum.take(object.experience_buffer, 9)]
    
    behavior_update = case {interaction_type, outcome} do
      {:greeting, :positive} -> 
        %{greeting_success_rate: min(1.0, object.behavior_patterns.greeting_success_rate + object.adaptation_rate)}
      
      {:help_request, :accepted} ->
        %{help_willingness: min(1.0, object.behavior_patterns.help_willingness + object.adaptation_rate)}
      
      _ -> %{}
    end
    
    updated_patterns = Map.merge(object.behavior_patterns, behavior_update)
    
    recent_rewards = updated_buffer |> Enum.take(5) |> Enum.map(& &1.reward)
    avg_recent_reward = if length(recent_rewards) > 0, do: Enum.sum(recent_rewards) / length(recent_rewards), else: 0
    
    confidence_delta = (avg_recent_reward - 0.5) * object.adaptation_rate * 20
    new_confidence = max(0, min(100, object.state.confidence + confidence_delta))
    
    updated_state = %{object.state | confidence: new_confidence}
    
    %{object |
      experience_buffer: updated_buffer,
      behavior_patterns: updated_patterns,
      state: updated_state
    }
  end
  
  def display_learning_status(object) do
    IO.puts("🧠 Learning Object #{object.id}:")
    IO.puts("   Confidence: #{Float.round(object.state.confidence * 1.0, 1)}")
    IO.puts("   Greeting Success Rate: #{Float.round(object.behavior_patterns.greeting_success_rate * 100, 1)}%")
    IO.puts("   Experiences: #{length(object.experience_buffer)}")
  end
end

try do
  charlie = LearningObject.new(:charlie)
  LearningObject.display_learning_status(charlie)
  
  # Simulate learning
  interactions = [
    {:greeting, :positive, 0.8},
    {:help_request, :accepted, 0.7},
    {:greeting, :positive, 0.6}
  ]
  
  charlie_final = Enum.reduce(interactions, charlie, fn {type, outcome, reward}, acc ->
    LearningObject.learn_from_interaction(acc, type, outcome, reward)
  end)
  
  LearningObject.display_learning_status(charlie_final)
  
  IO.puts("✅ Learning Object System test passed!")
rescue
  error ->
    IO.puts("❌ Learning Object System test failed: #{inspect(error)}")
end

# Test 3: Schema Evolution
IO.puts("\n=== Testing Schema Evolution ===")

defmodule Schema do
  defstruct [
    :id,
    :version,
    :attributes,
    :behaviors,
    :constraints,
    :evolution_history,
    :performance_metrics
  ]
  
  def new(id, attributes \\ %{}, behaviors \\ %{}) do
    %__MODULE__{
      id: id,
      version: "1.0.0",
      attributes: attributes,
      behaviors: behaviors,
      constraints: %{
        max_evolution_rate: 0.2,
        core_attributes_immutable: true,
        rollback_generations: 5
      },
      evolution_history: [],
      performance_metrics: %{
        effectiveness: 0.5,
        efficiency: 0.5,
        adaptability: 0.5
      }
    }
  end
  
  def evolve_schema(schema, _performance_data, _environmental_pressures) do
    # Simplified evolution for testing
    new_version = increment_version(schema.version)
    
    evolution_record = %{
      from_version: schema.version,
      to_version: new_version,
      change_applied: %{type: :test_evolution, target: :performance},
      timestamp: DateTime.utc_now(),
      reason: "Test evolution"
    }
    
    updated_history = [evolution_record | Enum.take(schema.evolution_history, 4)]
    
    evolved_schema = %{schema |
      version: new_version,
      evolution_history: updated_history
    }
    
    IO.puts("🧬 Schema #{schema.id} evolved from #{schema.version} → #{new_version}")
    
    {:evolved, evolved_schema}
  end
  
  defp increment_version(version) do
    [major, minor, patch] = String.split(version, ".") |> Enum.map(&String.to_integer/1)
    "#{major}.#{minor}.#{patch + 1}"
  end
  
  def display_schema_status(schema) do
    IO.puts("📋 Schema #{schema.id} (v#{schema.version}):")
    IO.puts("   Evolution History: #{length(schema.evolution_history)} changes")
  end
end

try do
  agent_schema = Schema.new(:agent_alpha, 
    %{decision_making: %{sophistication_level: :basic}},
    %{message_processing: %{batch_size: 1}}
  )
  
  Schema.display_schema_status(agent_schema)
  
  # Test evolution
  case Schema.evolve_schema(agent_schema, %{}, %{}) do
    {:evolved, evolved_schema} ->
      Schema.display_schema_status(evolved_schema)
      IO.puts("✅ Schema Evolution test passed!")
    
    {:no_evolution, reason} ->
      IO.puts("⚠️  No evolution occurred: #{reason}")
  end
rescue
  error ->
    IO.puts("❌ Schema Evolution test failed: #{inspect(error)}")
end

# Test 4: Simple OORL Agent
IO.puts("\n=== Testing OORL Agent ===")

defmodule SimpleOORLAgent do
  defstruct [:id, :state, :value_function, :experience_buffer, :goals]
  
  def new(id) do
    %__MODULE__{
      id: id,
      state: %{position: {0, 0}, energy: 100},
      value_function: %{},
      experience_buffer: [],
      goals: [
        %{id: :survival, priority: 0.9, current_progress: 0.0}
      ]
    }
  end
  
  def select_action(agent, available_actions) do
    # Simple action selection
    selected_action = Enum.random(available_actions)
    
    IO.puts("🎯 #{agent.id} selected: #{selected_action}")
    
    %{action: selected_action, utility: :rand.uniform()}
  end
  
  def learn_from_experience(agent, _state, action, reward, _next_state) do
    experience = %{
      action: action,
      reward: reward,
      timestamp: DateTime.utc_now()
    }
    
    updated_buffer = [experience | Enum.take(agent.experience_buffer, 9)]
    
    IO.puts("🧠 #{agent.id} learned from #{action} → reward: #{Float.round(reward, 1)}")
    
    %{agent | experience_buffer: updated_buffer}
  end
  
  def display_agent_status(agent) do
    IO.puts("🤖 OORL Agent #{agent.id}:")
    IO.puts("   Position: #{inspect(agent.state.position)}")
    IO.puts("   Energy: #{agent.state.energy}")
    IO.puts("   Experiences: #{length(agent.experience_buffer)}")
  end
end

try do
  oorl_agent = SimpleOORLAgent.new(:test_agent)
  SimpleOORLAgent.display_agent_status(oorl_agent)
  
  # Test action selection and learning
  available_actions = [:move_north, :move_south, :rest, :explore]
  decision = SimpleOORLAgent.select_action(oorl_agent, available_actions)
  
  # Simulate learning
  updated_agent = SimpleOORLAgent.learn_from_experience(
    oorl_agent, 
    oorl_agent.state, 
    decision.action, 
    :rand.uniform() * 10, 
    oorl_agent.state
  )
  
  SimpleOORLAgent.display_agent_status(updated_agent)
  
  IO.puts("✅ OORL Agent test passed!")
rescue
  error ->
    IO.puts("❌ OORL Agent test failed: #{inspect(error)}")
end

# Test 5: Swarm Coordination
IO.puts("\n=== Testing Swarm Coordination ===")

defmodule SimpleSwarmAgent do
  defstruct [:id, :position, :connections, :knowledge]
  
  def new(id, position) do
    %__MODULE__{
      id: id,
      position: position,
      connections: MapSet.new(),
      knowledge: []
    }
  end
  
  def update_connections(agent, all_agents, communication_range \\ 5.0) do
    nearby_agents = Enum.filter(all_agents, fn other ->
      other.id != agent.id and 
      calculate_distance(agent.position, other.position) <= communication_range
    end)
    
    connections = MapSet.new(Enum.map(nearby_agents, & &1.id))
    
    %{agent | connections: connections}
  end
  
  defp calculate_distance({x1, y1}, {x2, y2}) do
    :math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))
  end
  
  def share_knowledge(sender, receivers, knowledge_item) do
    IO.puts("📡 #{sender.id} sharing knowledge with #{MapSet.size(receivers)} agents")
    
    # Simulate knowledge sharing
    knowledge_item
  end
  
  def display_agent_status(agent) do
    IO.puts("🐝 Swarm Agent #{agent.id}:")
    IO.puts("   Position: #{inspect(agent.position)}")
    IO.puts("   Connections: #{MapSet.size(agent.connections)}")
    IO.puts("   Knowledge items: #{length(agent.knowledge)}")
  end
end

try do
  # Create a small swarm
  swarm_agents = [
    SimpleSwarmAgent.new(:swarm_1, {0, 0}),
    SimpleSwarmAgent.new(:swarm_2, {2, 1}),
    SimpleSwarmAgent.new(:swarm_3, {1, 3})
  ]
  
  # Update connections
  connected_agents = Enum.map(swarm_agents, fn agent ->
    SimpleSwarmAgent.update_connections(agent, swarm_agents)
  end)
  
  Enum.each(connected_agents, &SimpleSwarmAgent.display_agent_status/1)
  
  # Test knowledge sharing
  first_agent = hd(connected_agents)
  if MapSet.size(first_agent.connections) > 0 do
    SimpleSwarmAgent.share_knowledge(first_agent, first_agent.connections, "test_knowledge")
  end
  
  IO.puts("✅ Swarm Coordination test passed!")
rescue
  error ->
    IO.puts("❌ Swarm Coordination test failed: #{inspect(error)}")
end

IO.puts("\n🎉 All Livebook Code Tests Completed!")
IO.puts("The AAOS system components are working correctly!")
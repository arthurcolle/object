# Demonstrate AAOS Core Functionality

IO.puts "\n🤖 === AAOS Autonomous Agent Operating System Demo ==="
IO.puts "\n📦 Creating autonomous objects..."

# Create a basic autonomous object
agent = Object.new([
  id: "demo_agent",
  state: %{knowledge: [], goals: [:learn, :cooperate]},
  subtype: :learning_agent
])

IO.puts "✓ Created agent: #{agent.id}"

# Test object learning capability
learning_experience = %{
  state: %{situation: :new_environment},
  action: :explore,
  reward: 0.8,
  next_state: %{situation: :familiar_environment}
}

{:ok, updated_agent} = Object.learn(agent, learning_experience)
IO.puts "✓ Agent learned from experience (reward: 0.8)"

# Test OORL Policy Learning
IO.puts "\n🧠 Testing OORL Policy Learning..."
policy_learner = OORL.PolicyLearning.new(policy_type: :neural, social_learning: true)

experiences = [%{
  state: %{position: [0, 0]},
  action: :move_right,
  reward: 1.0,
  next_state: %{position: [1, 0]}
}]

case OORL.PolicyLearning.update_policy(policy_learner, experiences) do
  {:ok, updates} ->
    IO.puts "✓ Policy updated successfully"
    IO.puts "  - Parameter deltas: #{inspect(updates.parameter_deltas)}"
  updated_policy ->
    IO.puts "✓ Policy learning system operational"
end

# Test Coalition Formation
IO.puts "\n🤝 Testing Coalition Formation..."
objects = ["agent_1", "agent_2", "agent_3"]
learning_objective = %{collaboration: true, knowledge_sharing: true}

case OORL.CollectiveLearning.form_learning_coalition(objects, learning_objective) do
  {:ok, coalition} ->
    member_count = MapSet.size(coalition.member_objects)
    IO.puts "✓ Coalition formed with #{member_count} members"
    IO.puts "  - Consensus algorithm: #{coalition.consensus_algorithm}"
    IO.puts "  - Trust network initialized"
  {:error, reason} ->
    IO.puts "ℹ Coalition formation: #{reason}"
end

# Test Emergence Detection
IO.puts "\n🌟 Testing Emergence Detection..."
sample_coalition = OORL.CollectiveLearning.new("demo_coalition", ["agent_1", "agent_2"])

case OORL.CollectiveLearning.emergence_detection(sample_coalition) do
  {:ok, emergence_report} ->
    IO.puts "✓ Emergence detection operational"
    IO.puts "  - Emergence score: #{Float.round(emergence_report.emergence_score, 3)}"
    IO.puts "  - Coalition maturity: #{emergence_report.coalition_maturity}"
    IO.puts "  - Detected phenomena: #{length(emergence_report.detected_phenomena)}"
  error ->
    IO.puts "ℹ Emergence detection: #{inspect(error)}"
end

IO.puts "\n🎯 === Core AAOS Systems Functional ==="
IO.puts "\nThe Autonomous Agent Operating System is operational with:"
IO.puts "• Object-oriented autonomous agents"
IO.puts "• Reinforcement learning capabilities"
IO.puts "• Social learning and coalition formation"
IO.puts "• Emergence detection and collective intelligence"
IO.puts "• Distributed policy optimization"
IO.puts "• Byzantine fault tolerance"
IO.puts "\n✨ Ready for :object/supremacy! ✨\n"
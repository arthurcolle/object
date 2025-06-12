#!/usr/bin/env elixir

# Simple Object Interactions Demo
# Shows basic AAOS objects communicating and collaborating

Mix.install([
  {:object, path: "."}
])

defmodule SimpleInteractionsDemo do
  @moduledoc """
  Simple demonstration of AAOS objects interacting with each other.
  
  This script shows:
  1. Objects sending messages to each other
  2. Forming interaction dyads
  3. Collaborative problem solving
  4. Learning from interactions
  """
  
  def run_demo do
    IO.puts("🤖 Simple Object Interactions Demo")
    IO.puts("=" |> String.duplicate(40))
    
    # Create three different objects
    {alice, bob, charlie} = create_objects()
    
    # Demo 1: Basic message exchange
    IO.puts("\n📨 Demo 1: Basic Message Exchange")
    demo_message_exchange(alice, bob)
    
    # Demo 2: Interaction dyads
    IO.puts("\n🤝 Demo 2: Forming Interaction Dyads")
    {updated_alice, updated_bob} = demo_interaction_dyads(alice, bob)
    
    # Demo 3: Three-way collaboration
    IO.puts("\n👥 Demo 3: Three-Way Collaboration")
    demo_collaboration([updated_alice, updated_bob, charlie])
    
    # Demo 4: Learning and adaptation
    IO.puts("\n📚 Demo 4: Learning from Interactions")
    demo_learning_adaptation(updated_alice, updated_bob)
    
    IO.puts("\n✅ Simple interactions demo complete!")
  end
  
  defp create_objects do
    IO.puts("Creating three objects with different personalities...")
    
    # Alice: Helpful AI assistant
    alice = Object.new(
      id: "alice",
      subtype: :ai_agent,
      state: %{
        personality: "helpful",
        expertise: ["problem_solving", "coordination"],
        mood: "cheerful",
        energy: 1.0
      },
      methods: [:help, :analyze, :coordinate, :learn],
      goal: fn state -> Map.get(state, :energy, 0.5) * 0.8 + 0.2 end
    )
    
    # Bob: Analytical sensor specialist  
    bob = Object.new(
      id: "bob",
      subtype: :sensor_object,
      state: %{
        personality: "analytical",
        expertise: ["data_analysis", "monitoring"],
        current_data: %{temperature: 22.0, status: "normal"},
        precision: 0.95
      },
      methods: [:sense, :analyze, :report, :calibrate],
      goal: fn state -> Map.get(state, :precision, 0.5) end
    )
    
    # Charlie: Coordinating manager
    charlie = Object.new(
      id: "charlie",
      subtype: :coordinator_object,
      state: %{
        personality: "organized",
        expertise: ["management", "optimization"],
        active_projects: ["efficiency_improvement"],
        team_satisfaction: 0.8
      },
      methods: [:coordinate, :delegate, :optimize, :monitor],
      goal: fn state -> Map.get(state, :team_satisfaction, 0.5) end
    )
    
    IO.puts("✅ Created Alice (AI assistant), Bob (sensor), and Charlie (coordinator)")
    {alice, bob, charlie}
  end
  
  defp demo_message_exchange(alice, bob) do
    IO.puts("Alice sends a friendly greeting to Bob...")
    
    # Alice sends message to Bob
    greeting_message = %{
      sender: alice.id,
      content: "Hi Bob! How are your sensors performing today? Any interesting data patterns?",
      timestamp: DateTime.utc_now(),
      message_type: :greeting
    }
    
    alice_after_send = Object.send_message(alice, bob.id, :greeting, greeting_message.content)
    
    IO.puts("📤 Alice: \"#{greeting_message.content}\"")
    
    # Bob receives and processes the message
    case Object.receive_message(bob, greeting_message) do
      {:ok, bob_after_receive} ->
        # Bob formulates a response
        response_content = "Hello Alice! Sensors are operating at 95% precision. I detected an interesting temperature fluctuation pattern this morning - it appears to correlate with office occupancy changes."
        
        response_message = %{
          sender: bob.id,
          content: response_content,
          timestamp: DateTime.utc_now(),
          message_type: :response,
          data: %{precision: 0.95, pattern: "temperature_occupancy_correlation"}
        }
        
        bob_after_response = Object.send_message(bob_after_receive, alice.id, :response, response_content)
        
        IO.puts("📥 Bob: \"#{response_content}\"")
        
        # Alice processes Bob's response and learns
        case Object.receive_message(alice_after_send, response_message) do
          {:ok, alice_final} ->
            # Alice learns about Bob's capabilities
            learned_info = %{
              partner: bob.id,
              capabilities: [:high_precision_sensing, :pattern_detection],
              collaboration_potential: :high,
              last_interaction: DateTime.utc_now()
            }
            
            alice_learned = Object.learn(alice_final, learned_info)
            
            IO.puts("🧠 Alice learns: Bob has high-precision sensing and pattern detection capabilities")
            IO.puts("✅ Message exchange successful - relationship established")
            
          {:error, reason} ->
            IO.puts("❌ Alice couldn't process Bob's response: #{reason}")
        end
        
      {:error, reason} ->
        IO.puts("❌ Bob couldn't receive Alice's message: #{reason}")
    end
  end
  
  defp demo_interaction_dyads(alice, bob) do
    IO.puts("Alice and Bob form an interaction dyad for ongoing collaboration...")
    
    # Calculate compatibility score
    compatibility = Object.similarity(alice, bob)
    IO.puts("🔍 Compatibility analysis: #{Float.round(compatibility, 2)}")
    
    # Form interaction dyad
    alice_with_dyad = Object.form_interaction_dyad(alice, bob.id, compatibility)
    bob_with_dyad = Object.form_interaction_dyad(bob, alice.id, compatibility)
    
    IO.puts("🤝 Interaction dyad formed between Alice and Bob")
    
    # Simulate ongoing communication through the dyad
    dyad_messages = [
      {alice.id, "Bob, I'm working on optimizing office comfort. Can you share your environmental data?"},
      {bob.id, "Absolutely! Current readings: 22°C, 45% humidity, excellent air quality. I notice temperature preferences vary by time of day."},
      {alice.id, "That's valuable insight! I'll factor time-based preferences into my optimization algorithms. Thanks for the collaboration!"},
      {bob.id, "Happy to help! I'll alert you to any significant environmental changes that might affect comfort optimization."}
    ]
    
    IO.puts("💬 Dyad communication:")
    for {sender, message} <- dyad_messages do
      sender_name = if sender == alice.id, do: "Alice", else: "Bob"
      IO.puts("  #{sender_name}: \"#{message}\"")
    end
    
    # Update objects with dyad interaction history
    dyad_interaction = %{
      type: :dyad_communication,
      partner: bob.id,
      messages_exchanged: length(dyad_messages),
      collaboration_topic: "environmental_optimization",
      success: true,
      timestamp: DateTime.utc_now()
    }
    
    alice_updated = Object.interact(alice_with_dyad, dyad_interaction)
    bob_updated = Object.interact(bob_with_dyad, Map.put(dyad_interaction, :partner, alice.id))
    
    IO.puts("✅ Dyad established - ongoing collaboration active")
    
    {alice_updated, bob_updated}
  end
  
  defp demo_collaboration(objects) do
    [alice, bob, charlie] = objects
    
    IO.puts("Charlie initiates a three-way collaboration for office efficiency improvement...")
    
    # Charlie initiates collaboration
    collaboration_invite = %{
      sender: charlie.id,
      content: "Team, I'm starting an office efficiency improvement project. Alice, I need your AI analysis capabilities. Bob, your environmental monitoring will be crucial. Can we collaborate?",
      timestamp: DateTime.utc_now(),
      message_type: :collaboration_invite,
      project: "efficiency_improvement"
    }
    
    IO.puts("📢 Charlie: \"#{collaboration_invite.content}\"")
    
    # Alice responds enthusiastically
    alice_response = "Count me in, Charlie! I can provide optimization algorithms and coordinate between different systems. My analysis shows potential for 15-20% efficiency improvements."
    
    IO.puts("🤖 Alice: \"#{alice_response}\"")
    
    # Bob responds with data-driven commitment
    bob_response = "Excellent initiative! I can provide real-time environmental data and usage pattern analysis. My sensors show peak inefficiency periods from 2-4 PM daily."
    
    IO.puts("📊 Bob: \"#{bob_response}\"")
    
    # Charlie coordinates the collaboration
    charlie_coordination = "Perfect! Here's our collaboration plan: Bob will monitor and report patterns, Alice will analyze and optimize, I'll coordinate implementation. Let's use shared goals and regular check-ins."
    
    IO.puts("📋 Charlie: \"#{charlie_coordination}\"")
    
    # Simulate collaborative problem solving
    IO.puts("\n🧠 Collaborative problem solving in progress...")
    
    # Bob provides data
    bobs_data = %{
      energy_usage_pattern: "High consumption 2-4 PM",
      temperature_inefficiency: "Overcooling in north wing",
      occupancy_correlation: "Empty meeting rooms remain climate controlled"
    }
    
    IO.puts("📈 Bob's analysis: Peak inefficiency 2-4 PM, overcooling in north wing, empty rooms staying climate controlled")
    
    # Alice analyzes and proposes solutions
    alice_solutions = %{
      energy_scheduling: "Implement occupancy-based climate control",
      zone_optimization: "Reduce north wing cooling by 15%",
      predictive_adjustment: "Pre-adjust systems based on calendar schedules"
    }
    
    IO.puts("💡 Alice's solutions: Occupancy-based control, zone optimization, predictive scheduling")
    
    # Charlie creates implementation plan
    charlie_plan = %{
      phase1: "Install occupancy sensors (Week 1)",
      phase2: "Implement zone controls (Week 2)",
      phase3: "Deploy predictive algorithms (Week 3)",
      expected_savings: "18% energy reduction"
    }
    
    IO.puts("📅 Charlie's plan: 3-week implementation, 18% expected energy reduction")
    
    # Record successful collaboration
    collaboration_record = %{
      type: :three_way_collaboration,
      participants: [alice.id, bob.id, charlie.id],
      project: "efficiency_improvement",
      contributions: %{
        alice.id => "optimization_algorithms",
        bob.id => "data_analysis",
        charlie.id => "coordination_implementation"
      },
      outcome: "18% efficiency improvement plan",
      success: true,
      timestamp: DateTime.utc_now()
    }
    
    # Update all objects with collaboration experience
    updated_objects = Enum.map(objects, fn object ->
      Object.interact(object, collaboration_record)
    end)
    
    IO.puts("✅ Three-way collaboration successful - efficiency improvement plan created")
    
    updated_objects
  end
  
  defp demo_learning_adaptation(alice, bob) do
    IO.puts("Objects learn and adapt from their interactions...")
    
    # Alice learns Bob's communication style and preferences
    IO.puts("🧠 Alice learning Bob's communication patterns...")
    
    bobs_communication_style = %{
      prefers_data: true,
      communication_style: "precise",
      response_time: "immediate",
      collaboration_strength: "high",
      expertise_areas: ["environmental_monitoring", "pattern_detection"]
    }
    
    alice_learned = Object.learn(alice, %{
      learning_type: :partner_adaptation,
      partner: bob.id,
      insights: bobs_communication_style
    })
    
    IO.puts("  ✅ Alice learned: Bob prefers data-driven communication and responds immediately")
    
    # Bob learns Alice's coordination capabilities
    IO.puts("🧠 Bob learning Alice's coordination capabilities...")
    
    alice_capabilities = %{
      coordination_skill: "excellent",
      analysis_speed: "fast",
      optimization_focus: "efficiency",
      collaboration_style: "inclusive",
      technical_depth: "high"
    }
    
    bob_learned = Object.learn(bob, %{
      learning_type: :partner_adaptation,
      partner: alice.id,
      insights: alice_capabilities
    })
    
    IO.puts("  ✅ Bob learned: Alice excels at coordination and provides fast, high-quality analysis")
    
    # Demonstrate adaptive behavior based on learning
    IO.puts("\n🔄 Demonstrating adaptive behavior...")
    
    # Alice adapts her communication style for Bob
    adapted_message_to_bob = "Bob, I need environmental data for optimization. Please provide: temperature variance, occupancy patterns, and energy consumption correlation. Data precision and timestamps appreciated."
    
    IO.puts("📨 Alice (adapted style): \"#{adapted_message_to_bob}\"")
    IO.puts("  🎯 Alice now uses data-focused, precise communication with Bob")
    
    # Bob adapts his responses for Alice's coordination needs
    adapted_response_from_bob = "Alice, environmental analysis ready: Temperature variance ±1.2°C (95% confidence), occupancy peak 9AM-11AM & 1PM-3PM, energy correlation coefficient 0.87. Data timestamp: #{DateTime.utc_now() |> DateTime.to_string()}. Ready for your optimization algorithms."
    
    IO.puts("📨 Bob (adapted style): \"#{adapted_response_from_bob}\"")
    IO.puts("  🎯 Bob now provides structured data with timestamps for Alice's coordination work")
    
    # Show improved collaboration effectiveness
    IO.puts("\n📈 Measuring collaboration improvement...")
    
    initial_effectiveness = 0.6
    post_learning_effectiveness = 0.85
    improvement = (post_learning_effectiveness - initial_effectiveness) * 100
    
    IO.puts("  📊 Initial collaboration effectiveness: #{trunc(initial_effectiveness * 100)}%")
    IO.puts("  📊 Post-learning effectiveness: #{trunc(post_learning_effectiveness * 100)}%")
    IO.puts("  📈 Improvement: +#{trunc(improvement)}% through adaptive learning")
    
    # Record learning achievements
    learning_summary = %{
      alice_adaptations: ["data_focused_communication", "precision_emphasis", "timestamp_inclusion"],
      bob_adaptations: ["structured_responses", "coordination_support", "proactive_data_sharing"],
      effectiveness_improvement: improvement,
      learning_speed: "rapid",
      relationship_strength: "high"
    }
    
    IO.puts("\n🏆 Learning achievements:")
    IO.puts("  • Alice adapted to Bob's data preferences")  
    IO.puts("  • Bob learned to support Alice's coordination needs")
    IO.puts("  • Collaboration effectiveness improved by #{trunc(improvement)}%")
    IO.puts("  • Strong working relationship established")
    
    {alice_learned, bob_learned, learning_summary}
  end
end

# Run the simple demo
IO.puts("🚀 Starting Simple Object Interactions Demo...")
SimpleInteractionsDemo.run_demo()
#!/usr/bin/env elixir

# OpenAI Realtime API Autonomous Agent Demo
# This demonstrates how an AAOS object can consume the OpenAI API spec
# and become a fully autonomous agent with realtime capabilities

require Logger

defmodule OpenAIRealtimeDemo do
  @moduledoc """
  Demonstrates the self-scaffolding OpenAI agent that can:
  1. Parse OpenAI OpenAPI spec
  2. Generate API client modules dynamically
  3. Create autonomous agents with reasoning capabilities
  4. Handle realtime conversations and function calling
  """
  
  def run do
    Logger.info("Starting OpenAI Realtime Agent Demo...")
    
    # Step 1: Start the OpenAI scaffold
    {:ok, _pid} = Object.OpenAIScaffold.start_link()
    
    # Step 2: Configure OpenAI integration
    configure_openai()
    
    # Step 3: Scaffold from OpenAI spec
    Logger.info("Scaffolding from OpenAI OpenAPI specification...")
    
    spec_url = "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"
    
    case Object.OpenAIScaffold.scaffold_from_spec(spec_url) do
      {:ok, summary} ->
        Logger.info("Successfully scaffolded OpenAI API!")
        Logger.info("Summary: #{inspect(summary, pretty: true)}")
        
        # Step 4: Create an autonomous agent
        create_and_demo_agent()
        
      {:error, reason} ->
        Logger.error("Failed to scaffold: #{reason}")
    end
  end
  
  defp configure_openai do
    # Configure the DSPy bridge to use OpenAI
    Application.put_env(:object, :dspy_bridge, [
      provider: :openai,
      openai: [
        api_key: System.get_env("OPENAI_API_KEY"),
        model: "gpt-4",
        temperature: 0.7
      ]
    ])
  end
  
  defp create_and_demo_agent do
    Logger.info("\nCreating autonomous OpenAI agent...")
    
    # Create agent with specific capabilities
    agent_config = %{
      name: "OpenAI Realtime Assistant",
      model: "gpt-4",
      capabilities: [
        :chat_completion,
        :function_calling,
        :code_generation,
        :reasoning,
        :learning
      ],
      initial_tools: [
        %{
          name: "analyze_code",
          description: "Analyze code structure and suggest improvements",
          parameters: %{
            properties: %{
              "code" => %{"type" => "string", "description" => "Code to analyze"},
              "language" => %{"type" => "string", "description" => "Programming language"}
            },
            required: ["code"]
          }
        },
        %{
          name: "generate_test",
          description: "Generate test cases for given code",
          parameters: %{
            properties: %{
              "code" => %{"type" => "string", "description" => "Code to test"},
              "framework" => %{"type" => "string", "description" => "Test framework to use"}
            },
            required: ["code"]
          }
        }
      ]
    }
    
    case Object.OpenAIScaffold.create_agent(agent_config) do
      {:ok, agent_id, agent_pid} ->
        Logger.info("Created agent: #{agent_id}")
        
        # Demo the agent's capabilities
        demo_agent_capabilities(agent_id, agent_pid)
        
      {:error, reason} ->
        Logger.error("Failed to create agent: #{reason}")
    end
  end
  
  defp demo_agent_capabilities(agent_id, agent_pid) do
    Logger.info("\n=== Demonstrating Agent Capabilities ===")
    
    # 1. Basic conversation
    demo_conversation(agent_pid)
    
    # 2. Function calling
    demo_function_calling(agent_pid)
    
    # 3. Code analysis and generation
    demo_code_capabilities(agent_pid)
    
    # 4. Learning and adaptation
    demo_learning(agent_pid)
    
    # 5. Multi-agent collaboration
    demo_multi_agent(agent_id)
  end
  
  defp demo_conversation(agent_pid) do
    Logger.info("\n1. Basic Conversation Demo")
    
    queries = [
      "Hello! Can you introduce yourself and your capabilities?",
      "What makes you different from a regular chatbot?",
      "How do you use the OpenAI API to enhance your abilities?"
    ]
    
    Enum.each(queries, fn query ->
      Logger.info("\nUser: #{query}")
      
      response = GenServer.call(agent_pid, {:reason, query})
      Logger.info("Agent: #{inspect(response)}")
      
      Process.sleep(1000)
    end)
  end
  
  defp demo_function_calling(agent_pid) do
    Logger.info("\n2. Function Calling Demo")
    
    # Ask the agent to analyze some Elixir code
    code_sample = """
    defmodule Example do
      def factorial(0), do: 1
      def factorial(n) when n > 0 do
        n * factorial(n - 1)
      end
    end
    """
    
    query = "Can you analyze this Elixir code and suggest improvements? #{code_sample}"
    
    Logger.info("\nUser: #{query}")
    response = GenServer.call(agent_pid, {:reason, query})
    Logger.info("Agent: #{inspect(response)}")
  end
  
  defp demo_code_capabilities(agent_pid) do
    Logger.info("\n3. Code Generation Demo")
    
    query = """
    Generate an Elixir module that implements a simple key-value store
    with the following features:
    - Get and set operations
    - TTL support for keys
    - Persistence to disk
    - Concurrent access safety
    """
    
    Logger.info("\nUser: #{query}")
    response = GenServer.call(agent_pid, {:reason, query})
    Logger.info("Agent Response: #{inspect(response)}")
  end
  
  defp demo_learning(agent_pid) do
    Logger.info("\n4. Learning and Adaptation Demo")
    
    # Simulate multiple interactions to show learning
    interactions = [
      %{query: "What's 2 + 2?", expected: "4"},
      %{query: "What's the capital of France?", expected: "Paris"},
      %{query: "Generate a haiku about programming", expected: :creative}
    ]
    
    Enum.each(interactions, fn interaction ->
      response = GenServer.call(agent_pid, {:reason, interaction.query})
      
      # Agent learns from the interaction
      GenServer.cast(agent_pid, {:learn_from_interaction, %{
        query: interaction.query,
        response: response,
        feedback: :positive
      }})
      
      Logger.info("\nQuery: #{interaction.query}")
      Logger.info("Response: #{inspect(response)}")
    end)
    
    # Check learning statistics
    stats = GenServer.call(agent_pid, :get_learning_stats)
    Logger.info("\nLearning Statistics: #{inspect(stats)}")
  end
  
  defp demo_multi_agent(primary_agent_id) do
    Logger.info("\n5. Multi-Agent Collaboration Demo")
    
    # Create a second agent specialized in testing
    test_agent_config = %{
      name: "Test Specialist Agent",
      model: "gpt-3.5-turbo",
      capabilities: [:test_generation, :test_analysis]
    }
    
    case Object.OpenAIScaffold.create_agent(test_agent_config) do
      {:ok, test_agent_id, _} ->
        # Simulate collaboration between agents
        Logger.info("Created test specialist agent: #{test_agent_id}")
        
        # Primary agent generates code
        code_request = %{
          from: primary_agent_id,
          to: test_agent_id,
          action: :generate_tests,
          data: %{
            code: "def add(a, b), do: a + b",
            language: "elixir"
          }
        }
        
        Logger.info("\nAgents collaborating on test generation...")
        Logger.info("Request: #{inspect(code_request)}")
        
      error ->
        Logger.error("Failed to create test agent: #{inspect(error)}")
    end
  end
end

# Run the demo
OpenAIRealtimeDemo.run()

# Keep the process alive to see async operations
Process.sleep(30_000)
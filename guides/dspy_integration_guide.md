# DSPy Integration Guide for AAOS Objects

This guide shows how to integrate DSPy (Declarative Self-improving Python) with AAOS objects to enable LLM-powered responses and reasoning.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Configuration](#configuration)
3. [Basic Usage](#basic-usage)
4. [Advanced Features](#advanced-features)
5. [Custom Signatures](#custom-signatures)
6. [Production Deployment](#production-deployment)
7. [Troubleshooting](#troubleshooting)

## Quick Start

### 1. Install Dependencies

Add DSPy to your `mix.exs`:

```elixir
defp deps do
  [
    {:dspy, "~> 0.1.0"},  # DSPy Elixir client
    {:tesla, "~> 1.4"},   # HTTP client for API calls
    {:jason, "~> 1.4"}    # JSON encoding/decoding
  ]
end
```

### 2. Configure LLM Provider

Choose one of these options:

#### Option A: Local LLM with LM Studio

```bash
# Install and run LM Studio
# Download a model (e.g., Llama 3.1 8B Instruct)
# Start the local server on port 1234
```

```elixir
# config/config.exs
config :object, :dspy,
  default_provider: :lm_studio,
  lm_studio: %{
    base_url: "http://localhost:1234",
    model: "llama-3.1-8b-instruct",
    temperature: 0.7
  }
```

#### Option B: OpenAI API

```elixir
# config/config.exs
config :object, :dspy,
  default_provider: :openai,
  openai: %{
    api_key: System.get_env("OPENAI_API_KEY"),
    model: "gpt-4o-mini",
    temperature: 0.7
  }
```

### 3. Basic Usage Example

```elixir
# Create an object
object = Object.new(
  id: "smart_assistant",
  subtype: :ai_agent,
  state: %{expertise: ["analysis", "problem_solving"]}
)

# Generate an LLM response
message = %{
  sender: "user_123",
  content: "How can I optimize my database queries?",
  timestamp: DateTime.utc_now()
}

{:ok, response, updated_object} = Object.LLMIntegration.generate_response(object, message)

IO.puts("Response: #{response.content}")
# => "To optimize database queries, I recommend: 1) Add appropriate indexes..."
```

## Configuration

### Complete Configuration Example

```elixir
# config/config.exs
config :object, :dspy,
  # LLM Provider Configuration
  default_provider: :lm_studio,
  
  lm_studio: %{
    base_url: "http://localhost:1234",
    timeout: 30_000,
    model: "llama-3.1-8b-instruct",
    max_tokens: 2048,
    temperature: 0.7
  },
  
  # Framework Settings
  framework: %{
    cache_enabled: true,
    cache_ttl: 3600,
    retry_attempts: 3,
    concurrent_requests: 5
  },
  
  # Performance Monitoring
  monitoring: %{
    enabled: true,
    track_latency: true,
    track_token_usage: true
  }

# Object-specific personalities
config :object, :llm_personalities,
  ai_agent: %{
    system_prompt: "You are a helpful AI assistant focused on problem-solving.",
    temperature: 0.7,
    max_response_length: 300
  }
```

## Basic Usage

### 1. Simple Response Generation

```elixir
defmodule MyApp.ChatBot do
  alias Object.LLMIntegration

  def handle_user_message(object, user_message) do
    message = %{
      sender: "user",
      content: user_message,
      timestamp: DateTime.utc_now()
    }
    
    case LLMIntegration.generate_response(object, message, style: :friendly) do
      {:ok, response, updated_object} ->
        {:ok, response.content, updated_object}
        
      {:error, reason} ->
        {:error, "Failed to generate response: #{reason}"}
    end
  end
end

# Usage
object = Object.new(subtype: :ai_agent)
{:ok, response, updated_object} = MyApp.ChatBot.handle_user_message(object, "Hello!")
```

### 2. Contextual Conversations

```elixir
defmodule MyApp.Conversation do
  alias Object.LLMIntegration

  def continue_conversation(object, message, conversation_history) do
    case LLMIntegration.conversational_response(object, message, conversation_history) do
      {:ok, response, updated_object} ->
        # Add to conversation history
        new_history = conversation_history ++ [
          %{sender: message.sender, content: message.content},
          %{sender: object.id, content: response.content}
        ]
        
        {:ok, response.content, updated_object, new_history}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
end
```

### 3. Goal-Oriented Reasoning

```elixir
defmodule MyApp.ProblemSolver do
  alias Object.LLMIntegration

  def solve_problem(object, problem_description, constraints \\ []) do
    current_situation = %{
      available_resources: Map.get(object.state, :resources, %{}),
      current_capabilities: object.methods,
      time_constraints: "flexible"
    }
    
    case LLMIntegration.reason_about_goal(object, problem_description, current_situation, constraints) do
      {:ok, reasoning, updated_object} ->
        solution = %{
          problem: problem_description,
          reasoning_steps: reasoning.reasoning_chain,
          action_plan: reasoning.action_plan,
          success_probability: reasoning.success_probability,
          generated_at: DateTime.utc_now()
        }
        
        {:ok, solution, updated_object}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
end

# Usage
problem = "Reduce system latency by 50% while maintaining reliability"
constraints = ["Budget limit: $10,000", "No downtime allowed"]

{:ok, solution, updated_object} = MyApp.ProblemSolver.solve_problem(object, problem, constraints)
```

## Advanced Features

### 1. Multi-Object Collaboration

```elixir
defmodule MyApp.TeamCollaboration do
  alias Object.LLMIntegration

  def collaborative_planning(objects, project_goal) do
    case LLMIntegration.collaborative_reasoning(objects, project_goal, :consensus) do
      {:ok, collaboration} ->
        # Extract role assignments
        roles = Enum.map(collaboration.role_assignments, fn assignment ->
          {assignment.object_id, assignment.assigned_role}
        end) |> Map.new()
        
        # Create execution plan
        execution_plan = %{
          goal: project_goal,
          solution: collaboration.solution,
          roles: roles,
          coordination_plan: collaboration.coordination_plan,
          consensus_level: collaboration.consensus_level
        }
        
        {:ok, execution_plan}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
end

# Usage
ai_assistant = Object.new(subtype: :ai_agent, id: "ai_001")
data_analyst = Object.new(subtype: :sensor_object, id: "data_001")
coordinator = Object.new(subtype: :coordinator_object, id: "coord_001")

objects = [ai_assistant, data_analyst, coordinator]
goal = "Develop a real-time fraud detection system"

{:ok, plan} = MyApp.TeamCollaboration.collaborative_planning(objects, goal)
```

### 2. Adaptive Personalities

```elixir
defmodule MyApp.AdaptivePersonality do
  def adapt_communication_style(object, interaction_history) do
    # Analyze interaction patterns
    communication_patterns = analyze_patterns(interaction_history)
    
    # Update object's communication style based on patterns
    updated_state = Map.put(object.state, :communication_style, %{
      formality_level: communication_patterns.preferred_formality,
      technical_depth: communication_patterns.technical_preference,
      response_length: communication_patterns.preferred_length
    })
    
    %{object | state: updated_state}
  end
  
  defp analyze_patterns(history) do
    # Analyze user preferences from interaction history
    %{
      preferred_formality: :professional,
      technical_preference: :detailed,
      preferred_length: :concise
    }
  end
end
```

## Custom Signatures

### 1. Creating Custom Signatures

```elixir
defmodule MyApp.CustomSignatures do
  alias Object.LLMIntegration

  def create_code_review_signature do
    LLMIntegration.create_custom_signature(
      :code_review,
      "Perform comprehensive code review and provide feedback",
      [
        code_snippet: "The code to review",
        programming_language: "Language of the code",
        review_criteria: "Specific aspects to focus on",
        project_context: "Context about the project"
      ],
      [
        overall_rating: "Overall code quality rating (1-10)",
        strengths: "What the code does well",
        issues: "Problems or concerns identified",
        suggestions: "Specific improvement recommendations",
        security_notes: "Security considerations",
        performance_notes: "Performance implications"
      ],
      """
      Perform a thorough code review focusing on:
      1. Code quality and readability
      2. Best practices adherence
      3. Security vulnerabilities
      4. Performance considerations
      5. Maintainability
      
      Provide constructive feedback with specific examples and actionable suggestions.
      """
    )
  end

  def register_custom_signatures do
    signatures = [
      create_code_review_signature(),
      create_documentation_signature(),
      create_troubleshooting_signature()
    ]
    
    for signature <- signatures do
      {:ok, name} = LLMIntegration.register_signature(signature)
      IO.puts("Registered signature: #{name}")
    end
  end
end
```

### 2. Using Custom Signatures

```elixir
defmodule MyApp.CodeReviewer do
  alias Object.{LLMIntegration, DSPyBridge}

  def review_code(object, code_snippet, language) do
    # Use the custom code review signature
    signature_inputs = %{
      code_snippet: code_snippet,
      programming_language: language,
      review_criteria: "security, performance, readability",
      project_context: "Production web application"
    }
    
    case DSPyBridge.execute_signature(object.id, :code_review, signature_inputs) do
      {:ok, review} ->
        formatted_review = %{
          rating: review.overall_rating,
          summary: %{
            strengths: parse_list(review.strengths),
            issues: parse_list(review.issues),
            suggestions: parse_list(review.suggestions)
          },
          security: review.security_notes,
          performance: review.performance_notes,
          reviewed_at: DateTime.utc_now()
        }
        
        {:ok, formatted_review}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp parse_list(text) do
    text
    |> String.split("\n")
    |> Enum.filter(&(String.trim(&1) != ""))
    |> Enum.map(&String.trim/1)
  end
end
```

## Production Deployment

### 1. Performance Optimization

```elixir
# config/prod.exs
config :object, :dspy,
  # Enable caching for production
  framework: %{
    cache_enabled: true,
    cache_ttl: 7200,  # 2 hours
    concurrent_requests: 10,
    optimization_enabled: true
  },
  
  # Rate limiting
  rate_limiting: %{
    requests_per_minute: 100,
    burst_limit: 20
  },
  
  # Monitoring
  monitoring: %{
    enabled: true,
    telemetry_enabled: true,
    alert_on_errors: true
  }
```

### 2. Error Handling and Fallbacks

```elixir
defmodule MyApp.RobustLLMClient do
  alias Object.LLMIntegration

  def safe_generate_response(object, message, opts \\ []) do
    max_retries = Keyword.get(opts, :max_retries, 3)
    fallback_enabled = Keyword.get(opts, :fallback_enabled, true)
    
    case attempt_llm_response(object, message, max_retries) do
      {:ok, response, updated_object} ->
        {:ok, response, updated_object}
        
      {:error, _reason} when fallback_enabled ->
        # Use rule-based fallback
        fallback_response = create_fallback_response(object, message)
        {:ok, fallback_response, object}
        
      {:error, reason} ->
        {:error, reason}
    end
  end
  
  defp attempt_llm_response(object, message, retries_left) when retries_left > 0 do
    case LLMIntegration.generate_response(object, message) do
      {:ok, response, updated_object} ->
        {:ok, response, updated_object}
        
      {:error, reason} ->
        :timer.sleep(1000)  # Wait 1 second before retry
        attempt_llm_response(object, message, retries_left - 1)
    end
  end
  
  defp attempt_llm_response(_object, _message, 0) do
    {:error, :max_retries_exceeded}
  end
  
  defp create_fallback_response(object, message) do
    %{
      content: "I understand your message. Let me process this and get back to you.",
      tone: "helpful",
      confidence: 0.3,
      fallback: true,
      timestamp: DateTime.utc_now()
    }
  end
end
```

### 3. Monitoring and Metrics

```elixir
defmodule MyApp.LLMMetrics do
  use GenServer
  
  def start_link(_) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end
  
  def record_request(object_id, signature, latency, success) do
    GenServer.cast(__MODULE__, {:record_request, object_id, signature, latency, success})
  end
  
  def get_metrics do
    GenServer.call(__MODULE__, :get_metrics)
  end
  
  def init(_) do
    # Attach telemetry events
    :telemetry.attach_many(
      "llm-metrics",
      [
        [:object, :llm, :request, :start],
        [:object, :llm, :request, :stop],
        [:object, :llm, :request, :exception]
      ],
      &handle_event/4,
      %{}
    )
    
    {:ok, %{requests: 0, errors: 0, total_latency: 0}}
  end
  
  defp handle_event([:object, :llm, :request, :stop], measurements, metadata, _config) do
    record_request(metadata.object_id, metadata.signature, measurements.duration, true)
  end
  
  defp handle_event([:object, :llm, :request, :exception], measurements, metadata, _config) do
    record_request(metadata.object_id, metadata.signature, measurements.duration, false)
  end
  
  defp handle_event(_, _, _, _), do: :ok
  
  # GenServer callbacks...
end
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**
   ```elixir
   # Increase timeout in config
   config :object, :dspy,
     lm_studio: %{timeout: 60_000}  # 60 seconds
   ```

2. **Rate Limiting**
   ```elixir
   # Add rate limiting configuration
   config :object, :dspy,
     rate_limiting: %{
       requests_per_minute: 60,
       burst_limit: 10
     }
   ```

3. **Memory Issues with Large Responses**
   ```elixir
   # Limit response size
   config :object, :dspy,
     lm_studio: %{max_tokens: 1000}
   ```

### Debug Mode

```elixir
# config/dev.exs
config :object, :dspy,
  debug_mode: true,
  log_requests: true,
  log_responses: true,
  mock_responses: false  # Set to true for testing without API calls
```

### Testing

```elixir
defmodule MyApp.LLMTest do
  use ExUnit.Case
  alias Object.LLMIntegration

  test "generates appropriate response" do
    # Use mocked responses in tests
    Application.put_env(:object, :dspy, mock_responses: true)
    
    object = Object.new(subtype: :ai_agent)
    message = %{sender: "test", content: "Hello"}
    
    {:ok, response, _updated_object} = LLMIntegration.generate_response(object, message)
    
    assert response.content != ""
    assert response.confidence > 0
  end
end
```

This guide provides a comprehensive overview of integrating DSPy with AAOS objects. Start with the basic examples and gradually incorporate more advanced features as needed.
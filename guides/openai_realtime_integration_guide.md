# OpenAI Realtime API Integration Guide for AAOS

## Overview

This guide explains how AAOS objects can consume the OpenAI OpenAPI specification and transform into fully autonomous agents with realtime API capabilities.

## Architecture

The integration consists of several key components:

### 1. OpenAI Scaffold (`Object.OpenAIScaffold`)

The scaffold is a self-building system that:
- Parses OpenAI's OpenAPI specification (YAML/JSON)
- Dynamically generates Elixir modules for each endpoint
- Creates type-safe schema modules
- Builds an intelligent API client

```elixir
# Start the scaffold
{:ok, _pid} = Object.OpenAIScaffold.start_link()

# Scaffold from OpenAI spec
{:ok, summary} = Object.OpenAIScaffold.scaffold_from_spec(
  "https://raw.githubusercontent.com/openai/openai-openapi/master/openapi.yaml"
)
```

### 2. OpenAI Client (`Object.OpenAIClient`)

A production-ready OpenAI API client that:
- Implements the same interface as MockLMStudio for compatibility
- Supports chat completions, function calling, and streaming
- Integrates with DSPy for structured reasoning
- Handles retries and error recovery

```elixir
# Configure OpenAI
Application.put_env(:object, :dspy_bridge, [
  provider: :openai,
  openai: [
    api_key: System.get_env("OPENAI_API_KEY"),
    model: "gpt-4"
  ]
])
```

### 3. Autonomous Agent Creation

The scaffold can create autonomous agents with:
- Self-directed reasoning capabilities
- Dynamic tool selection
- Learning from interactions
- Multi-agent collaboration

```elixir
# Create an autonomous agent
agent_config = %{
  name: "AI Assistant",
  model: "gpt-4",
  capabilities: [:chat, :function_calling, :reasoning],
  initial_tools: [
    %{
      name: "analyze_code",
      description: "Analyze code and suggest improvements",
      parameters: %{
        properties: %{
          "code" => %{"type" => "string"}
        }
      }
    }
  ]
}

{:ok, agent_id, agent_pid} = Object.OpenAIScaffold.create_agent(agent_config)
```

## How It Works

### 1. Specification Parsing

The scaffold downloads and parses the OpenAI OpenAPI spec:

```yaml
paths:
  /chat/completions:
    post:
      operationId: createChatCompletion
      parameters: [...]
      requestBody: [...]
      responses: [...]
```

### 2. Dynamic Module Generation

For each endpoint, the scaffold generates:

```elixir
defmodule Object.OpenAI.POST.ChatCompletions do
  def execute(params, options \\ []) do
    Object.OpenAIScaffold.APIClient.request(
      :post,
      "/chat/completions",
      params,
      options
    )
  end
end
```

### 3. Agent Reasoning

Agents use DSPy signatures for structured reasoning:

```elixir
signature = %{
  name: "OpenAIReasoning",
  inputs: ["query", "available_endpoints"],
  outputs: ["reasoning", "selected_endpoint", "parameters"],
  instructions: "Reason about the best API endpoint to use"
}

{:ok, result} = DSPyBridge.execute_signature(signature, inputs)
```

### 4. Self-Modification

Agents can modify their own behavior:
- Learn from successful/failed API calls
- Update tool selection strategies
- Evolve reasoning patterns

## Usage Examples

### Basic Chat Completion

```elixir
# Agent automatically selects chat completion endpoint
response = GenServer.call(agent_pid, {:reason, "Tell me a joke"})
```

### Function Calling

```elixir
# Agent uses function calling for complex tasks
response = GenServer.call(agent_pid, {:reason, 
  "Analyze this Elixir code and generate tests for it: #{code}"
})
```

### Multi-Agent Collaboration

```elixir
# Create specialized agents
{:ok, coder_id, _} = create_agent(%{name: "Coder", specialization: :code_generation})
{:ok, tester_id, _} = create_agent(%{name: "Tester", specialization: :test_generation})

# Agents collaborate automatically
Object.send_message(coder_id, %{
  action: :collaborate,
  with: tester_id,
  task: "Create a web scraper with comprehensive tests"
})
```

## Configuration

### Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_MODEL="gpt-4"  # or gpt-3.5-turbo
```

### Application Config

```elixir
# config/config.exs
config :object, :openai_scaffold,
  auto_generate_modules: true,
  enable_caching: true,
  max_retries: 3
```

## Advanced Features

### 1. Streaming Responses

```elixir
# Enable streaming for real-time responses
agent_config = %{
  streaming: true,
  on_token: fn token -> IO.write(token) end
}
```

### 2. Custom Tools

```elixir
# Add custom tools to agents
Object.OpenAIScaffold.add_tool(agent_id, %{
  name: "database_query",
  description: "Query the application database",
  parameters: %{
    properties: %{
      "query" => %{"type" => "string", "description" => "SQL query"}
    }
  },
  handler: fn params -> 
    # Custom implementation
    execute_query(params["query"])
  end
})
```

### 3. Learning and Adaptation

```elixir
# Enable continuous learning
Object.OpenAIScaffold.enable_learning(agent_id, %{
  store_interactions: true,
  learn_from_feedback: true,
  adaptation_rate: 0.1
})
```

## Best Practices

1. **API Key Security**: Never commit API keys. Use environment variables.
2. **Rate Limiting**: The client handles rate limits automatically.
3. **Error Handling**: Always handle potential API failures gracefully.
4. **Cost Management**: Monitor token usage with agent metrics.
5. **Caching**: Enable caching to reduce API calls and costs.

## Troubleshooting

### Common Issues

1. **API Key Not Found**
   ```elixir
   # Check configuration
   Application.get_env(:object, :dspy_bridge)[:openai][:api_key]
   ```

2. **Module Generation Fails**
   ```elixir
   # Enable debug logging
   Logger.configure(level: :debug)
   ```

3. **Agent Not Responding**
   ```elixir
   # Check agent status
   Process.alive?(agent_pid)
   ```

## Integration with AAOS Features

The OpenAI integration seamlessly works with other AAOS components:

- **OORL Learning**: Agents can use reinforcement learning
- **Hierarchical Coordination**: Agents can form hierarchies
- **Byzantine Fault Tolerance**: Multi-agent consensus
- **P2P Networking**: Distributed agent networks

## Future Enhancements

1. **Vision API Integration**: Support for image analysis
2. **Audio/Speech**: Realtime voice interactions
3. **Fine-tuning**: Custom model training
4. **Embeddings**: Semantic search and retrieval
5. **Assistants API**: Persistent agent states

## Conclusion

The OpenAI scaffold transforms static API specifications into living, autonomous agents that can reason, learn, and collaborate. This represents a new paradigm in AI integration where APIs become self-aware and self-improving.

For more examples, see `/examples/openai_realtime_agent_demo.exs`.
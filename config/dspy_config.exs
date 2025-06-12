# DSPy Integration Configuration for AAOS Objects
# This file shows how to configure DSPy integration for LLM-powered object responses

import Config

# DSPy Configuration
config :object, :dspy,
  # LM Studio Configuration (local LLM server)
  lm_studio: %{
    base_url: "http://localhost:1234",
    timeout: 30_000,
    streaming: true,
    model: "llama-3.1-8b-instruct",
    max_tokens: 2048,
    temperature: 0.7
  },
  
  # OpenAI Configuration (alternative)
  openai: %{
    api_key: System.get_env("OPENAI_API_KEY"),
    model: "gpt-4o-mini",
    max_tokens: 1500,
    temperature: 0.7,
    timeout: 30_000
  },
  
  # Anthropic Claude Configuration (alternative)
  anthropic: %{
    api_key: System.get_env("ANTHROPIC_API_KEY"),
    model: "claude-3-haiku-20240307",
    max_tokens: 1500,
    temperature: 0.7
  },
  
  # Default LLM Provider (choose one: :lm_studio, :openai, :anthropic)
  default_provider: :openai,
  
  # DSPy Framework Settings
  framework: %{
    cache_enabled: true,
    cache_ttl: 3600,  # 1 hour
    retry_attempts: 3,
    retry_delay: 1000,  # 1 second
    concurrent_requests: 5,
    optimization_enabled: true
  },
  
  # Signature Templates Configuration
  signature_templates: %{
    # Pre-defined signatures for common object interactions
    response_generation: %{
      max_length: 200,
      style_options: [:professional, :casual, :technical, :friendly],
      default_style: :professional
    },
    
    goal_reasoning: %{
      max_steps: 10,
      include_risk_assessment: true,
      include_alternatives: true,
      confidence_threshold: 0.6
    },
    
    collaborative_reasoning: %{
      max_participants: 5,
      consensus_threshold: 0.7,
      synthesis_enabled: true
    }
  },
  
  # Performance Monitoring
  monitoring: %{
    enabled: true,
    log_requests: true,
    log_responses: false,  # Set to true for debugging
    track_latency: true,
    track_token_usage: true,
    alert_on_errors: true
  }

# Object-specific LLM Configurations
config :object, :llm_personalities,
  ai_agent: %{
    system_prompt: """
    You are an advanced AI assistant object in an autonomous agent system. 
    You are helpful, analytical, and focused on problem-solving. 
    Communicate clearly and provide actionable insights.
    """,
    temperature: 0.7,
    style: :professional,
    max_response_length: 300
  },
  
  sensor_object: %{
    system_prompt: """
    You are a sensor object that collects and analyzes environmental data.
    Be precise, factual, and data-driven in your responses.
    Focus on accuracy and reliability of information.
    """,
    temperature: 0.3,
    style: :technical,
    max_response_length: 200
  },
  
  coordinator_object: %{
    system_prompt: """
    You are a coordination object responsible for managing resources and organizing tasks.
    Be efficient, organized, and strategic in your communication.
    Focus on optimization and clear delegation.
    """,
    temperature: 0.5,
    style: :professional,
    max_response_length: 250
  },
  
  human_client: %{
    system_prompt: """
    You represent a human user in the system.
    Be curious, ask clarifying questions, and express human-like concerns and interests.
    """,
    temperature: 0.8,
    style: :casual,
    max_response_length: 150
  }

# Development and Testing Configuration
if Mix.env() == :dev do
  config :object, :dspy,
    # Use mock responses for development
    mock_responses: true,
    mock_delay: 100,  # Simulate network delay
    
    # Enhanced logging for development
    debug_mode: true,
    log_level: :debug
end

if Mix.env() == :test do
  config :object, :dspy,
    # Always use mocks in tests
    mock_responses: true,
    mock_delay: 0,
    
    # Disable external API calls
    external_calls_enabled: false
end

# Production Configuration
if Mix.env() == :prod do
  config :object, :dspy,
    # Production LLM settings
    cache_enabled: true,
    optimization_enabled: true,
    
    # Error handling
    fallback_enabled: true,
    graceful_degradation: true,
    
    # Security
    rate_limiting: %{
      requests_per_minute: 100,
      burst_limit: 20
    },
    
    # Monitoring
    telemetry_enabled: true,
    metrics_collection: true
end
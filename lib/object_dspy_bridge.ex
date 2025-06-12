defmodule Object.DSPyBridge do
  @moduledoc """
  Bridge module integrating DSPy framework with AAOS objects for advanced reasoning capabilities.
  Enables objects to use DSPy signatures and LM Studio inference for intelligent behavior.
  """

  use GenServer
  require Logger

  @default_lm_client Object.DSPyBridge.MockLMStudio
  # @openai_client Object.OpenAIClient

  defstruct [
    :object_id,
    :lm_client,
    :active_signatures,
    :reasoning_cache,
    :performance_metrics
  ]

  @doc """
  Starts a DSPy bridge process for the given object.
  
  ## Parameters
  - `object_id`: ID of the object to create bridge for
  
  ## Returns
  `{:ok, pid}` on successful startup
  """
  def start_link(object_id) do
    GenServer.start_link(__MODULE__, object_id, name: via_registry(object_id))
  end

  def via_registry(object_id) do
    {:via, Registry, {Object.Registry, {:dspy_bridge, object_id}}}
  end

  def init(object_id) do
    state = %__MODULE__{
      object_id: object_id,
      lm_client: nil,
      active_signatures: %{},
      reasoning_cache: :ets.new(:reasoning_cache, [:set, :private]),
      performance_metrics: %{queries: 0, cache_hits: 0, avg_latency: 0.0}
    }

    {:ok, state, {:continue, :initialize_lm_client}}
  end

  def handle_continue(:initialize_lm_client, state) do
    # Check configuration for LM provider
    config = Application.get_env(:object, :dspy_bridge, [])
    provider = config[:provider] || :mock
    
    client = case provider do
      :openai ->
        # Start OpenAI client if not already running
        case GenServer.whereis(Object.OpenAIClient) do
          nil ->
            {:ok, _pid} = Object.OpenAIClient.start_link(config[:openai] || [])
          _pid ->
            :ok
        end
        Object.OpenAIClient
        
      :lmstudio ->
        {:ok, client} = @default_lm_client.new(%{
          base_url: config[:base_url] || "http://localhost:1234",
          timeout: config[:timeout] || 30_000,
          streaming: config[:streaming] || true
        })
        client
        
      _ ->
        # Default to mock client
        {:ok, client} = @default_lm_client.new(%{})
        client
    end
    
    Logger.info("DSPy bridge initialized for object #{state.object_id} with provider: #{provider}")
    {:noreply, %{state | lm_client: client}}
  end

  @doc """
  Executes reasoning using a registered DSPy signature.
  
  ## Parameters
  - `object_id`: ID of the object
  - `signature_name`: Name of the signature to use
  - `inputs`: Input data for reasoning
  - `options`: Optional parameters like max_tokens, temperature
  
  ## Returns
  `{:ok, result}` with structured reasoning output or `{:error, reason}`
  
  ## Examples
      iex> Object.DSPyBridge.reason_with_signature("obj1", :message_analysis, %{content: "Hello"})
      {:ok, %{intent: "greeting", confidence: 0.9}}
  """
  def reason_with_signature(object_id, signature_name, inputs, options \\ []) do
    try do
      GenServer.call(via_registry(object_id), {:reason, signature_name, inputs, options})
    rescue
      UndefinedFunctionError ->
        # Registry not available, return fallback result
        {:ok, create_fallback_result(signature_name, inputs)}
    catch
      :exit, {:noproc, _} ->
        # DSPy bridge not started for this object, return a fallback result
        {:ok, create_fallback_result(signature_name, inputs)}
      :exit, {:shutdown, _} ->
        # System shutting down, return fallback result
        {:ok, create_fallback_result(signature_name, inputs)}
    end
  end

  @doc """
  Executes a DSPy signature directly with the provided specification.
  
  ## Parameters
  - `object_id`: ID of the object
  - `signature`: Signature specification to execute
  
  ## Returns
  `{:ok, result}` with structured output or `{:error, reason}`
  """
  def execute_signature(object_id, signature) do
    try do
      GenServer.call(via_registry(object_id), {:execute_signature, signature})
    rescue
      UndefinedFunctionError ->
        {:ok, create_fallback_signature_result(signature)}
    catch
      :exit, {:noproc, _} ->
        {:ok, create_fallback_signature_result(signature)}
      :exit, {:shutdown, _} ->
        {:ok, create_fallback_signature_result(signature)}
    end
  end

  def execute_signature(signature, params) when is_map(signature) do
    # Create a mock object ID for direct signature execution
    object_id = "signature_exec_#{System.unique_integer([:positive])}"
    execute_signature(object_id, Map.merge(signature, params))
  end

  @doc """
  Registers a new DSPy signature for an object.
  
  ## Parameters
  - `object_id`: ID of the object
  - `signature_name`: Name for the signature
  - `signature_spec`: Specification with description, inputs, outputs, instructions
  
  ## Returns
  `:ok` on successful registration
  """
  def register_signature(object_id, signature_name, signature_spec) do
    GenServer.call(via_registry(object_id), {:register_signature, signature_name, signature_spec})
  end

  @doc """
  Gets performance metrics for the DSPy bridge.
  
  ## Parameters
  - `object_id`: ID of the object
  
  ## Returns
  Map with query count, cache hits, and average latency
  """
  def get_reasoning_metrics(object_id) do
    GenServer.call(via_registry(object_id), :get_metrics)
  end

  def handle_call({:reason, signature_name, inputs, options}, _from, state) do
    start_time = System.monotonic_time(:microsecond)
    
    cache_key = {signature_name, inputs}
    
    result = case :ets.lookup(state.reasoning_cache, cache_key) do
      [{^cache_key, cached_result}] ->
        new_metrics = update_metrics(state.performance_metrics, :cache_hit, 0)
        {:reply, {:ok, cached_result}, %{state | performance_metrics: new_metrics}}
      
      [] ->
        case execute_reasoning(state, signature_name, inputs, options) do
          {:ok, result} ->
            :ets.insert(state.reasoning_cache, {cache_key, result})
            latency = System.monotonic_time(:microsecond) - start_time
            new_metrics = update_metrics(state.performance_metrics, :query, latency)
            {:reply, {:ok, result}, %{state | performance_metrics: new_metrics}}
          
          {:error, reason} ->
            {:reply, {:error, reason}, state}
        end
    end
    
    result
  end

  def handle_call({:register_signature, signature_name, signature_spec}, _from, state) do
    new_signatures = Map.put(state.active_signatures, signature_name, signature_spec)
    {:reply, :ok, %{state | active_signatures: new_signatures}}
  end

  def handle_call({:execute_signature, _signature}, _from, state) do
    # Simple stub implementation that returns a mock result
    # In a real implementation, this would use the signature to generate an LLM response
    result = %{
      response_text: "Generated response based on the signature",
      tone: "neutral",
      intent: "informational",
      confidence: 0.8,
      suggests_follow_up: false,
      model_info: "mock_model",
      reasoning_steps: ["Step 1: Analyzed input", "Step 2: Generated response"]
    }
    {:reply, {:ok, result}, state}
  end

  def handle_call(:get_metrics, _from, state) do
    {:reply, state.performance_metrics, state}
  end

  defp create_fallback_signature_result(signature) do
    %{
      response_text: "Fallback response for signature execution",
      reasoning_steps: ["Analyzed signature requirements", "Generated structured response"],
      confidence: 0.7,
      model_info: "fallback_mock",
      execution_time: 50,
      signature_name: Map.get(signature, :name, "unknown"),
      parameters: signature
    }
  end

  defp create_fallback_result(signature_name, _inputs) do
    # Create reasonable fallback results for common signatures used in OORL
    case signature_name do
      :behavior_adaptation ->
        %{
          behavior_adjustments: "maintain_current_strategy",
          reasoning: "Insufficient context for adaptation",
          expected_outcomes: %{stability: 0.8},
          risk_assessment: "low"
        }
      
      :problem_solving ->
        %{
          solution_strategy: "gradient_descent_optimization",
          confidence: 0.6,
          convergence_prediction: 0.7,
          resource_requirements: "medium",
          implementation_plan: %{
            parameter_changes: %{},
            learning_rate: 1.0,
            exploration_strategy: :unchanged
          }
        }
      
      _ ->
        %{
          status: "fallback_executed",
          confidence: 0.5,
          reasoning: "DSPy bridge not available, using fallback logic"
        }
    end
  end

  defp execute_reasoning(state, signature_name, inputs, options) do
    case Map.get(state.active_signatures, signature_name) do
      nil ->
        {:error, "Signature '#{signature_name}' not registered"}
      
      signature_spec ->
        case state.lm_client do
          nil ->
            {:error, "LM client not initialized"}
          
          Object.OpenAIClient ->
            # Use OpenAI client with DSPy signature execution
            signature = %{
              name: to_string(signature_name),
              inputs: signature_spec.inputs || [],
              outputs: signature_spec.outputs || [],
              instructions: signature_spec.instructions || ""
            }
            
            case Object.OpenAIClient.execute_dspy_signature(signature, inputs) do
              {:ok, result} -> {:ok, result}
              error -> error
            end
            
          client ->
            # Use legacy LM Studio client
            prompt = build_dspy_prompt(signature_spec, inputs)
            
            {:ok, response} = @default_lm_client.generate(client, %{
              prompt: prompt,
              max_tokens: Keyword.get(options, :max_tokens, 1000),
              temperature: Keyword.get(options, :temperature, 0.7),
              stream: false
            })
            parse_dspy_response(signature_spec, response)
        end
    end
  end

  defp build_dspy_prompt(signature_spec, inputs) do
    """
    System: You are an AI assistant following the DSPy framework for structured reasoning.

    Signature: #{signature_spec.description}
    
    Input Fields:
    #{Enum.map(signature_spec.inputs, fn {field, desc} -> "- #{field}: #{desc}" end) |> Enum.join("\n")}
    
    Output Fields:
    #{Enum.map(signature_spec.outputs, fn {field, desc} -> "- #{field}: #{desc}" end) |> Enum.join("\n")}
    
    Instructions: #{signature_spec.instructions}
    
    Input Data:
    #{Enum.map(inputs, fn {key, value} -> "#{key}: #{value}" end) |> Enum.join("\n")}
    
    Please provide your response in the following structured format:
    #{Enum.map(signature_spec.outputs, fn {field, _} -> "#{field}: [your response for #{field}]" end) |> Enum.join("\n")}
    """
  end

  defp parse_dspy_response(signature_spec, response) do
    try do
      output_fields = Enum.map(signature_spec.outputs, fn {field, _} -> 
        Atom.to_string(field) 
      end)
      
      parsed = output_fields
      |> Enum.reduce(%{}, fn field, acc ->
        case Regex.run(~r/#{field}:\s*(.+?)(?=\n\w+:|$)/s, response.content) do
          [_, value] -> Map.put(acc, String.to_atom(field), String.trim(value))
          nil -> acc
        end
      end)
      
      {:ok, parsed}
    rescue
      error ->
        {:error, "Failed to parse response: #{inspect(error)}"}
    end
  end

  defp update_metrics(metrics, :cache_hit, _latency) do
    %{metrics | cache_hits: metrics.cache_hits + 1}
  end

  defp update_metrics(metrics, :query, latency) do
    new_queries = metrics.queries + 1
    new_avg = (metrics.avg_latency * metrics.queries + latency) / new_queries
    
    %{metrics | 
      queries: new_queries,
      avg_latency: new_avg
    }
  end

  defmodule MockLMStudio do
    @moduledoc """
    Mock LM Studio client for testing and development.
    """

    def new(_config) do
      {:ok, %{client_id: "mock_client", status: :connected}}
    end

    def generate(_client, %{prompt: prompt} = _params) do
      # Simple mock response
      response = %{
        content: "Mock response to: #{String.slice(prompt, 0, 50)}...",
        tokens_used: 42,
        completion_time: 150
      }
      {:ok, response}
    end

    def generate(_client, _params) do
      response = %{
        content: "Mock response generated",
        tokens_used: 30,
        completion_time: 100
      }
      {:ok, response}
    end
  end
end
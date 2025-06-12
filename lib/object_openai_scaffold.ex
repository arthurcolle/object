defmodule Object.OpenAIScaffold do
  @moduledoc """
  Self-scaffolding object that consumes OpenAI OpenAPI specification
  and generates an autonomous agent with full API capabilities.
  
  This module transforms OpenAPI specs into living, breathing AAOS objects
  that can reason about and execute API operations autonomously.
  """
  
  use GenServer
  require Logger
  alias Object.{DSPyBridge, LLMIntegration, FunctionCalling}
  
  defstruct [
    :spec,
    :endpoints,
    :schemas,
    :auth_config,
    :generated_modules,
    :active_agents,
    :api_client
  ]
  
  # Public API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def scaffold_from_spec(spec_url_or_path) do
    GenServer.call(__MODULE__, {:scaffold, spec_url_or_path}, :infinity)
  end
  
  def create_agent(agent_config) do
    GenServer.call(__MODULE__, {:create_agent, agent_config})
  end
  
  # GenServer callbacks
  
  def init(_opts) do
    state = %__MODULE__{
      spec: nil,
      endpoints: %{},
      schemas: %{},
      auth_config: nil,
      generated_modules: [],
      active_agents: %{},
      api_client: nil
    }
    
    {:ok, state}
  end
  
  def handle_call({:scaffold, spec_source}, _from, state) do
    Logger.info("Starting OpenAI API scaffolding from: #{spec_source}")
    
    with {:ok, spec} <- load_spec(spec_source),
         {:ok, parsed} <- parse_spec(spec),
         {:ok, modules} <- generate_modules(parsed),
         {:ok, client} <- create_api_client(parsed.auth_config) do
      
      new_state = %{state | 
        spec: spec,
        endpoints: parsed.endpoints,
        schemas: parsed.schemas,
        auth_config: parsed.auth_config,
        generated_modules: modules,
        api_client: client
      }
      
      {:reply, {:ok, summarize_scaffolding(new_state)}, new_state}
    else
      error -> {:reply, error, state}
    end
  end
  
  def handle_call({:create_agent, config}, _from, state) do
    agent_id = generate_agent_id()
    
    agent_spec = %{
      id: agent_id,
      type: :openai_agent,
      capabilities: extract_capabilities(state.endpoints),
      reasoning_engine: :dspy,
      memory: %{
        conversation_history: [],
        learned_patterns: [],
        tool_usage_stats: %{}
      },
      config: Map.merge(default_agent_config(), config)
    }
    
    case spawn_autonomous_agent(agent_spec, state) do
      {:ok, agent_pid} ->
        new_state = put_in(state.active_agents[agent_id], agent_pid)
        {:reply, {:ok, agent_id, agent_pid}, new_state}
      error ->
        {:reply, error, state}
    end
  end
  
  # Private functions
  
  defp load_spec(source) when is_binary(source) do
    cond do
      String.starts_with?(source, "http") ->
        fetch_remote_spec(source)
      File.exists?(source) ->
        load_local_spec(source)
      true ->
        {:error, "Invalid spec source: #{source}"}
    end
  end
  
  defp fetch_remote_spec(url) do
    case HTTPoison.get(url) do
      {:ok, %{status_code: 200, body: body}} ->
        parse_yaml_or_json(body)
      error ->
        {:error, "Failed to fetch spec: #{inspect(error)}"}
    end
  end
  
  defp load_local_spec(path) do
    case File.read(path) do
      {:ok, content} -> parse_yaml_or_json(content)
      error -> error
    end
  end
  
  defp parse_yaml_or_json(content) do
    # Try YAML first, then JSON
    case YamlElixir.read_from_string(content) do
      {:ok, spec} -> {:ok, spec}
      _ ->
        case Jason.decode(content) do
          {:ok, spec} -> {:ok, spec}
          _ -> {:error, "Failed to parse spec as YAML or JSON"}
        end
    end
  end
  
  defp parse_spec(spec) do
    try do
      parsed = %{
        endpoints: extract_endpoints(spec),
        schemas: extract_schemas(spec),
        auth_config: extract_auth_config(spec),
        metadata: extract_metadata(spec)
      }
      
      {:ok, parsed}
    rescue
      e -> {:error, "Failed to parse spec: #{Exception.message(e)}"}
    end
  end
  
  defp extract_endpoints(spec) do
    paths = Map.get(spec, "paths", %{})
    
    Enum.reduce(paths, %{}, fn {path, methods}, acc ->
      Enum.reduce(methods, acc, fn {method, config}, acc2 ->
        endpoint_key = "#{String.upcase(method)}_#{path}"
        endpoint_info = %{
          path: path,
          method: String.upcase(method),
          operation_id: config["operationId"],
          summary: config["summary"],
          parameters: parse_parameters(config),
          request_body: parse_request_body(config),
          responses: parse_responses(config),
          tags: config["tags"] || []
        }
        
        Map.put(acc2, endpoint_key, endpoint_info)
      end)
    end)
  end
  
  defp parse_parameters(config) do
    params = config["parameters"] || []
    
    Enum.map(params, fn param ->
      %{
        name: param["name"],
        in: param["in"],
        required: param["required"] || false,
        schema: param["schema"],
        description: param["description"]
      }
    end)
  end
  
  defp parse_request_body(config) do
    case config["requestBody"] do
      nil -> nil
      body ->
        %{
          required: body["required"] || false,
          content: body["content"],
          description: body["description"]
        }
    end
  end
  
  defp parse_responses(config) do
    responses = config["responses"] || %{}
    
    Enum.reduce(responses, %{}, fn {status, response}, acc ->
      Map.put(acc, status, %{
        description: response["description"],
        content: response["content"],
        headers: response["headers"]
      })
    end)
  end
  
  defp extract_schemas(spec) do
    components = Map.get(spec, "components", %{})
    Map.get(components, "schemas", %{})
  end
  
  defp extract_auth_config(spec) do
    components = Map.get(spec, "components", %{})
    security_schemes = Map.get(components, "securitySchemes", %{})
    
    # For OpenAI, we expect ApiKeyAuth
    case Map.get(security_schemes, "ApiKeyAuth") do
      nil -> nil
      scheme ->
        %{
          type: scheme["type"],
          in: scheme["in"],
          name: scheme["name"],
          api_key: System.get_env("OPENAI_API_KEY")
        }
    end
  end
  
  defp extract_metadata(spec) do
    %{
      title: get_in(spec, ["info", "title"]),
      version: get_in(spec, ["info", "version"]),
      description: get_in(spec, ["info", "description"]),
      servers: spec["servers"] || []
    }
  end
  
  defp generate_modules(parsed) do
    try do
      modules = []
      
      # Generate endpoint modules
      endpoint_modules = generate_endpoint_modules(parsed.endpoints, parsed.schemas)
      modules = modules ++ endpoint_modules
      
      # Generate schema modules
      schema_modules = generate_schema_modules(parsed.schemas)
      modules = modules ++ schema_modules
      
      # Generate client module
      client_module = generate_client_module(parsed)
      modules = [client_module | modules]
      
      {:ok, modules}
    rescue
      e -> {:error, "Failed to generate modules: #{Exception.message(e)}"}
    end
  end
  
  defp generate_endpoint_modules(endpoints, schemas) do
    Enum.map(endpoints, fn {key, endpoint} ->
      module_name = endpoint_to_module_name(key)
      
      module_code = """
      defmodule #{module_name} do
        @moduledoc \"\"\"
        Auto-generated module for #{endpoint.method} #{endpoint.path}
        #{endpoint.summary}
        \"\"\"
        
        def execute(params, options \\ []) do
          Object.OpenAIScaffold.APIClient.request(
            :#{String.downcase(endpoint.method)},
            "#{endpoint.path}",
            params,
            options
          )
        end
        
        def execute_async(params, options \\ []) do
          Task.async(fn -> execute(params, options) end)
        end
        
        def schema do
          #{inspect(endpoint, pretty: true)}
        end
      end
      """
      
      compile_module(module_code)
      module_name
    end)
  end
  
  defp generate_schema_modules(schemas) do
    Enum.map(schemas, fn {name, schema} ->
      module_name = schema_to_module_name(name)
      
      module_code = """
      defmodule #{module_name} do
        @moduledoc \"\"\"
        Auto-generated schema module for #{name}
        \"\"\"
        
        use TypedStruct
        
        typedstruct do
          #{generate_struct_fields(schema)}
        end
        
        def validate(data) do
          # Add validation logic based on schema
          {:ok, struct(__MODULE__, data)}
        end
      end
      """
      
      compile_module(module_code)
      module_name
    end)
  end
  
  defp generate_client_module(parsed) do
    module_code = """
    defmodule Object.OpenAIScaffold.APIClient do
      @moduledoc \"\"\"
      Auto-generated API client for OpenAI
      \"\"\"
      
      def request(method, path, params, options \\ []) do
        # Implementation will use the actual HTTP client
        # with auth config from the scaffold
        Object.OpenAIScaffold.execute_request(method, path, params, options)
      end
    end
    """
    
    compile_module(module_code)
    "Object.OpenAIScaffold.APIClient"
  end
  
  defp compile_module(code) do
    # In production, this would compile the module
    # For now, we'll store the code
    Logger.debug("Generated module code: #{code}")
    code
  end
  
  defp endpoint_to_module_name(key) do
    parts = String.split(key, "_")
    method = List.first(parts)
    path_parts = parts |> List.delete_at(0) |> Enum.join("_")
    
    camelized = path_parts
    |> String.split("/")
    |> Enum.filter(&(&1 != ""))
    |> Enum.map(&Macro.camelize/1)
    |> Enum.join("")
    
    "Object.OpenAI.#{method}.#{camelized}"
  end
  
  defp schema_to_module_name(name) do
    "Object.OpenAI.Schema.#{Macro.camelize(name)}"
  end
  
  defp generate_struct_fields(schema) do
    properties = schema["properties"] || %{}
    required = schema["required"] || []
    
    Enum.map(properties, fn {name, prop} ->
      type_spec = infer_elixir_type(prop)
      required_spec = if name in required, do: ", enforce: true", else: ""
      
      "field :#{name}, #{type_spec}#{required_spec}"
    end)
    |> Enum.join("\n    ")
  end
  
  defp infer_elixir_type(prop) do
    case prop["type"] do
      "string" -> "String.t()"
      "integer" -> "integer()"
      "number" -> "float()"
      "boolean" -> "boolean()"
      "array" -> "list()"
      "object" -> "map()"
      _ -> "any()"
    end
  end
  
  defp create_api_client(auth_config) do
    # Create HTTP client with auth configuration
    client_config = %{
      base_url: "https://api.openai.com/v1",
      headers: build_auth_headers(auth_config),
      timeout: 30_000,
      retry_config: %{
        max_retries: 3,
        retry_delay: 1000
      }
    }
    
    {:ok, client_config}
  end
  
  defp build_auth_headers(nil), do: []
  defp build_auth_headers(auth_config) do
    case auth_config.type do
      "http" ->
        [{"Authorization", "Bearer #{auth_config.api_key}"}]
      _ ->
        []
    end
  end
  
  defp spawn_autonomous_agent(agent_spec, scaffold_state) do
    # Create an AAOS object with OpenAI capabilities
    object_spec = %{
      name: "openai_agent_#{agent_spec.id}",
      type: :ai_agent,
      state: %{
        agent_spec: agent_spec,
        scaffold: scaffold_state,
        conversation_history: [],
        active_tools: []
      },
      goals: [
        "Provide intelligent responses using OpenAI API",
        "Learn from interactions to improve performance",
        "Autonomously select and use appropriate tools",
        "Maintain context across conversations"
      ],
      methods: %{
        # Core reasoning method
        reason: fn object, query ->
          reason_about_query(object, query, scaffold_state)
        end,
        
        # Tool selection method
        select_tools: fn object, context ->
          select_relevant_tools(object, context, scaffold_state.endpoints)
        end,
        
        # API execution method
        execute_api: fn object, endpoint, params ->
          execute_api_call(object, endpoint, params, scaffold_state)
        end,
        
        # Learning method
        learn_from_interaction: fn object, interaction ->
          update_learning_state(object, interaction)
        end
      }
    }
    
    Object.Supervisor.create_object(object_spec)
  end
  
  defp reason_about_query(object, query, scaffold_state) do
    # Use DSPy bridge for reasoning
    signature = %{
      name: "OpenAIReasoning",
      inputs: ["query", "available_endpoints", "conversation_history"],
      outputs: ["reasoning", "selected_endpoint", "parameters"],
      instructions: """
      Given a user query, available OpenAI endpoints, and conversation history,
      reason about the best approach to handle the query. Select the appropriate
      endpoint and generate necessary parameters.
      """
    }
    
    inputs = %{
      "query" => query,
      "available_endpoints" => summarize_endpoints(scaffold_state.endpoints),
      "conversation_history" => get_recent_history(object)
    }
    
    case DSPyBridge.execute_signature(signature, inputs) do
      {:ok, result} ->
        {:ok, %{
          reasoning: result["reasoning"],
          endpoint: result["selected_endpoint"],
          params: parse_parameters_from_reasoning(result["parameters"])
        }}
      error ->
        error
    end
  end
  
  defp select_relevant_tools(object, context, endpoints) do
    # Filter endpoints based on context
    relevant = Enum.filter(endpoints, fn {_key, endpoint} ->
      context_matches_endpoint?(context, endpoint)
    end)
    
    # Rank by relevance
    ranked = Enum.sort_by(relevant, fn {_key, endpoint} ->
      calculate_relevance_score(context, endpoint)
    end, &>=/2)
    
    # Return top tools
    Enum.take(ranked, 5)
  end
  
  defp execute_api_call(object, endpoint_key, params, scaffold_state) do
    endpoint = scaffold_state.endpoints[endpoint_key]
    
    # Build request
    request = build_request(endpoint, params, scaffold_state.api_client)
    
    # Execute with retry logic
    case execute_with_retry(request) do
      {:ok, response} ->
        # Update object state with successful call
        update_object_state(object, :api_success, {endpoint_key, response})
        {:ok, response}
        
      {:error, reason} ->
        # Update object state with failure
        update_object_state(object, :api_failure, {endpoint_key, reason})
        {:error, reason}
    end
  end
  
  defp update_learning_state(object, interaction) do
    # Extract patterns from interaction
    patterns = extract_interaction_patterns(interaction)
    
    # Update object's learned patterns
    Object.update_state(object, fn state ->
      learned = Map.get(state, :learned_patterns, [])
      %{state | learned_patterns: [patterns | learned]}
    end)
    
    # Update tool usage statistics
    if interaction[:tool_used] do
      Object.update_state(object, fn state ->
        stats = Map.get(state, :tool_usage_stats, %{})
        tool = interaction.tool_used
        count = Map.get(stats, tool, 0) + 1
        %{state | tool_usage_stats: Map.put(stats, tool, count)}
      end)
    end
  end
  
  defp summarize_scaffolding(state) do
    %{
      endpoints_count: map_size(state.endpoints),
      schemas_count: map_size(state.schemas),
      generated_modules: length(state.generated_modules),
      auth_configured: state.auth_config != nil,
      sample_endpoints: state.endpoints |> Map.keys() |> Enum.take(5)
    }
  end
  
  defp generate_agent_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
  
  defp default_agent_config do
    %{
      model: "gpt-4",
      temperature: 0.7,
      max_tokens: 2000,
      stream: false,
      auto_retry: true,
      learning_enabled: true
    }
  end
  
  defp summarize_endpoints(endpoints) do
    Enum.map(endpoints, fn {key, endpoint} ->
      "#{key}: #{endpoint.summary}"
    end)
    |> Enum.join("\n")
  end
  
  defp get_recent_history(object) do
    Object.get_state(object)
    |> Map.get(:conversation_history, [])
    |> Enum.take(-10)
  end
  
  defp parse_parameters_from_reasoning(param_string) do
    # Parse parameters from LLM reasoning output
    case Jason.decode(param_string) do
      {:ok, params} -> params
      _ -> %{}
    end
  end
  
  defp extract_capabilities(endpoints) do
    # Extract capabilities from endpoints map
    endpoints
    |> Enum.map(fn {_key, endpoint} ->
      %{
        operation: endpoint.operation_id || "#{endpoint.method}_#{endpoint.path}",
        method: endpoint.method,
        path: endpoint.path,
        summary: endpoint.summary,
        tags: endpoint.tags
      }
    end)
    |> Enum.uniq()
  end
  
  defp context_matches_endpoint?(context, endpoint) do
    # Simple keyword matching for now
    context_lower = String.downcase(context)
    
    endpoint_text = "#{endpoint.path} #{endpoint.summary} #{Enum.join(endpoint.tags, " ")}"
    |> String.downcase()
    
    String.contains?(endpoint_text, context_lower) or
    String.contains?(context_lower, endpoint_text)
  end
  
  defp calculate_relevance_score(context, endpoint) do
    # More sophisticated scoring
    base_score = 0.0
    
    # Check for exact operation match
    if String.contains?(context, endpoint.operation_id || ""), do: base_score + 0.5
    
    # Check tags
    tag_score = Enum.count(endpoint.tags, fn tag ->
      String.contains?(String.downcase(context), String.downcase(tag))
    end) * 0.2
    
    # Check summary
    summary_score = if String.contains?(String.downcase(context), 
                                      String.downcase(endpoint.summary || "")), 
                    do: 0.3, else: 0
    
    base_score + tag_score + summary_score
  end
  
  defp build_request(endpoint, params, client_config) do
    %{
      method: String.to_atom(String.downcase(endpoint.method)),
      url: "#{client_config.base_url}#{endpoint.path}",
      headers: client_config.headers,
      body: build_request_body(endpoint, params),
      options: [
        timeout: client_config.timeout,
        recv_timeout: client_config.timeout
      ]
    }
  end
  
  defp build_request_body(endpoint, params) do
    case endpoint.request_body do
      nil -> ""
      _ -> Jason.encode!(params)
    end
  end
  
  defp execute_with_retry(request, retries \\ 3) do
    case HTTPoison.request(
      request.method,
      request.url,
      request.body,
      request.headers,
      request.options
    ) do
      {:ok, %{status_code: code, body: body}} when code in 200..299 ->
        {:ok, Jason.decode!(body)}
        
      {:ok, %{status_code: code}} when code in 500..599 and retries > 0 ->
        Process.sleep(1000)
        execute_with_retry(request, retries - 1)
        
      {:ok, response} ->
        {:error, "API error: #{response.status_code} - #{response.body}"}
        
      {:error, reason} when retries > 0 ->
        Process.sleep(1000)
        execute_with_retry(request, retries - 1)
        
      error ->
        error
    end
  end
  
  defp update_object_state(object, event, data) do
    Object.update_state(object, fn state ->
      history = Map.get(state, :event_history, [])
      event_record = %{
        event: event,
        data: data,
        timestamp: DateTime.utc_now()
      }
      %{state | event_history: [event_record | history]}
    end)
  end
  
  defp extract_interaction_patterns(interaction) do
    %{
      query_type: classify_query(interaction.query),
      endpoint_used: interaction.endpoint,
      success: interaction.success,
      response_time: interaction.response_time,
      tokens_used: interaction.tokens_used
    }
  end
  
  defp classify_query(query) do
    cond do
      String.contains?(query, ["create", "generate", "make"]) -> :creative
      String.contains?(query, ["analyze", "explain", "understand"]) -> :analytical
      String.contains?(query, ["chat", "talk", "converse"]) -> :conversational
      String.contains?(query, ["code", "program", "function"]) -> :technical
      true -> :general
    end
  end
  
  # Public execution function for API calls
  def execute_request(method, path, params, options) do
    GenServer.call(__MODULE__, {:execute_request, method, path, params, options})
  end
  
  def handle_call({:execute_request, method, path, params, options}, _from, state) do
    request = %{
      method: method,
      url: "#{state.api_client.base_url}#{path}",
      headers: state.api_client.headers ++ Keyword.get(options, :headers, []),
      body: Jason.encode!(params),
      options: [
        timeout: Keyword.get(options, :timeout, state.api_client.timeout),
        recv_timeout: Keyword.get(options, :timeout, state.api_client.timeout)
      ]
    }
    
    result = execute_with_retry(request)
    {:reply, result, state}
  end
end
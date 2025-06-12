defmodule Object.OpenAIClient do
  @moduledoc """
  OpenAI API client implementation for AAOS.
  Implements the same interface as MockLMStudio for drop-in replacement.
  """
  
  use GenServer
  require Logger
  
  @base_url "https://api.openai.com/v1"
  @default_model "gpt-4"
  @default_timeout 60_000
  
  defstruct [
    :api_key,
    :model,
    :temperature,
    :max_tokens,
    :base_url,
    :timeout,
    :http_client
  ]
  
  # Client API
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def completion(prompt, opts \\ []) do
    GenServer.call(__MODULE__, {:completion, prompt, opts}, @default_timeout)
  end
  
  def chat_completion(messages, opts \\ []) do
    GenServer.call(__MODULE__, {:chat_completion, messages, opts}, @default_timeout)
  end
  
  def function_call(messages, functions, opts \\ []) do
    GenServer.call(__MODULE__, {:function_call, messages, functions, opts}, @default_timeout)
  end
  
  # GenServer callbacks
  
  def init(opts) do
    state = %__MODULE__{
      api_key: opts[:api_key] || System.get_env("OPENAI_API_KEY"),
      model: opts[:model] || @default_model,
      temperature: opts[:temperature] || 0.7,
      max_tokens: opts[:max_tokens] || 2000,
      base_url: opts[:base_url] || @base_url,
      timeout: opts[:timeout] || @default_timeout,
      http_client: HTTPoison
    }
    
    if is_nil(state.api_key) do
      Logger.error("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    end
    
    {:ok, state}
  end
  
  def handle_call({:completion, prompt, opts}, _from, state) do
    # Convert to chat format for consistency
    messages = [%{"role" => "user", "content" => prompt}]
    
    case make_chat_request(messages, opts, state) do
      {:ok, response} ->
        # Extract content from chat response
        content = get_in(response, ["choices", Access.at(0), "message", "content"]) || ""
        {:reply, {:ok, content}, state}
        
      error ->
        {:reply, error, state}
    end
  end
  
  def handle_call({:chat_completion, messages, opts}, _from, state) do
    case make_chat_request(messages, opts, state) do
      {:ok, response} ->
        {:reply, {:ok, response}, state}
      error ->
        {:reply, error, state}
    end
  end
  
  def handle_call({:function_call, messages, functions, opts}, _from, state) do
    # Add function calling to request
    opts_with_functions = Keyword.put(opts, :functions, functions)
    
    case make_chat_request(messages, opts_with_functions, state) do
      {:ok, response} ->
        # Extract function call if present
        case get_in(response, ["choices", Access.at(0), "message", "function_call"]) do
          nil ->
            # No function call, return regular message
            content = get_in(response, ["choices", Access.at(0), "message", "content"]) || ""
            {:reply, {:ok, %{type: :message, content: content}}, state}
            
          function_call ->
            # Parse function call
            {:reply, {:ok, %{
              type: :function_call,
              name: function_call["name"],
              arguments: parse_arguments(function_call["arguments"])
            }}, state}
        end
        
      error ->
        {:reply, error, state}
    end
  end
  
  # Private functions
  
  defp make_chat_request(messages, opts, state) do
    model = Keyword.get(opts, :model, state.model)
    temperature = Keyword.get(opts, :temperature, state.temperature)
    max_tokens = Keyword.get(opts, :max_tokens, state.max_tokens)
    
    body = %{
      "model" => model,
      "messages" => format_messages(messages),
      "temperature" => temperature,
      "max_tokens" => max_tokens
    }
    
    # Add functions if provided
    body = case Keyword.get(opts, :functions) do
      nil -> body
      functions -> Map.put(body, "functions", format_functions(functions))
    end
    
    # Add function_call if specified
    body = case Keyword.get(opts, :function_call) do
      nil -> body
      "auto" -> Map.put(body, "function_call", "auto")
      "none" -> Map.put(body, "function_call", "none")
      specific -> Map.put(body, "function_call", %{"name" => specific})
    end
    
    # Add streaming if requested
    body = if Keyword.get(opts, :stream, false) do
      Map.put(body, "stream", true)
    else
      body
    end
    
    headers = [
      {"Authorization", "Bearer #{state.api_key}"},
      {"Content-Type", "application/json"}
    ]
    
    request_opts = [
      timeout: state.timeout,
      recv_timeout: state.timeout
    ]
    
    url = "#{state.base_url}/chat/completions"
    
    Logger.debug("Making OpenAI request to #{url}")
    
    case state.http_client.post(url, Jason.encode!(body), headers, request_opts) do
      {:ok, %{status_code: 200, body: response_body}} ->
        case Jason.decode(response_body) do
          {:ok, parsed} -> {:ok, parsed}
          error -> {:error, "Failed to parse response: #{inspect(error)}"}
        end
        
      {:ok, %{status_code: status, body: error_body}} ->
        error_info = case Jason.decode(error_body) do
          {:ok, parsed} -> parsed["error"]["message"] || "Unknown error"
          _ -> error_body
        end
        {:error, "OpenAI API error (#{status}): #{error_info}"}
        
      {:error, %HTTPoison.Error{reason: reason}} ->
        {:error, "HTTP request failed: #{inspect(reason)}"}
        
      error ->
        {:error, "Unexpected error: #{inspect(error)}"}
    end
  end
  
  defp format_messages(messages) when is_list(messages) do
    Enum.map(messages, &format_message/1)
  end
  
  defp format_message(%{"role" => _role, "content" => _content} = msg), do: msg
  defp format_message(%{role: role, content: content}) do
    %{"role" => to_string(role), "content" => content}
  end
  defp format_message(content) when is_binary(content) do
    %{"role" => "user", "content" => content}
  end
  
  defp format_functions(functions) when is_list(functions) do
    Enum.map(functions, &format_function/1)
  end
  
  defp format_function(func) when is_map(func) do
    %{
      "name" => func[:name] || func["name"],
      "description" => func[:description] || func["description"] || "",
      "parameters" => format_parameters(func[:parameters] || func["parameters"] || %{})
    }
  end
  
  defp format_parameters(params) when is_map(params) do
    # Ensure parameters have the correct JSON Schema format
    base = %{
      "type" => "object",
      "properties" => params["properties"] || params[:properties] || %{},
      "required" => params["required"] || params[:required] || []
    }
    
    base
  end
  
  defp parse_arguments(args) when is_binary(args) do
    case Jason.decode(args) do
      {:ok, parsed} -> parsed
      _ -> %{}
    end
  end
  defp parse_arguments(args), do: args
  
  # Compatibility layer for DSPyBridge interface
  
  def execute_dspy_signature(signature, inputs) do
    # Convert DSPy signature to OpenAI chat format
    system_prompt = build_system_prompt(signature)
    user_prompt = build_user_prompt(signature, inputs)
    
    messages = [
      %{"role" => "system", "content" => system_prompt},
      %{"role" => "user", "content" => user_prompt}
    ]
    
    case chat_completion(messages) do
      {:ok, response} ->
        # Parse structured output from response
        content = get_in(response, ["choices", Access.at(0), "message", "content"]) || ""
        parse_dspy_response(content, signature.outputs)
        
      error ->
        error
    end
  end
  
  defp build_system_prompt(signature) do
    """
    #{signature.instructions}
    
    You must respond with a JSON object containing the following fields:
    #{Enum.map(signature.outputs, fn output -> "- #{output}: <#{output}_value>" end) |> Enum.join("\n")}
    
    Be precise and follow the instructions exactly.
    """
  end
  
  defp build_user_prompt(_signature, inputs) do
    input_text = Enum.map(inputs, fn {key, value} ->
      "#{key}: #{value}"
    end) |> Enum.join("\n\n")
    
    """
    Given the following inputs:
    
    #{input_text}
    
    Please provide the requested outputs in JSON format.
    """
  end
  
  defp parse_dspy_response(content, expected_outputs) do
    # Try to extract JSON from the response
    case extract_json(content) do
      {:ok, json} ->
        # Validate all expected outputs are present
        missing = Enum.filter(expected_outputs, fn output ->
          not Map.has_key?(json, output)
        end)
        
        if Enum.empty?(missing) do
          {:ok, json}
        else
          {:error, "Missing outputs: #{Enum.join(missing, ", ")}"}
        end
        
      _ ->
        # Fallback: try to parse as plain text
        {:ok, %{"response" => content}}
    end
  end
  
  defp extract_json(content) do
    # Try to find JSON in the content
    case Regex.run(~r/\{[^{}]*\}/, content) do
      [json_str] ->
        Jason.decode(json_str)
      _ ->
        # Try parsing the entire content
        Jason.decode(content)
    end
  end
  
  # Streaming support
  
  def stream_chat_completion(messages, opts \\ []) do
    opts_with_stream = Keyword.put(opts, :stream, true)
    GenServer.call(__MODULE__, {:stream_chat, messages, opts_with_stream}, :infinity)
  end
  
  def handle_call({:stream_chat, messages, opts}, from, state) do
    # Start async task for streaming
    Task.start(fn ->
      result = stream_request(messages, opts, state)
      GenServer.reply(from, result)
    end)
    
    {:noreply, state}
  end
  
  defp stream_request(_messages, _opts, _state) do
    # Implementation for SSE streaming
    # This would use a streaming HTTP client to handle Server-Sent Events
    {:error, "Streaming not yet implemented"}
  end
  
  # Function calling helpers
  
  def create_function_schema(name, description, parameters) do
    %{
      "name" => name,
      "description" => description,
      "parameters" => %{
        "type" => "object",
        "properties" => parameters.properties,
        "required" => parameters.required || []
      }
    }
  end
  
  def execute_function_result(function_name, result, opts \\ []) do
    # Continue conversation with function result
    messages = [
      %{
        "role" => "function",
        "name" => function_name,
        "content" => Jason.encode!(result)
      }
    ]
    
    chat_completion(messages, opts)
  end
end
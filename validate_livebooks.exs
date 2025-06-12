# Livebook Validation Script
# Validates that all livebooks contain working Elixir code

IO.puts("📝 Validating AAOS Livebook Files...")

defmodule LivebookValidator do
  def validate_livebook(file_path) do
    IO.puts("\n🔍 Validating: #{Path.basename(file_path)}")
    
    case File.read(file_path) do
      {:ok, content} ->
        elixir_blocks = extract_elixir_blocks(content)
        IO.puts("   Found #{length(elixir_blocks)} Elixir code blocks")
        
        validate_code_blocks(elixir_blocks, Path.basename(file_path))
        
      {:error, reason} ->
        IO.puts("   ❌ Failed to read file: #{reason}")
        false
    end
  end
  
  defp extract_elixir_blocks(content) do
    # Extract code blocks between ```elixir and ```
    regex = ~r/```elixir\n(.*?)\n```/s
    
    Regex.scan(regex, content, capture: :all_but_first)
    |> Enum.map(&hd/1)
    |> Enum.reject(&(&1 == ""))
  end
  
  defp validate_code_blocks(blocks, filename) do
    all_valid = Enum.all?(blocks, fn block ->
      validate_single_block(block, filename)
    end)
    
    if all_valid do
      IO.puts("   ✅ All code blocks are syntactically valid")
    else
      IO.puts("   ⚠️  Some code blocks have syntax issues")
    end
    
    all_valid
  end
  
  defp validate_single_block(code, filename) do
    try do
      Code.string_to_quoted!(code)
      true
    rescue
      error ->
        IO.puts("   ❌ Syntax error in #{filename}:")
        IO.puts("      #{inspect(error)}")
        IO.puts("      Code preview: #{String.slice(code, 0, 100)}...")
        false
    end
  end
  
  def validate_key_concepts(file_path) do
    {:ok, content} = File.read(file_path)
    filename = Path.basename(file_path)
    
    # Check for key AAOS concepts based on filename
    concepts = case filename do
      "aaos_basics.livemd" ->
        ["defmodule BasicObject", "send_message", "receive_message", "learning", "goals"]
      
      "meta_schema_evolution.livemd" ->
        ["defmodule Schema", "evolve_schema", "meta_learning", "adaptation", "evolution_history"]
      
      "oorl_learning.livemd" ->
        ["OORLAgent", "social_learning", "coalition", "reinforcement", "policy"]
      
      "collective_intelligence.livemd" ->
        ["SwarmAgent", "collective", "emergence", "coordination", "problem_solving"]
      
      _ ->
        []
    end
    
    found_concepts = Enum.filter(concepts, fn concept ->
      String.contains?(content, concept)
    end)
    
    IO.puts("   Key concepts found: #{length(found_concepts)}/#{length(concepts)}")
    
    if length(found_concepts) == length(concepts) do
      IO.puts("   ✅ All expected concepts present")
    else
      missing = concepts -- found_concepts
      IO.puts("   ⚠️  Missing concepts: #{Enum.join(missing, ", ")}")
    end
    
    length(found_concepts) / length(concepts)
  end
end

# Validate all livebook files
livebook_files = [
  "/Users/agent/object/notebooks/aaos_basics.livemd",
  "/Users/agent/object/notebooks/meta_schema_evolution.livemd", 
  "/Users/agent/object/notebooks/oorl_learning.livemd",
  "/Users/agent/object/notebooks/collective_intelligence.livemd"
]

validation_results = Enum.map(livebook_files, fn file ->
  syntax_valid = LivebookValidator.validate_livebook(file)
  concept_coverage = LivebookValidator.validate_key_concepts(file)
  
  %{
    file: Path.basename(file),
    syntax_valid: syntax_valid,
    concept_coverage: concept_coverage
  }
end)

IO.puts("\n📊 Validation Summary:")
IO.puts("=" |> String.duplicate(50))

total_files = length(validation_results)
syntax_valid_count = Enum.count(validation_results, & &1.syntax_valid)
avg_concept_coverage = validation_results
                      |> Enum.map(& &1.concept_coverage)
                      |> Enum.sum()
                      |> (fn total -> total / total_files end).()

IO.puts("Files validated: #{total_files}")
IO.puts("Syntax valid: #{syntax_valid_count}/#{total_files}")
IO.puts("Average concept coverage: #{Float.round(avg_concept_coverage * 100, 1)}%")

Enum.each(validation_results, fn result ->
  status = if result.syntax_valid and result.concept_coverage > 0.8, do: "✅", else: "⚠️"
  IO.puts("#{status} #{result.file}: syntax #{if result.syntax_valid, do: "OK", else: "FAIL"}, concepts #{Float.round(result.concept_coverage * 100, 1)}%")
end)

overall_success = syntax_valid_count == total_files and avg_concept_coverage > 0.8

if overall_success do
  IO.puts("\n🎉 All livebooks are valid and ready for use!")
  IO.puts("Users can run these interactively to explore AAOS concepts.")
else
  IO.puts("\n⚠️  Some livebooks need attention before deployment.")
end

IO.puts("\n🔗 Livebook Usage Instructions:")
IO.puts("1. Install Livebook: mix escript.install hex livebook")
IO.puts("2. Start Livebook: livebook server")
IO.puts("3. Open any .livemd file in the notebooks/ directory")
IO.puts("4. Run cells interactively to see AAOS in action!")
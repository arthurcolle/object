# Find and fix syntax errors in collective intelligence livebook

{:ok, content} = File.read("/Users/agent/object/notebooks/collective_intelligence.livemd")

# Extract elixir blocks
regex = ~r/```elixir\n(.*?)\n```/s
blocks = Regex.scan(regex, content, capture: :all_but_first) |> Enum.map(&hd/1)

IO.puts("Found #{length(blocks)} code blocks")

Enum.with_index(blocks, 1)
|> Enum.each(fn {block, index} ->
  IO.puts("\n=== Block #{index} ===")
  
  try do
    Code.string_to_quoted!(block)
    IO.puts("✅ Block #{index} is valid")
  rescue
    error ->
    IO.puts("❌ Block #{index} has error:")
    IO.puts("#{inspect(error)}")
    IO.puts("\nFirst 300 chars:")
    IO.puts(String.slice(block, 0, 300))
    IO.puts("...")
  end
end)
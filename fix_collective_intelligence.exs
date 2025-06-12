# Extract and test the problematic SwarmAgent module

code = """
defmodule SwarmAgent do
  defstruct [
    :id,
    :position,
    :velocity,
    :energy,
    :local_knowledge,
    :communication_range,
    :swarm_connections,
    :role,
    :behavioral_state,
    :coordination_memory
  ]
  
  def new(id, position) do
    %__MODULE__{
      id: id,
      position: position,
      velocity: {0.0, 0.0},
      energy: 100,
      local_knowledge: %{
        explored_areas: MapSet.new(),
        resource_locations: [],
        danger_zones: [],
        optimal_paths: %{}
      },
      communication_range: 5.0,
      swarm_connections: MapSet.new(),
      role: :explorer,
      behavioral_state: :searching,
      coordination_memory: []
    }
  end
end
"""

try do
  Code.string_to_quoted!(code)
  IO.puts("✅ SwarmAgent definition is valid")
rescue
  error ->
    IO.puts("❌ SwarmAgent has error:")
    IO.puts(inspect(error))
end
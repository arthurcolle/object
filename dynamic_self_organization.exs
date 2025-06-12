#!/usr/bin/env elixir

# Dynamic Self-Organization Demo - Watch structures emerge in real-time
# Run with: elixir dynamic_self_organization.exs

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule DynamicSelfOrg do
  @moduledoc """
  Shows dramatic self-organization with visual network formation
  """

  defmodule Agent do
    defstruct [:id, :type, :position, :energy, :connections, :cluster_id, :specialization_level, :age]
    
    def new(id) do
      types = [:explorer, :builder, :connector, :optimizer, :leader]
      %Agent{
        id: id,
        type: Enum.random(types),
        position: {Enum.random(1..50), Enum.random(1..50)},
        energy: Enum.random(50..100),
        connections: [],
        cluster_id: nil,
        specialization_level: 0,
        age: 0
      }
    end
    
    def move_towards(agent, target_pos, step_size \\ 2) do
      {x, y} = agent.position
      {tx, ty} = target_pos
      
      dx = if tx > x, do: min(step_size, tx - x), else: max(-step_size, tx - x)
      dy = if ty > y, do: min(step_size, ty - y), else: max(-step_size, ty - y)
      
      %{agent | position: {x + dx, y + dy}}
    end
    
    def age_agent(agent) do
      # Agents become more specialized over time
      specialization_boost = if agent.age > 5, do: 1, else: 0
      %{agent | 
        age: agent.age + 1, 
        specialization_level: agent.specialization_level + specialization_boost
      }
    end
  end

  defmodule World do
    defstruct [:agents, :generation, :clusters, :global_connections, :phase]
    
    def new(num_agents) do
      agents = 
        1..num_agents
        |> Enum.map(&Agent.new/1)
        |> Map.new(&{&1.id, &1})
      
      %World{
        agents: agents,
        generation: 0,
        clusters: [],
        global_connections: [],
        phase: :exploration
      }
    end
    
    def evolve(world) do
      world
      |> update_phase()
      |> move_agents()
      |> form_connections()
      |> create_clusters()
      |> age_agents()
      |> Map.update!(:generation, &(&1 + 1))
    end
    
    defp update_phase(world) do
      phase = cond do
        world.generation < 3 -> :exploration
        world.generation < 8 -> :clustering
        world.generation < 12 -> :specialization
        true -> :optimization
      end
      %{world | phase: phase}
    end
    
    defp move_agents(world) do
      case world.phase do
        :exploration ->
          # Random movement
          updated_agents = Map.new(world.agents, fn {id, agent} ->
            new_pos = {
              max(1, min(50, elem(agent.position, 0) + Enum.random(-3..3))),
              max(1, min(50, elem(agent.position, 1) + Enum.random(-3..3)))
            }
            {id, %{agent | position: new_pos}}
          end)
          %{world | agents: updated_agents}
          
        :clustering ->
          # Move towards similar agents
          updated_agents = Map.new(world.agents, fn {id, agent} ->
            similar_agents = find_similar_agents(world, agent)
            if length(similar_agents) > 0 do
              target = Enum.random(similar_agents)
              {id, Agent.move_towards(agent, target.position)}
            else
              {id, agent}
            end
          end)
          %{world | agents: updated_agents}
          
        _ -> world
      end
    end
    
    defp find_similar_agents(world, agent) do
      world.agents
      |> Map.values()
      |> Enum.filter(fn other -> 
        other.id != agent.id and other.type == agent.type
      end)
    end
    
    defp form_connections(world) do
      new_connections = 
        world.agents
        |> Map.values()
        |> Enum.flat_map(fn agent ->
          nearby_agents = get_nearby_agents(world, agent, connection_radius(world.phase))
          
          nearby_agents
          |> Enum.filter(&should_connect?(agent, &1, world.phase))
          |> Enum.map(&{agent.id, &1.id})
        end)
        |> Enum.uniq()
      
      # Update agent connections
      connection_map = 
        new_connections
        |> Enum.reduce(%{}, fn {a, b}, acc ->
          acc
          |> Map.update(a, [b], &[b | &1])
          |> Map.update(b, [a], &[a | &1])
        end)
      
      updated_agents = Map.new(world.agents, fn {id, agent} ->
        connections = Map.get(connection_map, id, []) |> Enum.uniq()
        {id, %{agent | connections: connections}}
      end)
      
      %{world | agents: updated_agents, global_connections: new_connections}
    end
    
    defp connection_radius(:exploration), do: 8
    defp connection_radius(:clustering), do: 12
    defp connection_radius(:specialization), do: 15
    defp connection_radius(:optimization), do: 20
    
    defp should_connect?(agent1, agent2, phase) do
      case phase do
        :exploration -> 
          # Connect randomly based on proximity
          :rand.uniform() < 0.3
          
        :clustering -> 
          # Connect to similar types
          agent1.type == agent2.type or :rand.uniform() < 0.2
          
        :specialization ->
          # Connect complementary types
          complementary_types?(agent1.type, agent2.type) or agent1.type == agent2.type
          
        :optimization ->
          # Connect high-energy agents
          (agent1.energy + agent2.energy) > 120
      end
    end
    
    defp complementary_types?(:explorer, :builder), do: true
    defp complementary_types?(:builder, :connector), do: true
    defp complementary_types?(:connector, :optimizer), do: true
    defp complementary_types?(:optimizer, :leader), do: true
    defp complementary_types?(:leader, :explorer), do: true
    defp complementary_types?(a, b), do: complementary_types?(b, a)
    
    defp get_nearby_agents(world, agent, radius) do
      {x, y} = agent.position
      
      world.agents
      |> Map.values()
      |> Enum.filter(fn other ->
        {ox, oy} = other.position
        other.id != agent.id and
        :math.sqrt(:math.pow(x - ox, 2) + :math.pow(y - oy, 2)) <= radius
      end)
    end
    
    defp create_clusters(world) do
      # Find clusters using connected components
      clusters = find_clusters(world.global_connections, Map.keys(world.agents))
      
      # Assign cluster IDs to agents
      cluster_assignments = 
        clusters
        |> Enum.with_index()
        |> Enum.flat_map(fn {cluster_agents, idx} ->
          Enum.map(cluster_agents, &{&1, idx})
        end)
        |> Map.new()
      
      updated_agents = Map.new(world.agents, fn {id, agent} ->
        cluster_id = Map.get(cluster_assignments, id)
        {id, %{agent | cluster_id: cluster_id}}
      end)
      
      cluster_info = 
        clusters
        |> Enum.with_index()
        |> Enum.map(fn {cluster_agents, idx} ->
          agents = Enum.map(cluster_agents, &world.agents[&1])
          types = agents |> Enum.map(& &1.type) |> Enum.frequencies()
          
          %{
            id: idx,
            size: length(cluster_agents),
            agents: cluster_agents,
            dominant_type: types |> Enum.max_by(&elem(&1, 1)) |> elem(0),
            diversity: map_size(types),
            avg_energy: agents |> Enum.map(& &1.energy) |> Enum.sum() |> div(length(agents)),
            center: calculate_center(agents)
          }
        end)
        |> Enum.filter(&(&1.size > 1))  # Only keep multi-agent clusters
      
      %{world | agents: updated_agents, clusters: cluster_info}
    end
    
    defp find_clusters(connections, all_agents) do
      graph = build_graph(connections)
      find_connected_components(graph, all_agents)
    end
    
    defp build_graph(connections) do
      Enum.reduce(connections, %{}, fn {a, b}, graph ->
        graph
        |> Map.update(a, [b], &[b | &1])
        |> Map.update(b, [a], &[a | &1])
      end)
    end
    
    defp find_connected_components(graph, all_nodes) do
      {_, components} = 
        Enum.reduce(all_nodes, {MapSet.new(), []}, fn node, {visited, components} ->
          if MapSet.member?(visited, node) do
            {visited, components}
          else
            component = dfs(graph, node, MapSet.new()) |> MapSet.to_list()
            {MapSet.union(visited, MapSet.new(component)), [component | components]}
          end
        end)
      
      components
    end
    
    defp dfs(graph, node, visited) do
      if MapSet.member?(visited, node) do
        visited
      else
        visited = MapSet.put(visited, node)
        neighbors = Map.get(graph, node, [])
        
        Enum.reduce(neighbors, visited, fn neighbor, acc ->
          dfs(graph, neighbor, acc)
        end)
      end
    end
    
    defp calculate_center(agents) do
      positions = Enum.map(agents, & &1.position)
      avg_x = positions |> Enum.map(&elem(&1, 0)) |> Enum.sum() |> div(length(positions))
      avg_y = positions |> Enum.map(&elem(&1, 1)) |> Enum.sum() |> div(length(positions))
      {avg_x, avg_y}
    end
    
    defp age_agents(world) do
      updated_agents = Map.new(world.agents, fn {id, agent} ->
        {id, Agent.age_agent(agent)}
      end)
      %{world | agents: updated_agents}
    end
  end

  def run_demo() do
    IO.puts("\n🌟 DYNAMIC SELF-ORGANIZATION DEMO")
    IO.puts("=" <> String.duplicate("=", 60))
    IO.puts("Watch autonomous agents discover, cluster, specialize & optimize!")
    
    world = World.new(15)
    
    IO.puts("\n🎬 Starting with #{map_size(world.agents)} random agents...")
    display_world_state(world)
    
    # Evolution loop with dramatic changes
    final_world = 
      1..15
      |> Enum.reduce(world, fn gen, current_world ->
        Process.sleep(800)  # Pause for dramatic effect
        
        evolved_world = World.evolve(current_world)
        
        IO.puts("\n" <> String.duplicate("-", 60))
        IO.puts("🔄 GENERATION #{gen} - PHASE: #{String.upcase(to_string(evolved_world.phase))}")
        IO.puts(String.duplicate("-", 60))
        
        display_phase_behavior(evolved_world)
        display_world_state(evolved_world)
        display_network_visualization(evolved_world)
        
        if dramatic_change?(current_world, evolved_world) do
          IO.puts("💥 MAJOR STRUCTURAL CHANGE DETECTED!")
        end
        
        evolved_world
      end)
    
    IO.puts("\n🎉 FINAL SELF-ORGANIZED STATE")
    IO.puts("=" <> String.duplicate("=", 60))
    analyze_final_structure(final_world)
  end
  
  defp display_phase_behavior(world) do
    case world.phase do
      :exploration -> 
        IO.puts("🔍 Agents exploring and making random connections...")
      :clustering -> 
        IO.puts("🏘️  Agents clustering with similar types...")
      :specialization -> 
        IO.puts("⚡ Agents specializing and forming complementary bonds...")
      :optimization -> 
        IO.puts("🚀 High-energy agents optimizing network structure...")
    end
  end
  
  defp display_world_state(world) do
    type_counts = 
      world.agents
      |> Map.values()
      |> Enum.map(& &1.type)
      |> Enum.frequencies()
    
    cluster_count = length(world.clusters)
    connection_count = length(world.global_connections)
    
    IO.puts("📊 Connections: #{connection_count} | Clusters: #{cluster_count}")
    IO.puts("🤖 Agent Types: #{format_type_counts(type_counts)}")
    
    if cluster_count > 0 do
      largest_cluster = Enum.max_by(world.clusters, & &1.size)
      IO.puts("🏆 Largest Cluster: #{largest_cluster.size} agents (#{largest_cluster.dominant_type}s)")
      
      specialized_clusters = Enum.count(world.clusters, &(&1.diversity == 1))
      IO.puts("⚡ Specialized Clusters: #{specialized_clusters}/#{cluster_count}")
    end
  end
  
  defp format_type_counts(counts) do
    counts
    |> Enum.map(fn {type, count} -> "#{type}:#{count}" end)
    |> Enum.join(" ")
  end
  
  defp display_network_visualization(world) do
    if length(world.clusters) > 1 do
      IO.puts("\n🕸️  Network Structure:")
      
      world.clusters
      |> Enum.take(3)  # Show top 3 clusters
      |> Enum.each(fn cluster ->
        specialization = if cluster.diversity == 1, do: "SPECIALIZED", else: "MIXED"
        energy_level = cond do
          cluster.avg_energy > 80 -> "HIGH"
          cluster.avg_energy > 60 -> "MED"
          true -> "LOW"
        end
        
        IO.puts("  └─ #{cluster.dominant_type} cluster (#{cluster.size}) [#{specialization}/#{energy_level}]")
      end)
    end
  end
  
  defp dramatic_change?(old_world, new_world) do
    cluster_change = abs(length(new_world.clusters) - length(old_world.clusters))
    connection_change = abs(length(new_world.global_connections) - length(old_world.global_connections))
    
    cluster_change >= 2 or connection_change >= 5
  end
  
  defp analyze_final_structure(world) do
    display_world_state(world)
    
    IO.puts("\n🔬 EMERGENT STRUCTURE ANALYSIS:")
    
    # Specialization analysis
    specialized_clusters = Enum.count(world.clusters, &(&1.diversity == 1))
    total_clusters = length(world.clusters)
    
    if specialized_clusters > 0 do
      specialization_ratio = (specialized_clusters / max(total_clusters, 1) * 100) |> round()
      IO.puts("✅ Specialization: #{specialization_ratio}% of clusters are specialized")
    end
    
    # Network efficiency
    total_possible_connections = div(map_size(world.agents) * (map_size(world.agents) - 1), 2)
    actual_connections = length(world.global_connections)
    efficiency = (actual_connections / total_possible_connections * 100) |> round()
    
    IO.puts("✅ Network Efficiency: #{efficiency}% connectivity")
    
    # Hub detection
    connection_counts = 
      world.global_connections
      |> Enum.flat_map(fn {a, b} -> [a, b] end)
      |> Enum.frequencies()
    
    hubs = Enum.count(connection_counts, fn {_, count} -> count >= 4 end)
    IO.puts("✅ Hub Formation: #{hubs} coordination hubs emerged")
    
    # Energy distribution
    energies = world.agents |> Map.values() |> Enum.map(& &1.energy)
    avg_energy = Enum.sum(energies) / length(energies) |> round()
    IO.puts("✅ Average Energy: #{avg_energy} (#{energy_assessment(avg_energy)})")
    
    IO.puts("\n💫 Self-organization created #{total_clusters} distinct communities")
    IO.puts("   from #{map_size(world.agents)} independent agents with NO central control!")
  end
  
  defp energy_assessment(avg) when avg > 80, do: "HIGH PERFORMANCE"
  defp energy_assessment(avg) when avg > 60, do: "STABLE"
  defp energy_assessment(_), do: "NEEDS OPTIMIZATION"
end

# Run the dynamic demo
DynamicSelfOrg.run_demo()
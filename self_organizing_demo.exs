#!/usr/bin/env elixir

# Self-Organizing System Demo
# Run with: elixir self_organizing_demo.exs

Mix.install([
  {:jason, "~> 1.4"}
])

defmodule SelfOrganizingDemo do
  @moduledoc """
  Demonstrates autonomous agents self-organizing into emergent structures
  """

  defmodule Agent do
    defstruct [:id, :type, :capabilities, :connections, :position, :resources, :reputation]
    
    def new(id, type) do
      %Agent{
        id: id,
        type: type,
        capabilities: generate_capabilities(type),
        connections: [],
        position: {Enum.random(1..100), Enum.random(1..100)},
        resources: Enum.random(10..50),
        reputation: 50
      }
    end
    
    defp generate_capabilities(:worker), do: [:compute, :execute]
    defp generate_capabilities(:coordinator), do: [:plan, :delegate, :coordinate]
    defp generate_capabilities(:specialist), do: [:analyze, :optimize, :innovate]
    defp generate_capabilities(:trader), do: [:negotiate, :trade, :value]
  end

  defmodule Network do
    defstruct [:agents, :connections, :clusters, :generation]
    
    def new() do
      %Network{
        agents: %{},
        connections: [],
        clusters: [],
        generation: 0
      }
    end
    
    def add_agent(network, agent) do
      %{network | agents: Map.put(network.agents, agent.id, agent)}
    end
    
    def get_nearby_agents(network, agent, radius \\ 20) do
      {x, y} = agent.position
      
      network.agents
      |> Map.values()
      |> Enum.filter(fn other ->
        {ox, oy} = other.position
        other.id != agent.id and 
        :math.sqrt(:math.pow(x - ox, 2) + :math.pow(y - oy, 2)) <= radius
      end)
    end
    
    def form_connections(network) do
      new_connections = 
        network.agents
        |> Map.values()
        |> Enum.flat_map(fn agent ->
          nearby = get_nearby_agents(network, agent)
          
          # Form connections based on complementary capabilities
          nearby
          |> Enum.filter(fn other ->
            has_complementary_capabilities?(agent, other) and
            connection_beneficial?(agent, other)
          end)
          |> Enum.map(fn other -> {agent.id, other.id} end)
        end)
        |> Enum.uniq()
      
      %{network | connections: network.connections ++ new_connections}
    end
    
    defp has_complementary_capabilities?(agent1, agent2) do
      common = MapSet.intersection(
        MapSet.new(agent1.capabilities),
        MapSet.new(agent2.capabilities)
      )
      MapSet.size(common) < 2  # Some overlap but not identical
    end
    
    defp connection_beneficial?(agent1, agent2) do
      # Connect if resources are complementary or reputation is high
      (agent1.resources + agent2.resources > 60) or
      (agent1.reputation + agent2.reputation > 120)
    end
    
    def detect_clusters(network) do
      # Use connected components to find emergent clusters
      graph = build_graph(network.connections)
      clusters = find_connected_components(graph, Map.keys(network.agents))
      
      cluster_info = 
        clusters
        |> Enum.with_index()
        |> Enum.map(fn {cluster_agents, idx} ->
          agents = Enum.map(cluster_agents, &network.agents[&1])
          
          %{
            id: idx,
            agents: cluster_agents,
            size: length(cluster_agents),
            types: agents |> Enum.map(& &1.type) |> Enum.frequencies(),
            capabilities: agents |> Enum.flat_map(& &1.capabilities) |> Enum.uniq(),
            avg_resources: agents |> Enum.map(& &1.resources) |> Enum.sum() |> div(length(agents)),
            center: calculate_center(agents)
          }
        end)
      
      %{network | clusters: cluster_info}
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
            component = dfs(graph, node, MapSet.new())
            {MapSet.union(visited, component), [MapSet.to_list(component) | components]}
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
    
    def evolve(network) do
      network
      |> form_connections()
      |> detect_clusters()
      |> optimize_structure()
      |> Map.update!(:generation, &(&1 + 1))
    end
    
    defp optimize_structure(network) do
      # Agents adapt based on cluster performance
      updated_agents = 
        Map.new(network.agents, fn {id, agent} ->
          cluster = find_agent_cluster(network, id)
          updated_agent = adapt_agent(agent, cluster, network)
          {id, updated_agent}
        end)
      
      %{network | agents: updated_agents}
    end
    
    defp find_agent_cluster(network, agent_id) do
      Enum.find(network.clusters, fn cluster ->
        agent_id in cluster.agents
      end)
    end
    
    defp adapt_agent(agent, nil, _network), do: agent
    defp adapt_agent(agent, cluster, _network) do
      # Increase reputation based on cluster success
      reputation_boost = min(cluster.size * 2, 20)
      
      # Specialization: gain capabilities common in cluster
      common_capabilities = 
        cluster.capabilities
        |> Enum.take(1)  # Take most common capability
      
      new_capabilities = (agent.capabilities ++ common_capabilities) |> Enum.uniq()
      
      %{agent | 
        reputation: min(agent.reputation + reputation_boost, 100),
        capabilities: new_capabilities
      }
    end
  end

  def run_demo() do
    IO.puts("\n🤖 SELF-ORGANIZING SYSTEM DEMO")
    IO.puts("=" <> String.duplicate("=", 50))
    
    # Initialize random agents
    agents = [
      Agent.new(1, :worker),
      Agent.new(2, :coordinator), 
      Agent.new(3, :specialist),
      Agent.new(4, :worker),
      Agent.new(5, :trader),
      Agent.new(6, :specialist),
      Agent.new(7, :coordinator),
      Agent.new(8, :worker),
      Agent.new(9, :trader),
      Agent.new(10, :specialist)
    ]
    
    # Create initial network
    network = 
      agents
      |> Enum.reduce(Network.new(), &Network.add_agent(&2, &1))
    
    IO.puts("\n📊 Initial State:")
    display_network_state(network)
    
    # Let the system self-organize over multiple generations
    IO.puts("\n🔄 Self-Organization Process:")
    
    final_network = 
      Enum.reduce(1..5, network, fn generation, net ->
        IO.puts("\n--- Generation #{generation} ---")
        evolved_net = Network.evolve(net)
        display_evolution(net, evolved_net)
        evolved_net
      end)
    
    IO.puts("\n🎯 Final Self-Organized State:")
    display_final_results(final_network)
    
    # Show emergent behaviors
    analyze_emergence(final_network)
  end
  
  defp display_network_state(network) do
    IO.puts("Agents: #{map_size(network.agents)}")
    IO.puts("Connections: #{length(network.connections)}")
    IO.puts("Clusters: #{length(network.clusters)}")
    
    type_counts = 
      network.agents
      |> Map.values()
      |> Enum.map(& &1.type)
      |> Enum.frequencies()
    
    IO.puts("Agent Types: #{inspect(type_counts)}")
  end
  
  defp display_evolution(old_network, new_network) do
    new_connections = length(new_network.connections) - length(old_network.connections)
    
    IO.puts("  New connections formed: #{new_connections}")
    IO.puts("  Clusters detected: #{length(new_network.clusters)}")
    
    if length(new_network.clusters) > 0 do
      largest_cluster = Enum.max_by(new_network.clusters, & &1.size)
      IO.puts("  Largest cluster: #{largest_cluster.size} agents")
      IO.puts("    Types: #{inspect(largest_cluster.types)}")
      IO.puts("    Capabilities: #{inspect(largest_cluster.capabilities)}")
    end
  end
  
  defp display_final_results(network) do
    display_network_state(network)
    
    IO.puts("\n🏘️  Emergent Clusters:")
    network.clusters
    |> Enum.with_index()
    |> Enum.each(fn {cluster, idx} ->
      IO.puts("  Cluster #{idx + 1}:")
      IO.puts("    Size: #{cluster.size} agents")
      IO.puts("    Specialization: #{inspect(cluster.types)}")
      IO.puts("    Capabilities: #{inspect(cluster.capabilities)}")
      IO.puts("    Avg Resources: #{cluster.avg_resources}")
      IO.puts("    Center: #{inspect(cluster.center)}")
    end)
  end
  
  defp analyze_emergence(network) do
    IO.puts("\n🌟 Emergent Behaviors Detected:")
    
    # Specialization emergence
    specialized_clusters = 
      network.clusters
      |> Enum.filter(fn cluster ->
        dominant_type_ratio = 
          cluster.types
          |> Map.values()
          |> Enum.max()
          |> div(cluster.size)
        
        dominant_type_ratio >= 0.6  # 60% or more of same type
      end)
    
    if length(specialized_clusters) > 0 do
      IO.puts("✓ Specialization: #{length(specialized_clusters)} specialized clusters formed")
    end
    
    # Hub emergence (high-connection agents)
    connection_counts = 
      network.connections
      |> Enum.flat_map(fn {a, b} -> [a, b] end)
      |> Enum.frequencies()
    
    hubs = 
      connection_counts
      |> Enum.filter(fn {_agent, count} -> count >= 3 end)
      |> Enum.map(fn {agent_id, count} -> 
        agent = network.agents[agent_id]
        {agent.type, count}
      end)
    
    if length(hubs) > 0 do
      IO.puts("✓ Hub Formation: #{length(hubs)} coordination hubs emerged")
      IO.puts("  Hub types: #{inspect(hubs)}")
    end
    
    # Resource concentration
    resource_variance = calculate_resource_variance(network)
    if resource_variance > 200 do
      IO.puts("✓ Resource Concentration: Wealth inequality emerged (variance: #{resource_variance})")
    end
    
    # Capability diversity
    total_unique_capabilities = 
      network.agents
      |> Map.values()
      |> Enum.flat_map(& &1.capabilities)
      |> Enum.uniq()
      |> length()
    
    IO.puts("✓ Capability Evolution: #{total_unique_capabilities} unique capabilities across network")
    
    IO.puts("\n🎉 Self-organization complete! The system has evolved emergent structures without central control.")
  end
  
  defp calculate_resource_variance(network) do
    resources = network.agents |> Map.values() |> Enum.map(& &1.resources)
    mean = Enum.sum(resources) / length(resources)
    
    variance = 
      resources
      |> Enum.map(fn r -> :math.pow(r - mean, 2) end)
      |> Enum.sum()
      |> Kernel./(length(resources))
    
    round(variance)
  end
end

# Run the demo
SelfOrganizingDemo.run_demo()
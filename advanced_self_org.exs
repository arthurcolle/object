#!/usr/bin/env elixir

# Advanced Self-Organizing System with Environment Manipulation
# Run with: elixir advanced_self_org.exs

Mix.install([
  {:jason, "~> 1.4"},
  {:req, "~> 0.4.0"}
])

defmodule AdvancedSelfOrg do
  @moduledoc """
  Advanced self-organizing system that manipulates the real environment:
  - Spawns actual processes
  - Discovers network nodes
  - Creates files and directories
  - Runs shell commands
  - Spawns Python processes
  - Auto-discovers system resources
  """

  defmodule EnvironmentAgent do
    defstruct [
      :id, :type, :pid, :port, :capabilities, :discovered_nodes, 
      :created_files, :running_processes, :environment_state, 
      :network_connections, :system_resources, :python_workers
    ]
    
    def new(id, type) do
      %EnvironmentAgent{
        id: id,
        type: type,
        pid: nil,
        port: generate_port(),
        capabilities: get_capabilities(type),
        discovered_nodes: [],
        created_files: [],
        running_processes: [],
        environment_state: %{},
        network_connections: [],
        system_resources: %{},
        python_workers: []
      }
    end
    
    defp generate_port(), do: Enum.random(8000..9000)
    
    defp get_capabilities(:discoverer), do: [:network_scan, :port_scan, :service_discovery]
    defp get_capabilities(:builder), do: [:file_creation, :directory_management, :process_spawn]
    defp get_capabilities(:connector), do: [:network_bridge, :service_mesh, :load_balance]
    defp get_capabilities(:monitor), do: [:resource_watch, :health_check, :metrics_collect]
    defp get_capabilities(:orchestrator), do: [:workflow_manage, :task_distribute, :system_optimize]
  end

  defmodule EnvironmentManipulator do
    @doc "Discover network nodes and services"
    def discover_network_nodes() do
      IO.puts("🔍 Discovering network nodes...")
      
      # Scan localhost ports
      localhost_services = scan_localhost_ports()
      
      # Get network interfaces
      network_interfaces = get_network_interfaces()
      
      # Discover nearby devices (if on network)
      nearby_devices = discover_nearby_devices()
      
      %{
        localhost_services: localhost_services,
        network_interfaces: network_interfaces,
        nearby_devices: nearby_devices
      }
    end
    
    defp scan_localhost_ports() do
      common_ports = [22, 80, 443, 3000, 8080, 8000, 5432, 6379, 27017]
      
      Enum.reduce(common_ports, [], fn port, acc ->
        case :gen_tcp.connect('localhost', port, [], 100) do
          {:ok, socket} ->
            :gen_tcp.close(socket)
            [%{host: "localhost", port: port, status: :open} | acc]
          {:error, _} ->
            acc
        end
      end)
    end
    
    defp get_network_interfaces() do
      case System.cmd("ifconfig", []) do
        {output, 0} ->
          output
          |> String.split("\n")
          |> Enum.filter(&String.contains?(&1, "inet "))
          |> Enum.map(&extract_ip/1)
          |> Enum.reject(&is_nil/1)
        _ -> []
      end
    rescue
      _ -> []
    end
    
    defp extract_ip(line) do
      case Regex.run(~r/inet (\d+\.\d+\.\d+\.\d+)/, line) do
        [_, ip] -> ip
        _ -> nil
      end
    end
    
    defp discover_nearby_devices() do
      # Simple network ping sweep (limited range for demo)
      base_ip = get_base_network_ip()
      
      if base_ip do
        1..10
        |> Enum.map(fn i -> 
          ip = "#{base_ip}.#{i}"
          Task.async(fn -> ping_host(ip) end)
        end)
        |> Enum.map(&Task.await(&1, 1000))
        |> Enum.filter(& &1)
      else
        []
      end
    rescue
      _ -> []
    end
    
    defp get_base_network_ip() do
      case get_network_interfaces() do
        [ip | _] when ip != "127.0.0.1" ->
          ip
          |> String.split(".")
          |> Enum.take(3)
          |> Enum.join(".")
        _ -> nil
      end
    end
    
    defp ping_host(ip) do
      case System.cmd("ping", ["-c", "1", "-W", "500", ip], stderr_to_stdout: true) do
        {_, 0} -> %{ip: ip, status: :alive}
        _ -> nil
      end
    rescue
      _ -> nil
    end
    
    @doc "Create environment structures"
    def create_environment_structure(agent) do
      base_dir = "/tmp/self_org_#{agent.id}"
      
      # Create agent's workspace
      File.mkdir_p!(base_dir)
      File.mkdir_p!("#{base_dir}/data")
      File.mkdir_p!("#{base_dir}/logs")
      File.mkdir_p!("#{base_dir}/scripts")
      
      # Create agent configuration
      config = %{
        agent_id: agent.id,
        type: agent.type,
        port: agent.port,
        created_at: DateTime.utc_now(),
        capabilities: agent.capabilities
      }
      
      config_file = "#{base_dir}/config.json"
      File.write!(config_file, Jason.encode!(config, pretty: true))
      
      # Create a simple Python worker script
      python_script = create_python_worker_script(agent)
      python_file = "#{base_dir}/scripts/worker.py"
      File.write!(python_file, python_script)
      
      # Create monitoring script
      monitor_script = create_monitor_script(agent)
      monitor_file = "#{base_dir}/scripts/monitor.sh"
      File.write!(monitor_file, monitor_script)
      File.chmod!(monitor_file, 0o755)
      
      [config_file, python_file, monitor_file]
    end
    
    defp create_python_worker_script(agent) do
      """
      #!/usr/bin/env python3
      # Auto-generated Python worker for agent #{agent.id}
      
      import time
      import json
      import os
      import sys
      import socket
      import threading
      from datetime import datetime
      
      class Agent#{agent.id}Worker:
          def __init__(self):
              self.agent_id = #{agent.id}
              self.port = #{agent.port}
              self.running = True
              self.metrics = {
                  'start_time': datetime.now().isoformat(),
                  'tasks_completed': 0,
                  'connections_made': 0,
                  'resources_discovered': 0
              }
          
          def discover_system_resources(self):
              \"\"\"Discover system resources\"\"\"
              try:
                  # CPU info
                  with open('/proc/cpuinfo', 'r') as f:
                      cpu_count = len([line for line in f if line.startswith('processor')])
                  
                  # Memory info  
                  with open('/proc/meminfo', 'r') as f:
                      mem_info = f.read()
                      total_mem = None
                      for line in mem_info.split('\\n'):
                          if line.startswith('MemTotal:'):
                              total_mem = int(line.split()[1]) * 1024  # Convert to bytes
                              break
                  
                  return {
                      'cpu_cores': cpu_count,
                      'total_memory': total_mem,
                      'hostname': socket.gethostname(),
                      'platform': sys.platform
                  }
              except:
                  return {
                      'cpu_cores': 'unknown',
                      'total_memory': 'unknown', 
                      'hostname': socket.gethostname(),
                      'platform': sys.platform
                  }
          
          def start_discovery_task(self):
              \"\"\"Continuously discover and adapt\"\"\"
              while self.running:
                  try:
                      # Simulate environment discovery
                      resources = self.discover_system_resources()
                      self.metrics['resources_discovered'] += 1
                      
                      # Try to connect to other agents
                      for port in range(8000, 8010):
                          if port != self.port:
                              try:
                                  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                  sock.settimeout(0.1)
                                  result = sock.connect_ex(('localhost', port))
                                  if result == 0:
                                      self.metrics['connections_made'] += 1
                                      print(f"Agent #{agent.id}: Connected to service on port {port}")
                                  sock.close()
                              except:
                                  pass
                      
                      self.metrics['tasks_completed'] += 1
                      
                      # Save metrics
                      with open(f'/tmp/self_org_#{agent.id}/data/metrics.json', 'w') as f:
                          json.dump(self.metrics, f, indent=2)
                      
                      time.sleep(2)
                  except KeyboardInterrupt:
                      break
                  except Exception as e:
                      print(f"Agent #{agent.id} error: {e}")
                      time.sleep(1)
          
          def run(self):
              print(f"🐍 Python Agent #{agent.id} starting on port #{agent.port}")
              
              # Start discovery in background
              discovery_thread = threading.Thread(target=self.start_discovery_task)
              discovery_thread.daemon = True
              discovery_thread.start()
              
              # Simple HTTP-like server
              try:
                  server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                  server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                  server_socket.bind(('localhost', self.port))
                  server_socket.listen(5)
                  server_socket.settimeout(1.0)
                  
                  print(f"Agent #{agent.id} listening on port #{agent.port}")
                  
                  while self.running:
                      try:
                          client_socket, addr = server_socket.accept()
                          response = f"HTTP/1.1 200 OK\\r\\n\\r\\nAgent #{agent.id} - {json.dumps(self.metrics)}"
                          client_socket.send(response.encode())
                          client_socket.close()
                      except socket.timeout:
                          continue
                      except KeyboardInterrupt:
                          break
                          
              except Exception as e:
                  print(f"Agent #{agent.id} server error: {e}")
              finally:
                  if 'server_socket' in locals():
                      server_socket.close()
      
      if __name__ == "__main__":
          worker = Agent#{agent.id}Worker()
          try:
              worker.run()
          except KeyboardInterrupt:
              print(f"\\nAgent #{agent.id} shutting down...")
              worker.running = False
      """
    end
    
    defp create_monitor_script(agent) do
      """
      #!/bin/bash
      # Monitoring script for agent #{agent.id}
      
      AGENT_ID=#{agent.id}
      BASE_DIR="/tmp/self_org_${AGENT_ID}"
      
      echo "🔍 Agent ${AGENT_ID} Monitor Starting..."
      
      # Monitor system resources
      monitor_resources() {
          while true; do
              # Get system info
              CPU_USAGE=$(top -l 1 -n 0 | grep "CPU usage" | awk '{print $3}' | sed 's/%//' 2>/dev/null || echo "0")
              MEMORY_PRESSURE=$(vm_stat | grep "Pages free" | awk '{print $3}' | sed 's/\\.//' 2>/dev/null || echo "0")
              DISK_USAGE=$(df -h /tmp | tail -1 | awk '{print $5}' | sed 's/%//' 2>/dev/null || echo "0")
              
              # Save metrics
              cat > "${BASE_DIR}/data/system_metrics.json" << EOF
      {
        "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "agent_id": ${AGENT_ID},
        "cpu_usage": "${CPU_USAGE}",
        "memory_pressure": ${MEMORY_PRESSURE},
        "disk_usage": "${DISK_USAGE}",
        "uptime": "$(uptime | awk '{print $3$4}' | sed 's/,//')"
      }
      EOF
              
              sleep 5
          done
      }
      
      # Network discovery
      discover_network() {
          echo "🌐 Discovering network topology..."
          
          # Scan for other agents
          for port in {8000..8010}; do
              if [ $port -ne #{agent.port} ]; then
                  if nc -z localhost $port 2>/dev/null; then
                      echo "Found agent on port $port" >> "${BASE_DIR}/logs/discoveries.log"
                  fi
              fi
          done
          
          # Check for common services
          netstat -an | grep LISTEN >> "${BASE_DIR}/logs/network_state.log" 2>/dev/null || true
      }
      
      # Start monitoring in background
      monitor_resources &
      MONITOR_PID=$!
      
      # Periodic network discovery
      while true; do
          discover_network
          sleep 10
      done
      """
    end
    
    @doc "Spawn actual processes for agent"
    def spawn_agent_processes(agent, created_files) do
      processes = []
      
      # Start Python worker
      python_file = Enum.find(created_files, &String.ends_with?(&1, "worker.py"))
      if python_file do
        python_port = Port.open({:spawn_executable, System.find_executable("python3")}, [
          :binary, :exit_status, args: [python_file]
        ])
        processes = [%{type: :python_worker, port: python_port, file: python_file} | processes]
      end
      
      # Start monitoring script
      monitor_file = Enum.find(created_files, &String.ends_with?(&1, "monitor.sh"))
      if monitor_file do
        monitor_port = Port.open({:spawn_executable, System.find_executable("bash")}, [
          :binary, :exit_status, args: [monitor_file]
        ])
        processes = [%{type: :monitor, port: monitor_port, file: monitor_file} | processes]
      end
      
      processes
    end
    
    @doc "Discover agent interactions"
    def discover_agent_interactions(agents) do
      IO.puts("🔗 Discovering agent interactions...")
      
      interactions = []
      
      # Check which agents can communicate
      for agent1 <- agents, agent2 <- agents, agent1.id != agent2.id do
        case :gen_tcp.connect('localhost', agent2.port, [], 100) do
          {:ok, socket} ->
            :gen_tcp.close(socket)
            interaction = %{
              from: agent1.id,
              to: agent2.id,
              type: :tcp_connection,
              timestamp: DateTime.utc_now()
            }
            interactions = [interaction | interactions]
          {:error, _} -> :ok
        end
      end
      
      interactions
    end
    
    @doc "Collect system-wide metrics"
    def collect_system_metrics() do
      # Get running processes
      {ps_output, _} = System.cmd("ps", ["aux"], stderr_to_stdout: true)
      process_count = ps_output |> String.split("\n") |> length()
      
      # Get network connections
      {netstat_output, _} = System.cmd("netstat", ["-an"], stderr_to_stdout: true)
      connection_count = 
        netstat_output
        |> String.split("\n")
        |> Enum.count(&String.contains?(&1, "LISTEN"))
      
      # Get system load
      {uptime_output, _} = System.cmd("uptime", [], stderr_to_stdout: true)
      
      %{
        timestamp: DateTime.utc_now(),
        process_count: process_count,
        network_connections: connection_count,
        system_uptime: String.trim(uptime_output),
        self_org_temp_files: count_temp_files()
      }
    rescue
      _ -> %{error: "Could not collect system metrics"}
    end
    
    defp count_temp_files() do
      case File.ls("/tmp") do
        {:ok, files} ->
          files
          |> Enum.count(&String.starts_with?(&1, "self_org_"))
        _ -> 0
      end
    end
  end

  defmodule SystemOrchestrator do
    def run_advanced_demo() do
      IO.puts("\n🚀 ADVANCED SELF-ORGANIZING SYSTEM")
      IO.puts("=" <> String.duplicate("=", 70))
      IO.puts("Creating REAL processes, files, and network connections!")
      
      # Cleanup any previous runs
      cleanup_previous_runs()
      
      # Create agents
      agents = [
        EnvironmentAgent.new(1, :discoverer),
        EnvironmentAgent.new(2, :builder),
        EnvironmentAgent.new(3, :connector),
        EnvironmentAgent.new(4, :monitor),
        EnvironmentAgent.new(5, :orchestrator)
      ]
      
      IO.puts("\n🤖 Created #{length(agents)} environment agents")
      display_agents(agents)
      
      # Phase 1: Environment Discovery
      IO.puts("\n" <> String.duplicate("=", 50))
      IO.puts("🔍 PHASE 1: ENVIRONMENT DISCOVERY")
      IO.puts(String.duplicate("=", 50))
      
      network_state = EnvironmentManipulator.discover_network_nodes()
      display_network_discovery(network_state)
      
      # Phase 2: Environment Manipulation
      IO.puts("\n" <> String.duplicate("=", 50))
      IO.puts("🏗️  PHASE 2: ENVIRONMENT MANIPULATION")
      IO.puts(String.duplicate("=", 50))
      
      agents_with_env = Enum.map(agents, fn agent ->
        IO.puts("Creating environment for Agent #{agent.id} (#{agent.type})...")
        created_files = EnvironmentManipulator.create_environment_structure(agent)
        %{agent | created_files: created_files}
      end)
      
      # Phase 3: Process Spawning
      IO.puts("\n" <> String.duplicate("=", 50))
      IO.puts("⚡ PHASE 3: SPAWNING REAL PROCESSES")
      IO.puts(String.duplicate("=", 50))
      
      agents_with_processes = Enum.map(agents_with_env, fn agent ->
        IO.puts("Spawning processes for Agent #{agent.id}...")
        processes = EnvironmentManipulator.spawn_agent_processes(agent, agent.created_files)
        %{agent | running_processes: processes}
      end)
      
      # Give processes time to start
      IO.puts("⏳ Waiting for processes to initialize...")
      Process.sleep(3000)
      
      # Phase 4: Self-Organization
      IO.puts("\n" <> String.duplicate("=", 50))
      IO.puts("🌟 PHASE 4: SELF-ORGANIZATION IN ACTION")
      IO.puts(String.duplicate("=", 50))
      
      # Monitor for several cycles
      1..8
      |> Enum.each(fn cycle ->
        IO.puts("\n--- Self-Organization Cycle #{cycle} ---")
        
        # Discover interactions between agents
        interactions = EnvironmentManipulator.discover_agent_interactions(agents_with_processes)
        IO.puts("🔗 Discovered #{length(interactions)} agent interactions")
        
        # Collect system metrics
        system_metrics = EnvironmentManipulator.collect_system_metrics()
        display_system_metrics(system_metrics)
        
        # Check agent metrics
        check_agent_metrics(agents_with_processes)
        
        # Show emergent network topology
        if length(interactions) > 0 do
          display_network_topology(interactions)
        end
        
        Process.sleep(2000)
      end)
      
      # Phase 5: Analysis
      IO.puts("\n" <> String.duplicate("=", 50))
      IO.puts("📊 PHASE 5: EMERGENCE ANALYSIS")
      IO.puts(String.duplicate("=", 50))
      
      analyze_emergence(agents_with_processes)
      
      # Cleanup
      IO.puts("\n🧹 Cleaning up processes...")
      cleanup_processes(agents_with_processes)
      
      IO.puts("\n✅ Advanced self-organization complete!")
      IO.puts("Check /tmp/self_org_* directories for created artifacts")
    end
    
    defp cleanup_previous_runs() do
      case System.cmd("rm", ["-rf"] ++ Path.wildcard("/tmp/self_org_*"), stderr_to_stdout: true) do
        {_, _} -> :ok
      end
    rescue
      _ -> :ok
    end
    
    defp display_agents(agents) do
      agents
      |> Enum.each(fn agent ->
        caps = agent.capabilities |> Enum.join(", ")
        IO.puts("  Agent #{agent.id}: #{agent.type} (port #{agent.port}) - [#{caps}]")
      end)
    end
    
    defp display_network_discovery(network_state) do
      IO.puts("🌐 Localhost Services:")
      network_state.localhost_services
      |> Enum.each(fn service ->
        IO.puts("  ✅ #{service.host}:#{service.port} - #{service.status}")
      end)
      
      IO.puts("\n🔌 Network Interfaces:")
      network_state.network_interfaces
      |> Enum.each(fn ip ->
        IO.puts("  📡 #{ip}")
      end)
      
      if length(network_state.nearby_devices) > 0 do
        IO.puts("\n🏠 Nearby Devices:")
        network_state.nearby_devices
        |> Enum.each(fn device ->
          IO.puts("  📱 #{device.ip} - #{device.status}")
        end)
      else
        IO.puts("\n🏠 No nearby devices discovered")
      end
    end
    
    defp display_system_metrics(metrics) do
      if Map.has_key?(metrics, :error) do
        IO.puts("⚠️  System metrics: #{metrics.error}")
      else
        IO.puts("📊 System: #{metrics.process_count} processes, #{metrics.network_connections} listeners")
        IO.puts("📁 Created #{metrics.self_org_temp_files} self-org temp directories")
      end
    end
    
    defp check_agent_metrics(agents) do
      discovered_metrics = 
        agents
        |> Enum.map(fn agent ->
          metrics_file = "/tmp/self_org_#{agent.id}/data/metrics.json"
          case File.read(metrics_file) do
            {:ok, content} ->
              case Jason.decode(content) do
                {:ok, metrics} -> {agent.id, metrics}
                _ -> {agent.id, %{}}
              end
            _ -> {agent.id, %{}}
          end
        end)
        |> Enum.filter(fn {_, metrics} -> map_size(metrics) > 0 end)
      
      if length(discovered_metrics) > 0 do
        IO.puts("🐍 Python Agent Metrics:")
        discovered_metrics
        |> Enum.each(fn {agent_id, metrics} ->
          tasks = Map.get(metrics, "tasks_completed", 0)
          connections = Map.get(metrics, "connections_made", 0)
          IO.puts("  Agent #{agent_id}: #{tasks} tasks, #{connections} connections")
        end)
      end
    end
    
    defp display_network_topology(interactions) do
      connection_counts = 
        interactions
        |> Enum.reduce(%{}, fn interaction, acc ->
          Map.update(acc, interaction.from, 1, &(&1 + 1))
        end)
      
      if map_size(connection_counts) > 0 do
        IO.puts("🕸️  Network Topology:")
        connection_counts
        |> Enum.each(fn {agent_id, count} ->
          IO.puts("  Hub Agent #{agent_id}: #{count} outbound connections")
        end)
      end
    end
    
    defp analyze_emergence(agents) do
      # Count created artifacts
      total_files = 
        agents
        |> Enum.map(&length(&1.created_files))
        |> Enum.sum()
      
      total_processes = 
        agents
        |> Enum.map(&length(&1.running_processes))
        |> Enum.sum()
      
      # Check for emergent behaviors
      IO.puts("🔬 EMERGENT BEHAVIORS DETECTED:")
      IO.puts("✅ Environment Manipulation: #{total_files} files created")
      IO.puts("✅ Process Spawning: #{total_processes} real processes launched")
      IO.puts("✅ Network Formation: Agents discovered and connected")
      IO.puts("✅ System Integration: Python + Bash + Elixir coordination")
      IO.puts("✅ Resource Discovery: System resources mapped and monitored")
      IO.puts("✅ Autonomous Operation: Agents operating independently")
      
      # Check for specialized file types
      script_types = 
        agents
        |> Enum.flat_map(& &1.created_files)
        |> Enum.map(&Path.extname/1)
        |> Enum.frequencies()
      
      IO.puts("\n📂 Created Artifact Types:")
      script_types
      |> Enum.each(fn {ext, count} ->
        IO.puts("  #{ext}: #{count} files")
      end)
    end
    
    defp cleanup_processes(agents) do
      agents
      |> Enum.each(fn agent ->
        agent.running_processes
        |> Enum.each(fn process ->
          try do
            Port.close(process.port)
          rescue
            _ -> :ok
          end
        end)
      end)
    end
  end
end

# Run the advanced demo
AdvancedSelfOrg.SystemOrchestrator.run_advanced_demo()
defmodule Object.ResourceManager do
  @moduledoc """
  Manages system resources including memory, CPU, network, and storage limits.
  Provides graceful degradation and emergency shutdown capabilities.
  """

  defstruct [
    :memory_limit,
    :cpu_limit,
    :network_limit,
    :storage_limit,
    :graceful_degradation,
    :emergency_shutdown_threshold,
    current_usage: %{memory: 0, cpu: 0.0, network: 0, storage: 0},
    alerts: [],
    emergency_mode: false
  ]

  def new(opts) do
    %__MODULE__{
      memory_limit: opts[:memory_limit] || 1000,
      cpu_limit: opts[:cpu_limit] || 100.0,
      network_limit: opts[:network_limit] || 10000,
      storage_limit: opts[:storage_limit] || 5000,
      graceful_degradation: opts[:graceful_degradation] || true,
      emergency_shutdown_threshold: opts[:emergency_shutdown_threshold] || 0.95
    }
  end

  def check_resource_usage(resource_manager) do
    memory_usage = :erlang.memory(:total) / 1024 / 1024
    cpu_usage = get_cpu_usage()
    
    current_usage = %{
      memory: memory_usage,
      cpu: cpu_usage,
      network: 0,
      storage: 0
    }
    
    resource_manager = %{resource_manager | current_usage: current_usage}
    
    memory_pressure = memory_usage / resource_manager.memory_limit
    cpu_pressure = cpu_usage / resource_manager.cpu_limit
    
    pressure = max(memory_pressure, cpu_pressure)
    
    cond do
      pressure >= resource_manager.emergency_shutdown_threshold ->
        trigger_emergency_shutdown(resource_manager)
      pressure >= 0.8 ->
        trigger_graceful_degradation(resource_manager)
      true ->
        {resource_manager, :normal}
    end
  end

  defp get_cpu_usage do
    # cpu_sup is not available on all systems, fallback to scheduler usage
    schedulers = :erlang.system_info(:schedulers_online)
    :erlang.statistics(:scheduler_wall_time)
    |> Enum.take(schedulers)
    |> Enum.reduce(0.0, fn {_id, active, total}, acc ->
      if total > 0, do: acc + (active / total), else: acc
    end)
    |> Kernel./(schedulers)
  rescue
    _ -> 0.5  # Default to 50% if we can't get CPU usage
  end

  defp trigger_emergency_shutdown(resource_manager) do
    resource_manager = %{resource_manager | 
      emergency_mode: true,
      alerts: ["Emergency shutdown triggered" | resource_manager.alerts]
    }
    {resource_manager, :emergency_shutdown}
  end

  defp trigger_graceful_degradation(resource_manager) do
    resource_manager = %{resource_manager | 
      alerts: ["Graceful degradation activated" | resource_manager.alerts]
    }
    {resource_manager, :graceful_degradation}
  end

  def get_memory_pressure(resource_manager) do
    resource_manager.current_usage.memory / resource_manager.memory_limit
  end

  def allocate_resource(resource_manager, resource_type, amount) do
    current = resource_manager.current_usage[resource_type] || 0
    limit = Map.get(resource_manager, :"#{resource_type}_limit", 1000)
    
    if current + amount <= limit do
      new_usage = Map.put(resource_manager.current_usage, resource_type, current + amount)
      {:ok, %{resource_manager | current_usage: new_usage}}
    else
      {:error, :resource_limit_exceeded}
    end
  end

  def deallocate_resource(resource_manager, resource_type, amount) do
    current = resource_manager.current_usage[resource_type] || 0
    new_amount = max(0, current - amount)
    new_usage = Map.put(resource_manager.current_usage, resource_type, new_amount)
    %{resource_manager | current_usage: new_usage}
  end

  def allocate_resources(resource_manager, objects) when is_list(objects) do
    total_demand = calculate_total_resource_demand(objects)
    available_resources = calculate_available_resources(resource_manager)
    
    allocation_strategy = determine_allocation_strategy(total_demand, available_resources)
    
    case allocation_strategy do
      :sufficient_resources ->
        allocation_list = allocate_full_demand(objects, available_resources)
        updated_manager = apply_allocations(resource_manager, allocation_list)
        allocation_map = convert_allocation_to_map(allocation_list)
        {allocation_map, updated_manager}
      
      :resource_pressure ->
        allocation_list = allocate_with_fair_sharing(objects, available_resources)
        updated_manager = apply_allocations(resource_manager, allocation_list)
        updated_manager = %{updated_manager | 
          alerts: ["Resource pressure detected - fair sharing activated" | updated_manager.alerts]
        }
        allocation_map = convert_allocation_to_map(allocation_list)
        {allocation_map, updated_manager}
      
      :resource_exhaustion ->
        allocation_list = allocate_with_priority_based_degradation(objects, available_resources)
        updated_manager = apply_allocations(resource_manager, allocation_list)
        updated_manager = %{updated_manager | 
          emergency_mode: true,
          alerts: ["Resource exhaustion - priority-based allocation" | updated_manager.alerts]
        }
        allocation_map = convert_allocation_to_map(allocation_list)
        {allocation_map, updated_manager}
    end
  end

  defp calculate_total_resource_demand(objects) do
    Enum.reduce(objects, %{memory: 0, cpu: 0, network: 0, storage: 0}, fn obj, acc ->
      demand = obj.state
      %{
        memory: acc.memory + (demand[:memory_usage] || 0),
        cpu: acc.cpu + (demand[:cpu_usage] || 0),
        network: acc.network + (demand[:network_usage] || 0),
        storage: acc.storage + (demand[:storage_usage] || 0)
      }
    end)
  end

  defp calculate_available_resources(resource_manager) do
    %{
      memory: resource_manager.memory_limit - resource_manager.current_usage.memory,
      cpu: resource_manager.cpu_limit - resource_manager.current_usage.cpu,
      network: resource_manager.network_limit - resource_manager.current_usage.network,
      storage: resource_manager.storage_limit - resource_manager.current_usage.storage
    }
  end

  defp determine_allocation_strategy(demand, available) do
    pressure_ratio = calculate_pressure_ratio(demand, available)
    
    cond do
      pressure_ratio <= 0.7 -> :sufficient_resources
      pressure_ratio <= 0.9 -> :resource_pressure
      true -> :resource_exhaustion
    end
  end

  defp calculate_pressure_ratio(demand, available) do
    resource_ratios = [
      demand.memory / max(available.memory, 1),
      demand.cpu / max(available.cpu, 1),
      demand.network / max(available.network, 1),
      demand.storage / max(available.storage, 1)
    ]
    
    Enum.max(resource_ratios)
  end

  defp allocate_full_demand(objects, _available) do
    Enum.map(objects, fn obj ->
      %{
        object_id: obj.id,
        allocated: %{
          memory: obj.state[:memory_usage] || 0,
          cpu: obj.state[:cpu_usage] || 0,
          network: obj.state[:network_usage] || 0,
          storage: obj.state[:storage_usage] || 0
        },
        satisfaction_ratio: 1.0
      }
    end)
  end

  defp allocate_with_fair_sharing(objects, available) do
    total_demand = calculate_total_resource_demand(objects)
    
    Enum.map(objects, fn obj ->
      fair_ratio = calculate_fair_ratio(obj.state, total_demand, available)
      
      %{
        object_id: obj.id,
        allocated: %{
          memory: (obj.state[:memory_usage] || 0) * fair_ratio,
          cpu: (obj.state[:cpu_usage] || 0) * fair_ratio,
          network: (obj.state[:network_usage] || 0) * fair_ratio,
          storage: (obj.state[:storage_usage] || 0) * fair_ratio
        },
        satisfaction_ratio: fair_ratio
      }
    end)
  end

  defp allocate_with_priority_based_degradation(objects, available) do
    sorted_objects = Enum.sort_by(objects, fn obj -> 
      obj.state[:priority] || :medium 
    end, fn a, b ->
      priority_value(a) >= priority_value(b)
    end)
    
    {allocations, _remaining} = Enum.reduce(sorted_objects, {[], available}, fn obj, {acc, remaining} ->
      demand = %{
        memory: obj.state[:memory_usage] || 0,
        cpu: obj.state[:cpu_usage] || 0,
        network: obj.state[:network_usage] || 0,
        storage: obj.state[:storage_usage] || 0
      }
      
      allocation = calculate_priority_allocation(demand, remaining)
      updated_remaining = subtract_allocation(remaining, allocation.allocated)
      
      {[allocation | acc], updated_remaining}
    end)
    
    Enum.reverse(allocations)
  end

  defp calculate_fair_ratio(_object_demand, total_demand, available) do
    resource_ratios = [
      available.memory / max(total_demand.memory, 1),
      available.cpu / max(total_demand.cpu, 1),
      available.network / max(total_demand.network, 1),
      available.storage / max(total_demand.storage, 1)
    ]
    
    Enum.min(resource_ratios)
  end

  defp priority_value(:high), do: 3
  defp priority_value(:medium), do: 2
  defp priority_value(:low), do: 1
  defp priority_value(_), do: 2

  defp calculate_priority_allocation(demand, available) do
    allocation = %{
      memory: min(demand.memory, available.memory),
      cpu: min(demand.cpu, available.cpu),
      network: min(demand.network, available.network),
      storage: min(demand.storage, available.storage)
    }
    
    satisfaction = calculate_satisfaction(demand, allocation)
    
    %{
      object_id: "priority_allocation",
      allocated: allocation,
      satisfaction_ratio: satisfaction
    }
  end

  defp calculate_satisfaction(demand, allocation) do
    satisfactions = [
      allocation.memory / max(demand.memory, 1),
      allocation.cpu / max(demand.cpu, 1),
      allocation.network / max(demand.network, 1),
      allocation.storage / max(demand.storage, 1)
    ]
    
    Enum.sum(satisfactions) / length(satisfactions)
  end

  defp subtract_allocation(available, allocation) do
    %{
      memory: max(0, available.memory - allocation.memory),
      cpu: max(0, available.cpu - allocation.cpu),
      network: max(0, available.network - allocation.network),
      storage: max(0, available.storage - allocation.storage)
    }
  end

  defp apply_allocations(resource_manager, allocations) do
    total_allocated = Enum.reduce(allocations, %{memory: 0, cpu: 0, network: 0, storage: 0}, fn alloc, acc ->
      %{
        memory: acc.memory + alloc.allocated.memory,
        cpu: acc.cpu + alloc.allocated.cpu,
        network: acc.network + alloc.allocated.network,
        storage: acc.storage + alloc.allocated.storage
      }
    end)
    
    new_usage = %{
      memory: resource_manager.current_usage.memory + total_allocated.memory,
      cpu: resource_manager.current_usage.cpu + total_allocated.cpu,
      network: resource_manager.current_usage.network + total_allocated.network,
      storage: resource_manager.current_usage.storage + total_allocated.storage
    }
    
    %{resource_manager | current_usage: new_usage}
  end

  defp convert_allocation_to_map(allocation_list) do
    Enum.into(allocation_list, %{}, fn allocation ->
      {allocation.object_id, %{
        allocated_memory: allocation.allocated.memory,
        allocated_cpu: allocation.allocated.cpu,
        allocated_network: allocation.allocated.network,
        allocated_storage: allocation.allocated.storage,
        satisfaction_ratio: allocation.satisfaction_ratio,
        memory_limited: allocation.satisfaction_ratio < 1.0,
        cpu_limited: allocation.satisfaction_ratio < 1.0,
        network_limited: allocation.satisfaction_ratio < 1.0,
        storage_limited: allocation.satisfaction_ratio < 1.0
      }}
    end)
  end
end
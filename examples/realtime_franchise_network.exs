#!/usr/bin/env elixir

# Realtime Franchise Network Management
# Living multi-location business network with real-time operations and coordination

defmodule RealtimeFranchiseNetwork do
  @moduledoc """
  Realtime franchise network management system.
  
  Features:
  - Multi-location real-time coordination
  - Dynamic resource allocation across locations
  - Live performance monitoring and optimization
  - Autonomous regional management
  - Supply chain coordination
  - Interactive corporate control
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :headquarters,
    :regional_managers,
    :store_locations,
    :supply_chain,
    :knowledge_system,
    :network_metrics,
    :active_operations,
    :coordination_events,
    :running,
    :start_time,
    :current_time,
    :performance_dashboard
  ]
  
  @tick_interval 4000  # 4 seconds per business minute
  @business_hours {8, 22}  # 8 AM to 10 PM
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def stop do
    GenServer.call(__MODULE__, :stop)
  end
  
  def status do
    GenServer.call(__MODULE__, :status)
  end
  
  def get_network_metrics do
    GenServer.call(__MODULE__, :get_network_metrics)
  end
  
  def get_location(location_id) do
    GenServer.call(__MODULE__, {:get_location, location_id})
  end
  
  def get_regional_manager(region) do
    GenServer.call(__MODULE__, {:get_regional_manager, region})
  end
  
  def trigger_network_event(event_type, params \\ %{}) do
    GenServer.cast(__MODULE__, {:trigger_event, event_type, params})
  end
  
  def allocate_resources(from_location, to_location, resources) do
    GenServer.cast(__MODULE__, {:allocate_resources, from_location, to_location, resources})
  end
  
  def launch_promotion(promotion_data) do
    GenServer.cast(__MODULE__, {:launch_promotion, promotion_data})
  end
  
  def emergency_coordination(emergency_type, affected_locations) do
    GenServer.cast(__MODULE__, {:emergency_coordination, emergency_type, affected_locations})
  end
  
  @impl true
  def init(_opts) do
    IO.puts("🏢 Initializing Realtime Franchise Network...")
    
    # Create network components
    {headquarters, regional_managers, store_locations, supply_chain, knowledge_system} = create_franchise_network()
    
    state = %__MODULE__{
      headquarters: headquarters,
      regional_managers: regional_managers |> Enum.map(&{&1.id, &1}) |> Map.new(),
      store_locations: store_locations |> Enum.map(&{&1.id, &1}) |> Map.new(),
      supply_chain: supply_chain,
      knowledge_system: knowledge_system,
      network_metrics: initialize_network_metrics(),
      active_operations: %{},
      coordination_events: [],
      running: true,
      start_time: DateTime.utc_now(),
      current_time: DateTime.utc_now(),
      performance_dashboard: initialize_dashboard()
    }
    
    # Start network operations
    schedule_tick()
    
    IO.puts("✅ Franchise Network activated!")
    IO.puts("   • Headquarters: Central coordination")
    IO.puts("   • Regions: #{map_size(state.regional_managers)}")
    IO.puts("   • Locations: #{map_size(state.store_locations)}")
    IO.puts("   • Network operations: Active")
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:stop, _from, state) do
    IO.puts("🛑 Shutting down franchise network...")
    {:stop, :normal, :ok, %{state | running: false}}
  end
  
  @impl true
  def handle_call(:status, _from, state) do
    {open_hour, close_hour} = @business_hours
    current_hour = DateTime.to_time(state.current_time).hour
    is_business_hours = current_hour >= open_hour and current_hour < close_hour
    
    status = %{
      running: state.running,
      uptime: DateTime.diff(DateTime.utc_now(), state.start_time),
      current_time: state.current_time,
      business_hours: is_business_hours,
      total_locations: map_size(state.store_locations),
      total_regions: map_size(state.regional_managers),
      active_operations: map_size(state.active_operations),
      network_health: calculate_network_health(state),
      performance: state.performance_dashboard
    }
    {:reply, status, state}
  end
  
  @impl true
  def handle_call(:get_network_metrics, _from, state) do
    {:reply, state.network_metrics, state}
  end
  
  @impl true
  def handle_call({:get_location, location_id}, _from, state) do
    location = Map.get(state.store_locations, location_id)
    {:reply, location, state}
  end
  
  @impl true
  def handle_call({:get_regional_manager, region}, _from, state) do
    manager = Enum.find(state.regional_managers, fn {_, rm} ->
      get_in(rm.state, [:region]) == region
    end)
    {:reply, manager, state}
  end
  
  @impl true
  def handle_cast({:trigger_event, event_type, params}, state) do
    {new_state, result} = handle_network_event(state, event_type, params)
    log_coordination_event(new_state, event_type, result)
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:allocate_resources, from_location, to_location, resources}, state) do
    new_state = execute_resource_allocation(state, from_location, to_location, resources)
    IO.puts("🔄 Resource allocation: #{from_location} → #{to_location}")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:launch_promotion, promotion_data}, state) do
    new_state = launch_network_promotion(state, promotion_data)
    IO.puts("🎯 Network promotion launched: #{promotion_data.name}")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:emergency_coordination, emergency_type, affected_locations}, state) do
    new_state = coordinate_emergency_response(state, emergency_type, affected_locations)
    IO.puts("🚨 Emergency coordination activated: #{emergency_type}")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info(:tick, %{running: false} = state) do
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:tick, state) do
    tick_start = System.monotonic_time(:millisecond)
    
    # Advance business time (1 tick = 1 hour)
    new_time = DateTime.add(state.current_time, 3600, :second)
    
    # Main network coordination cycle
    new_state = %{state | current_time: new_time}
                |> process_headquarters_coordination()
                |> process_regional_management()
                |> process_store_operations()
                |> process_supply_chain_coordination()
                |> process_knowledge_sharing()
                |> update_network_metrics()
                |> process_autonomous_optimizations()
                |> cleanup_old_operations()
    
    # Performance tracking
    tick_duration = System.monotonic_time(:millisecond) - tick_start
    updated_dashboard = update_performance_dashboard(new_state, tick_duration)
    
    final_state = %{new_state | performance_dashboard: updated_dashboard}
    
    # Log business day events
    current_hour = DateTime.to_time(final_state.current_time).hour
    if current_hour == 8 do
      log_business_day_start(final_state)
    elseif current_hour == 22 do
      log_business_day_end(final_state)
    end
    
    schedule_tick()
    {:noreply, final_state}
  end
  
  # Private functions
  
  defp create_franchise_network do
    # Corporate Headquarters
    headquarters = Object.new(
      id: "hq_central_001",
      subtype: :coordinator_object,
      state: %{
        role: :corporate_headquarters,
        network_oversight: %{
          total_locations: 8,
          regions: 3,
          daily_revenue_target: 100000,
          customer_satisfaction_target: 4.5
        },
        strategic_initiatives: ["digital_transformation", "sustainability", "expansion"],
        decision_queue: [],
        coordination_efficiency: 0.9,
        brand_consistency_score: 0.92
      },
      methods: [:coordinate_network, :make_strategic_decisions, :allocate_budgets, :monitor_performance],
      goal: fn state ->
        efficiency = state.coordination_efficiency
        brand_consistency = state.brand_consistency_score
        queue_management = max(0.0, 1.0 - length(state.decision_queue) / 20.0)
        (efficiency + brand_consistency + queue_management) / 3
      end
    )
    
    # Regional Managers
    regional_managers = create_regional_managers()
    
    # Store Locations
    store_locations = create_store_locations()
    
    # Supply Chain Coordinator
    supply_chain = Object.new(
      id: "supply_chain_coordinator",
      subtype: :sensor_object,
      state: %{
        role: :supply_chain_coordinator,
        inventory_network: %{},
        supplier_performance: %{
          "Global Electronics" => %{reliability: 0.92, efficiency: 0.85},
          "Fashion Forward" => %{reliability: 0.88, efficiency: 0.80},
          "Home Essentials" => %{reliability: 0.95, efficiency: 0.90}
        },
        logistics_optimization: 0.87,
        cost_efficiency: 0.82,
        delivery_performance: 0.89
      },
      methods: [:optimize_supply_chain, :coordinate_deliveries, :manage_inventory, :negotiate_suppliers],
      goal: fn state ->
        logistics = state.logistics_optimization
        cost = state.cost_efficiency
        delivery = state.delivery_performance
        (logistics + cost + delivery) / 3
      end
    )
    
    # Knowledge Management System
    knowledge_system = Object.new(
      id: "knowledge_management",
      subtype: :ai_agent,
      state: %{
        role: :knowledge_coordinator,
        best_practices: %{},
        training_programs: ["customer_service", "sales_excellence", "operations"],
        innovation_pipeline: [],
        knowledge_sharing_effectiveness: 0.85,
        learning_acceleration: 0.78
      },
      methods: [:capture_knowledge, :share_best_practices, :develop_training, :foster_innovation],
      goal: fn state ->
        sharing = state.knowledge_sharing_effectiveness
        learning = state.learning_acceleration
        innovation = min(1.0, length(state.innovation_pipeline) / 10.0)
        (sharing + learning + innovation) / 3
      end
    )
    
    {headquarters, regional_managers, store_locations, supply_chain, knowledge_system}
  end
  
  defp create_regional_managers do
    [
      create_regional_manager("region_north", :north, ["store_north_metro", "store_north_suburban", "store_north_mall"]),
      create_regional_manager("region_south", :south, ["store_south_downtown", "store_south_plaza", "store_south_outlet"]),
      create_regional_manager("region_west", :west, ["store_west_coast", "store_west_valley"])
    ]
  end
  
  defp create_regional_manager(id, region, managed_stores) do
    Object.new(
      id: id,
      subtype: :coordinator_object,
      state: %{
        role: :regional_manager,
        region: region,
        managed_stores: managed_stores,
        regional_performance: %{
          revenue: 0,
          customer_satisfaction: 4.2,
          operational_efficiency: 0.85,
          market_share: 0.65
        },
        coordination_tasks: [],
        leadership_style: determine_leadership_style(region),
        performance_targets: %{
          daily_revenue: 30000,
          satisfaction: 4.4,
          efficiency: 0.90
        }
      },
      methods: [:coordinate_region, :support_stores, :analyze_performance, :implement_strategies],
      goal: fn state ->
        perf = state.regional_performance
        revenue_score = min(1.0, perf.revenue / state.performance_targets.daily_revenue)
        satisfaction_score = perf.customer_satisfaction / 5.0
        efficiency_score = perf.operational_efficiency
        (revenue_score + satisfaction_score + efficiency_score) / 3
      end
    )
  end
  
  defp determine_leadership_style(region) do
    case region do
      :north -> :performance_driven
      :south -> :collaborative_supportive
      :west -> :innovation_focused
    end
  end
  
  defp create_store_locations do
    [
      # North Region
      create_store_location("store_north_metro", :flagship, :north, %{
        location_type: :urban_metro,
        daily_traffic: 450,
        specializations: ["electronics", "business"]
      }),
      create_store_location("store_north_suburban", :standard, :north, %{
        location_type: :suburban,
        daily_traffic: 320,
        specializations: ["family", "clothing"]
      }),
      create_store_location("store_north_mall", :express, :north, %{
        location_type: :mall,
        daily_traffic: 280,
        specializations: ["convenience", "gifts"]
      }),
      
      # South Region
      create_store_location("store_south_downtown", :flagship, :south, %{
        location_type: :downtown,
        daily_traffic: 520,
        specializations: ["premium", "professional"]
      }),
      create_store_location("store_south_plaza", :standard, :south, %{
        location_type: :plaza,
        daily_traffic: 380,
        specializations: ["home", "electronics"]
      }),
      create_store_location("store_south_outlet", :outlet, :south, %{
        location_type: :outlet,
        daily_traffic: 290,
        specializations: ["discount", "bulk"]
      }),
      
      # West Region
      create_store_location("store_west_coast", :flagship, :west, %{
        location_type: :coastal,
        daily_traffic: 410,
        specializations: ["tech", "lifestyle"]
      }),
      create_store_location("store_west_valley", :standard, :west, %{
        location_type: :valley,
        daily_traffic: 340,
        specializations: ["outdoor", "family"]
      })
    ]
  end
  
  defp create_store_location(id, store_type, region, characteristics) do
    Object.new(
      id: id,
      subtype: :actuator_object,
      state: %{
        role: :franchise_store,
        store_type: store_type,
        region: region,
        characteristics: characteristics,
        daily_metrics: %{
          revenue: 0,
          customers: 0,
          satisfaction: 4.2,
          efficiency: 0.82
        },
        inventory_status: %{
          stock_level: 0.75,
          turnover_rate: 0.68,
          reorder_alerts: 0
        },
        staff_performance: %{
          productivity: 0.78,
          customer_service: 0.85,
          training_status: 0.90
        },
        operational_status: :open,
        current_promotions: []
      },
      methods: [:serve_customers, :manage_operations, :report_metrics, :coordinate_with_region],
      goal: fn state ->
        daily_target = case state.store_type do
          :flagship -> 15000
          :standard -> 10000
          :outlet -> 8000
          :express -> 6000
        end
        
        revenue_performance = min(1.0, state.daily_metrics.revenue / daily_target)
        satisfaction_performance = state.daily_metrics.satisfaction / 5.0
        efficiency_performance = state.daily_metrics.efficiency
        
        (revenue_performance + satisfaction_performance + efficiency_performance) / 3
      end
    )
  end
  
  defp initialize_network_metrics do
    %{
      total_network_revenue: 0,
      average_satisfaction: 4.2,
      operational_efficiency: 0.85,
      brand_consistency: 0.92,
      market_penetration: 0.68,
      customer_retention: 0.84,
      innovation_index: 0.72,
      sustainability_score: 0.79
    }
  end
  
  defp initialize_dashboard do
    %{
      coordination_speed: 0.0,
      decision_quality: 0.0,
      resource_utilization: 0.0,
      knowledge_flow: 0.0,
      network_resilience: 0.0
    }
  end
  
  defp process_headquarters_coordination(state) do
    hq = state.headquarters
    
    # HQ coordinates daily operations
    coordination_tasks = [
      "Monitor regional performance",
      "Allocate corporate resources",
      "Coordinate supply chain",
      "Manage brand consistency",
      "Strategic decision making"
    ]
    
    # Process decision queue
    decisions_made = min(3, length(hq.state.decision_queue))
    remaining_decisions = Enum.drop(hq.state.decision_queue, decisions_made)
    
    # Generate new strategic initiatives if needed
    new_initiatives = if :rand.uniform() < 0.1 do
      generate_strategic_initiative()
    else
      []
    end
    
    updated_hq = Object.update_state(hq, %{
      decision_queue: remaining_decisions,
      strategic_initiatives: hq.state.strategic_initiatives ++ new_initiatives,
      coordination_efficiency: min(1.0, hq.state.coordination_efficiency + 0.01)
    })
    
    %{state | headquarters: updated_hq}
  end
  
  defp generate_strategic_initiative do
    initiatives = [
      "Customer experience enhancement program",
      "Digital transformation acceleration",
      "Sustainability improvement initiative",
      "Staff development and retention program",
      "Supply chain optimization project",
      "Market expansion opportunity analysis"
    ]
    
    [Enum.random(initiatives)]
  end
  
  defp process_regional_management(state) do
    updated_managers = state.regional_managers
                      |> Enum.map(fn {id, manager} ->
                        updated_manager = process_regional_activities(manager, state)
                        {id, updated_manager}
                      end)
                      |> Map.new()
    
    %{state | regional_managers: updated_managers}
  end
  
  defp process_regional_activities(manager, state) do
    region = manager.state.region
    managed_stores = manager.state.managed_stores
    
    # Calculate regional performance from stores
    regional_performance = calculate_regional_performance(managed_stores, state.store_locations)
    
    # Generate coordination tasks
    new_tasks = generate_regional_tasks(manager, regional_performance)
    
    # Update regional performance targets
    updated_targets = adjust_performance_targets(manager.state.performance_targets, regional_performance)
    
    Object.update_state(manager, %{
      regional_performance: regional_performance,
      coordination_tasks: new_tasks,
      performance_targets: updated_targets
    })
  end
  
  defp calculate_regional_performance(store_ids, all_stores) do
    regional_stores = store_ids
                     |> Enum.map(&Map.get(all_stores, &1))
                     |> Enum.filter(& &1)
    
    if length(regional_stores) > 0 do
      total_revenue = Enum.sum(Enum.map(regional_stores, &get_in(&1.state, [:daily_metrics, :revenue])))
      avg_satisfaction = Enum.sum(Enum.map(regional_stores, &get_in(&1.state, [:daily_metrics, :satisfaction]))) / length(regional_stores)
      avg_efficiency = Enum.sum(Enum.map(regional_stores, &get_in(&1.state, [:daily_metrics, :efficiency]))) / length(regional_stores)
      
      %{
        revenue: total_revenue,
        customer_satisfaction: avg_satisfaction,
        operational_efficiency: avg_efficiency,
        market_share: 0.65 + (:rand.uniform() - 0.5) * 0.1
      }
    else
      %{revenue: 0, customer_satisfaction: 4.0, operational_efficiency: 0.8, market_share: 0.6}
    end
  end
  
  defp generate_regional_tasks(manager, performance) do
    base_tasks = ["Monitor store performance", "Coordinate supply deliveries", "Support staff training"]
    
    additional_tasks = cond do
      performance.revenue < manager.state.performance_targets.daily_revenue * 0.8 ->
        ["Implement revenue improvement strategies", "Analyze market conditions"]
      
      performance.customer_satisfaction < 4.0 ->
        ["Focus on customer service improvement", "Review customer feedback"]
      
      performance.operational_efficiency < 0.8 ->
        ["Optimize operational processes", "Review staffing levels"]
      
      true ->
        ["Explore growth opportunities", "Share best practices"]
    end
    
    base_tasks ++ additional_tasks
  end
  
  defp adjust_performance_targets(current_targets, performance) do
    # Dynamically adjust targets based on performance
    revenue_adjustment = if performance.revenue > current_targets.daily_revenue do
      current_targets.daily_revenue * 1.05
    else
      current_targets.daily_revenue * 0.98
    end
    
    %{current_targets |
      daily_revenue: revenue_adjustment,
      satisfaction: max(4.0, min(5.0, current_targets.satisfaction + (performance.customer_satisfaction - 4.2) * 0.1)),
      efficiency: max(0.7, min(1.0, current_targets.efficiency + (performance.operational_efficiency - 0.85) * 0.1))
    }
  end
  
  defp process_store_operations(state) do
    {open_hour, close_hour} = @business_hours
    current_hour = DateTime.to_time(state.current_time).hour
    is_business_hours = current_hour >= open_hour and current_hour < close_hour
    
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      updated_store = if is_business_hours do
                        simulate_business_hour_operations(store)
                      else
                        simulate_after_hours_operations(store)
                      end
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    %{state | store_locations: updated_stores}
  end
  
  defp simulate_business_hour_operations(store) do
    # Simulate customer traffic and sales
    base_traffic = get_in(store.state, [:characteristics, :daily_traffic]) || 300
    hourly_traffic = base_traffic / 14  # 14 operating hours
    
    # Add randomness
    actual_traffic = trunc(hourly_traffic * (0.8 + :rand.uniform() * 0.4))
    
    # Calculate revenue based on traffic and store type
    revenue_per_customer = case store.state.store_type do
      :flagship -> 35 + :rand.uniform(20)
      :standard -> 25 + :rand.uniform(15)
      :outlet -> 20 + :rand.uniform(10)
      :express -> 15 + :rand.uniform(10)
    end
    
    hourly_revenue = actual_traffic * revenue_per_customer
    
    # Update metrics
    current_revenue = get_in(store.state, [:daily_metrics, :revenue]) || 0
    current_customers = get_in(store.state, [:daily_metrics, :customers]) || 0
    
    # Simulate satisfaction and efficiency changes
    satisfaction_change = (:rand.uniform() - 0.5) * 0.1
    efficiency_change = (:rand.uniform() - 0.5) * 0.05
    
    new_satisfaction = max(3.0, min(5.0, store.state.daily_metrics.satisfaction + satisfaction_change))
    new_efficiency = max(0.5, min(1.0, store.state.daily_metrics.efficiency + efficiency_change))
    
    Object.update_state(store, %{
      daily_metrics: %{
        revenue: current_revenue + hourly_revenue,
        customers: current_customers + actual_traffic,
        satisfaction: new_satisfaction,
        efficiency: new_efficiency
      },
      operational_status: :open
    })
  end
  
  defp simulate_after_hours_operations(store) do
    # After hours: restocking, cleaning, maintenance
    current_hour = DateTime.to_time(DateTime.utc_now()).hour
    
    if current_hour == 2 do  # 2 AM restocking
      stock_improvement = 0.1
      new_stock_level = min(1.0, store.state.inventory_status.stock_level + stock_improvement)
      
      Object.update_state(store, %{
        inventory_status: %{store.state.inventory_status | stock_level: new_stock_level},
        operational_status: :maintenance
      })
    else
      Object.update_state(store, %{operational_status: :closed})
    end
  end
  
  defp process_supply_chain_coordination(state) do
    supply_chain = state.supply_chain
    
    # Monitor network inventory needs
    inventory_alerts = count_inventory_alerts(state.store_locations)
    
    # Optimize logistics
    logistics_improvement = if inventory_alerts > 0 do
      -0.02  # Decrease efficiency due to alerts
    else
      0.01   # Improve efficiency
    end
    
    new_logistics = max(0.6, min(1.0, supply_chain.state.logistics_optimization + logistics_improvement))
    
    # Process supplier performance
    updated_suppliers = supply_chain.state.supplier_performance
                       |> Enum.map(fn {supplier, perf} ->
                         reliability_change = (:rand.uniform() - 0.5) * 0.02
                         efficiency_change = (:rand.uniform() - 0.5) * 0.02
                         
                         updated_perf = %{
                           reliability: max(0.7, min(1.0, perf.reliability + reliability_change)),
                           efficiency: max(0.7, min(1.0, perf.efficiency + efficiency_change))
                         }
                         
                         {supplier, updated_perf}
                       end)
                       |> Map.new()
    
    updated_supply_chain = Object.update_state(supply_chain, %{
      logistics_optimization: new_logistics,
      supplier_performance: updated_suppliers,
      delivery_performance: calculate_delivery_performance(updated_suppliers)
    })
    
    %{state | supply_chain: updated_supply_chain}
  end
  
  defp count_inventory_alerts(stores) do
    stores
    |> Map.values()
    |> Enum.map(&get_in(&1.state, [:inventory_status, :reorder_alerts]))
    |> Enum.sum()
  end
  
  defp calculate_delivery_performance(suppliers) do
    avg_performance = suppliers
                     |> Map.values()
                     |> Enum.map(fn perf -> (perf.reliability + perf.efficiency) / 2 end)
                     |> Enum.sum()
                     |> then(&(&1 / map_size(suppliers)))
    
    avg_performance
  end
  
  defp process_knowledge_sharing(state) do
    knowledge_system = state.knowledge_system
    
    # Identify best practices from high-performing stores
    best_performers = state.store_locations
                     |> Map.values()
                     |> Enum.filter(&(&1.goal.(&1.state) > 0.85))
    
    new_best_practices = if length(best_performers) > 0 do
      extract_best_practices(best_performers)
    else
      []
    end
    
    # Update innovation pipeline
    new_innovations = if :rand.uniform() < 0.2 do
      generate_innovation()
    else
      []
    end
    
    updated_practices = Map.merge(knowledge_system.state.best_practices, 
                                  new_best_practices |> Enum.map(&{&1.area, &1}) |> Map.new())
    
    updated_knowledge = Object.update_state(knowledge_system, %{
      best_practices: updated_practices,
      innovation_pipeline: knowledge_system.state.innovation_pipeline ++ new_innovations
    })
    
    %{state | knowledge_system: updated_knowledge}
  end
  
  defp extract_best_practices(performers) do
    practices = [
      %{area: "customer_service", practice: "Enhanced greeting protocols", impact: 0.15},
      %{area: "operations", practice: "Efficient checkout processes", impact: 0.12},
      %{area: "inventory", practice: "Predictive restocking", impact: 0.18},
      %{area: "staff", practice: "Cross-training programs", impact: 0.14}
    ]
    
    Enum.take_random(practices, min(2, length(performers)))
  end
  
  defp generate_innovation do
    innovations = [
      "AI-powered customer service chatbot",
      "Automated inventory management system",
      "Mobile payment integration",
      "Customer behavior analytics platform",
      "Sustainable packaging initiative"
    ]
    
    [%{
      name: Enum.random(innovations),
      stage: "development",
      potential_impact: 0.1 + :rand.uniform() * 0.2,
      timeline: "6_months"
    }]
  end
  
  defp update_network_metrics(state) do
    # Calculate comprehensive network metrics
    total_revenue = state.store_locations
                   |> Map.values()
                   |> Enum.map(&get_in(&1.state, [:daily_metrics, :revenue]))
                   |> Enum.sum()
    
    avg_satisfaction = state.store_locations
                      |> Map.values()
                      |> Enum.map(&get_in(&1.state, [:daily_metrics, :satisfaction]))
                      |> Enum.sum()
                      |> then(&(&1 / max(1, map_size(state.store_locations))))
    
    avg_efficiency = state.store_locations
                    |> Map.values()
                    |> Enum.map(&get_in(&1.state, [:daily_metrics, :efficiency]))
                    |> Enum.sum()
                    |> then(&(&1 / max(1, map_size(state.store_locations))))
    
    # Brand consistency based on satisfaction variance
    satisfaction_variance = calculate_satisfaction_variance(state.store_locations)
    brand_consistency = max(0.7, 1.0 - satisfaction_variance / 2.0)
    
    updated_metrics = %{state.network_metrics |
      total_network_revenue: total_revenue,
      average_satisfaction: avg_satisfaction,
      operational_efficiency: avg_efficiency,
      brand_consistency: brand_consistency,
      innovation_index: calculate_innovation_index(state.knowledge_system)
    }
    
    %{state | network_metrics: updated_metrics}
  end
  
  defp calculate_satisfaction_variance(stores) do
    satisfactions = stores
                   |> Map.values()
                   |> Enum.map(&get_in(&1.state, [:daily_metrics, :satisfaction]))
    
    if length(satisfactions) > 1 do
      mean = Enum.sum(satisfactions) / length(satisfactions)
      variance = satisfactions
                |> Enum.map(&((&1 - mean) * (&1 - mean)))
                |> Enum.sum()
                |> then(&(&1 / length(satisfactions)))
      
      :math.sqrt(variance)
    else
      0.0
    end
  end
  
  defp calculate_innovation_index(knowledge_system) do
    pipeline_size = length(knowledge_system.state.innovation_pipeline)
    practices_count = map_size(knowledge_system.state.best_practices)
    
    innovation_score = min(1.0, (pipeline_size * 0.1 + practices_count * 0.05))
    innovation_score
  end
  
  defp process_autonomous_optimizations(state) do
    # Autonomous system optimizations
    optimizations = []
    
    # Revenue optimization
    if state.network_metrics.total_network_revenue < 80000 do
      optimizations = ["revenue_boost_initiative" | optimizations]
    end
    
    # Satisfaction optimization  
    if state.network_metrics.average_satisfaction < 4.2 do
      optimizations = ["customer_experience_enhancement" | optimizations]
    end
    
    # Efficiency optimization
    if state.network_metrics.operational_efficiency < 0.8 do
      optimizations = ["operational_efficiency_program" | optimizations]
    end
    
    # Execute optimizations
    if length(optimizations) > 0 do
      execute_autonomous_optimizations(state, optimizations)
    else
      state
    end
  end
  
  defp execute_autonomous_optimizations(state, optimizations) do
    IO.puts("🔧 Autonomous optimizations: #{Enum.join(optimizations, ", ")}")
    
    # Apply optimizations to relevant systems
    Enum.reduce(optimizations, state, fn optimization, acc_state ->
      case optimization do
        "revenue_boost_initiative" ->
          boost_network_revenue(acc_state)
        
        "customer_experience_enhancement" ->
          enhance_customer_experience(acc_state)
        
        "operational_efficiency_program" ->
          improve_operational_efficiency(acc_state)
        
        _ ->
          acc_state
      end
    end)
  end
  
  defp boost_network_revenue(state) do
    # Apply revenue boosting strategies to stores
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      revenue_boost = get_in(store.state, [:daily_metrics, :revenue]) * 0.05
                      updated_revenue = get_in(store.state, [:daily_metrics, :revenue]) + revenue_boost
                      
                      updated_store = Object.update_state(store, %{
                        daily_metrics: %{store.state.daily_metrics | revenue: updated_revenue}
                      })
                      
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    %{state | store_locations: updated_stores}
  end
  
  defp enhance_customer_experience(state) do
    # Improve customer satisfaction across network
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      satisfaction_boost = 0.1
                      new_satisfaction = min(5.0, store.state.daily_metrics.satisfaction + satisfaction_boost)
                      
                      updated_store = Object.update_state(store, %{
                        daily_metrics: %{store.state.daily_metrics | satisfaction: new_satisfaction}
                      })
                      
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    %{state | store_locations: updated_stores}
  end
  
  defp improve_operational_efficiency(state) do
    # Enhance operational efficiency across network
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      efficiency_boost = 0.05
                      new_efficiency = min(1.0, store.state.daily_metrics.efficiency + efficiency_boost)
                      
                      updated_store = Object.update_state(store, %{
                        daily_metrics: %{store.state.daily_metrics | efficiency: new_efficiency}
                      })
                      
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    %{state | store_locations: updated_stores}
  end
  
  defp cleanup_old_operations(state) do
    # Clean up old coordination events (keep last 50)
    cleaned_events = Enum.take(state.coordination_events, 50)
    
    # Clean up completed operations
    active_operations = state.active_operations
                       |> Enum.filter(fn {_, op} -> op.status != :completed end)
                       |> Map.new()
    
    %{state |
      coordination_events: cleaned_events,
      active_operations: active_operations
    }
  end
  
  defp update_performance_dashboard(state, tick_duration) do
    # Calculate dashboard metrics
    coordination_speed = 1000 / max(1, tick_duration)  # Operations per second
    
    decision_quality = state.headquarters.goal.(state.headquarters.state)
    
    resource_utilization = calculate_resource_utilization(state)
    
    knowledge_flow = state.knowledge_system.goal.(state.knowledge_system.state)
    
    network_resilience = calculate_network_resilience(state)
    
    %{
      coordination_speed: coordination_speed,
      decision_quality: decision_quality,
      resource_utilization: resource_utilization,
      knowledge_flow: knowledge_flow,
      network_resilience: network_resilience
    }
  end
  
  defp calculate_resource_utilization(state) do
    # Calculate how well resources are being used across the network
    store_performances = state.store_locations
                        |> Map.values()
                        |> Enum.map(&(&1.goal.(&1.state)))
    
    if length(store_performances) > 0 do
      Enum.sum(store_performances) / length(store_performances)
    else
      0.8
    end
  end
  
  defp calculate_network_resilience(state) do
    # Network resilience based on performance consistency
    regional_performances = state.regional_managers
                           |> Map.values()
                           |> Enum.map(&(&1.goal.(&1.state)))
    
    if length(regional_performances) > 1 do
      min_performance = Enum.min(regional_performances)
      max_performance = Enum.max(regional_performances)
      1.0 - (max_performance - min_performance)  # Lower variance = higher resilience
    else
      0.8
    end
  end
  
  defp calculate_network_health(state) do
    # Overall network health score
    metrics = state.network_metrics
    performance = state.performance_dashboard
    
    health_factors = [
      metrics.operational_efficiency,
      metrics.average_satisfaction / 5.0,
      metrics.brand_consistency,
      performance.decision_quality,
      performance.network_resilience
    ]
    
    Enum.sum(health_factors) / length(health_factors)
  end
  
  defp log_business_day_start(state) do
    IO.puts("\n🌅 Business Day Starting")
    IO.puts("   Network Revenue Target: $#{state.network_metrics.total_network_revenue}")
    IO.puts("   Network Health: #{Float.round(calculate_network_health(state) * 100, 1)}%")
    IO.puts("   Active Locations: #{map_size(state.store_locations)}")
  end
  
  defp log_business_day_end(state) do
    IO.puts("\n🌃 Business Day Ending")
    IO.puts("   Total Revenue: $#{trunc(state.network_metrics.total_network_revenue)}")
    IO.puts("   Avg Satisfaction: #{Float.round(state.network_metrics.average_satisfaction, 1)}/5.0")
    IO.puts("   Network Efficiency: #{Float.round(state.network_metrics.operational_efficiency * 100, 1)}%")
    IO.puts("   Brand Consistency: #{Float.round(state.network_metrics.brand_consistency * 100, 1)}%")
  end
  
  defp handle_network_event(state, event_type, params) do
    case event_type do
      :market_surge ->
        handle_market_surge(state, params)
      
      :supply_disruption ->
        handle_supply_disruption(state, params)
      
      :competitive_pressure ->
        handle_competitive_pressure(state, params)
      
      :expansion_opportunity ->
        handle_expansion_opportunity(state, params)
      
      :crisis_management ->
        handle_crisis_management(state, params)
      
      _ ->
        {state, %{result: :unknown_event}}
    end
  end
  
  defp handle_market_surge(state, _params) do
    # Market surge increases all store revenues
    surge_multiplier = 1.2 + :rand.uniform() * 0.3  # 20-50% increase
    
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      current_revenue = get_in(store.state, [:daily_metrics, :revenue])
                      boosted_revenue = current_revenue * surge_multiplier
                      
                      updated_store = Object.update_state(store, %{
                        daily_metrics: %{store.state.daily_metrics | revenue: boosted_revenue}
                      })
                      
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    IO.puts("📈 Market surge! Revenue boost: #{Float.round((surge_multiplier - 1) * 100, 1)}%")
    
    {%{state | store_locations: updated_stores}, %{result: :market_surge_handled, boost: surge_multiplier}}
  end
  
  defp handle_supply_disruption(state, params) do
    affected_region = Map.get(params, :region, :all)
    severity = Map.get(params, :severity, :moderate)
    
    # Impact supply chain and affected stores
    efficiency_impact = case severity do
      :minor -> 0.05
      :moderate -> 0.15
      :major -> 0.30
    end
    
    # Update supply chain
    updated_supply_chain = Object.update_state(state.supply_chain, %{
      logistics_optimization: max(0.5, state.supply_chain.state.logistics_optimization - efficiency_impact),
      delivery_performance: max(0.5, state.supply_chain.state.delivery_performance - efficiency_impact)
    })
    
    # Impact stores in affected region
    updated_stores = if affected_region == :all do
      impact_all_stores(state.store_locations, efficiency_impact)
    else
      impact_regional_stores(state.store_locations, affected_region, efficiency_impact)
    end
    
    IO.puts("🚚 Supply disruption (#{severity}) in #{affected_region}")
    
    {%{state | supply_chain: updated_supply_chain, store_locations: updated_stores}, 
     %{result: :supply_disruption_managed, severity: severity, region: affected_region}}
  end
  
  defp impact_all_stores(stores, impact) do
    stores
    |> Enum.map(fn {id, store} ->
      new_efficiency = max(0.5, store.state.daily_metrics.efficiency - impact)
      updated_store = Object.update_state(store, %{
        daily_metrics: %{store.state.daily_metrics | efficiency: new_efficiency}
      })
      {id, updated_store}
    end)
    |> Map.new()
  end
  
  defp impact_regional_stores(stores, target_region, impact) do
    stores
    |> Enum.map(fn {id, store} ->
      if get_in(store.state, [:region]) == target_region do
        new_efficiency = max(0.5, store.state.daily_metrics.efficiency - impact)
        updated_store = Object.update_state(store, %{
          daily_metrics: %{store.state.daily_metrics | efficiency: new_efficiency}
        })
        {id, updated_store}
      else
        {id, store}
      end
    end)
    |> Map.new()
  end
  
  defp handle_competitive_pressure(state, _params) do
    # Competitive pressure affects satisfaction and requires strategic response
    pressure_impact = 0.1 + :rand.uniform() * 0.1  # 10-20% impact
    
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      new_satisfaction = max(3.0, store.state.daily_metrics.satisfaction - pressure_impact)
                      updated_store = Object.update_state(store, %{
                        daily_metrics: %{store.state.daily_metrics | satisfaction: new_satisfaction}
                      })
                      {id, updated_store}
                    end)
                    |> Map.new()
    
    # HQ responds with strategic initiative
    updated_hq = Object.update_state(state.headquarters, %{
      strategic_initiatives: state.headquarters.state.strategic_initiatives ++ ["Competitive response program"],
      decision_queue: state.headquarters.state.decision_queue ++ ["Analyze competitive landscape", "Develop differentiation strategy"]
    })
    
    IO.puts("⚔️ Competitive pressure detected - strategic response initiated")
    
    {%{state | store_locations: updated_stores, headquarters: updated_hq}, 
     %{result: :competitive_pressure_handled, impact: pressure_impact}}
  end
  
  defp handle_expansion_opportunity(state, params) do
    target_region = Map.get(params, :region, :central)
    investment_required = Map.get(params, :investment, 1000000)
    
    # Create expansion opportunity
    expansion_opportunity = %{
      id: "expansion_#{:rand.uniform(1000)}",
      region: target_region,
      investment: investment_required,
      projected_roi: 2.2 + :rand.uniform() * 0.8,
      timeline: "12_months",
      status: :under_review
    }
    
    # Add to HQ decision queue
    updated_hq = Object.update_state(state.headquarters, %{
      decision_queue: state.headquarters.state.decision_queue ++ ["Evaluate expansion opportunity"],
      strategic_initiatives: state.headquarters.state.strategic_initiatives ++ ["#{target_region} region expansion"]
    })
    
    # Add to active operations
    updated_operations = Map.put(state.active_operations, expansion_opportunity.id, expansion_opportunity)
    
    IO.puts("🚀 Expansion opportunity identified: #{target_region} region")
    
    {%{state | headquarters: updated_hq, active_operations: updated_operations}, 
     %{result: :expansion_opportunity_created, region: target_region, investment: investment_required}}
  end
  
  defp handle_crisis_management(state, params) do
    crisis_type = Map.get(params, :type, :general)
    affected_locations = Map.get(params, :locations, [])
    
    # Activate crisis management protocols
    crisis_response = %{
      id: "crisis_#{:rand.uniform(1000)}",
      type: crisis_type,
      affected_locations: affected_locations,
      response_team: ["headquarters", "regional_managers"],
      status: :active,
      priority: :critical
    }
    
    # Update HQ with crisis management mode
    updated_hq = Object.update_state(state.headquarters, %{
      coordination_efficiency: state.headquarters.state.coordination_efficiency + 0.1,  # Crisis coordination boost
      decision_queue: ["Crisis response coordination" | state.headquarters.state.decision_queue]
    })
    
    # Activate emergency protocols in affected locations
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      if id in affected_locations or affected_locations == [] do
                        Object.update_state(store, %{operational_status: :emergency_mode})
                      else
                        {id, store}
                      end
                    end)
                    |> Map.new()
    
    updated_operations = Map.put(state.active_operations, crisis_response.id, crisis_response)
    
    IO.puts("🚨 Crisis management activated: #{crisis_type}")
    
    {%{state | headquarters: updated_hq, store_locations: updated_stores, active_operations: updated_operations}, 
     %{result: :crisis_management_activated, type: crisis_type}}
  end
  
  defp execute_resource_allocation(state, from_location, to_location, resources) do
    # Move resources between locations
    from_store = Map.get(state.store_locations, from_location)
    to_store = Map.get(state.store_locations, to_location)
    
    if from_store && to_store do
      # Simulate resource transfer
      allocation_operation = %{
        id: "allocation_#{:rand.uniform(1000)}",
        from: from_location,
        to: to_location,
        resources: resources,
        status: :in_progress,
        completion_time: DateTime.add(DateTime.utc_now(), 3600, :second)  # 1 hour
      }
      
      updated_operations = Map.put(state.active_operations, allocation_operation.id, allocation_operation)
      %{state | active_operations: updated_operations}
    else
      state
    end
  end
  
  defp launch_network_promotion(state, promotion_data) do
    # Launch promotion across network
    promotion = %{
      id: "promo_#{:rand.uniform(1000)}",
      name: promotion_data.name,
      type: Map.get(promotion_data, :type, :discount),
      value: Map.get(promotion_data, :value, 0.1),
      duration: Map.get(promotion_data, :duration, 7),
      target_locations: Map.get(promotion_data, :locations, :all),
      status: :active
    }
    
    # Apply promotion to stores
    updated_stores = state.store_locations
                    |> Enum.map(fn {id, store} ->
                      if promotion.target_locations == :all or id in promotion.target_locations do
                        current_promotions = store.state.current_promotions
                        updated_store = Object.update_state(store, %{
                          current_promotions: [promotion | current_promotions]
                        })
                        {id, updated_store}
                      else
                        {id, store}
                      end
                    end)
                    |> Map.new()
    
    updated_operations = Map.put(state.active_operations, promotion.id, promotion)
    
    %{state | store_locations: updated_stores, active_operations: updated_operations}
  end
  
  defp coordinate_emergency_response(state, emergency_type, affected_locations) do
    # Coordinate emergency response across network
    emergency_coordination = %{
      id: "emergency_#{:rand.uniform(1000)}",
      type: emergency_type,
      affected_locations: affected_locations,
      response_protocol: determine_response_protocol(emergency_type),
      coordination_team: ["headquarters"] ++ get_affected_regional_managers(affected_locations, state),
      status: :coordinating
    }
    
    # Update HQ coordination mode
    updated_hq = Object.update_state(state.headquarters, %{
      coordination_efficiency: min(1.0, state.headquarters.state.coordination_efficiency + 0.15)
    })
    
    updated_operations = Map.put(state.active_operations, emergency_coordination.id, emergency_coordination)
    
    %{state | headquarters: updated_hq, active_operations: updated_operations}
  end
  
  defp determine_response_protocol(emergency_type) do
    case emergency_type do
      :natural_disaster -> "Evacuate and secure facilities, coordinate with local authorities"
      :security_breach -> "Lockdown affected locations, investigate and secure systems"
      :supply_chain_failure -> "Activate backup suppliers, redistribute inventory"
      :system_outage -> "Switch to backup systems, coordinate manual operations"
      _ -> "Assess situation and coordinate appropriate response"
    end
  end
  
  defp get_affected_regional_managers(affected_locations, state) do
    # Find regional managers responsible for affected locations
    state.store_locations
    |> Enum.filter(fn {id, _} -> id in affected_locations end)
    |> Enum.map(fn {_, store} -> get_in(store.state, [:region]) end)
    |> Enum.uniq()
    |> Enum.map(fn region -> "region_#{region}" end)
  end
  
  defp log_coordination_event(state, event_type, result) do
    event = %{
      type: event_type,
      timestamp: DateTime.utc_now(),
      result: result
    }
    
    %{state | coordination_events: [event | state.coordination_events]}
  end
  
  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end
end

# Interactive control module
defmodule RealtimeFranchiseNetwork.Controller do
  @moduledoc """
  Interactive control interface for the franchise network
  """
  
  def start do
    IO.puts("\n🎮 Realtime Franchise Network Controller")
    IO.puts("=" |> String.duplicate(50))
    IO.puts("Commands:")
    IO.puts("  status            - Show network status")
    IO.puts("  metrics           - Show network metrics")
    IO.puts("  locations         - List all locations")
    IO.puts("  location <id>     - Show location details")
    IO.puts("  regions           - Show regional managers")
    IO.puts("  region <region>   - Show regional details")
    IO.puts("  surge             - Trigger market surge")
    IO.puts("  disruption <region> <severity> - Trigger supply disruption")
    IO.puts("  competition       - Trigger competitive pressure")
    IO.puts("  expand <region>   - Create expansion opportunity")
    IO.puts("  crisis <type>     - Trigger crisis management")
    IO.puts("  allocate <from> <to> <resources> - Allocate resources")
    IO.puts("  promotion <name>  - Launch network promotion")
    IO.puts("  emergency <type> <locations> - Emergency coordination")
    IO.puts("  stop              - Stop the network")
    IO.puts("  help              - Show this help")
    IO.puts("  quit              - Exit controller")
    IO.puts("")
    
    command_loop()
  end
  
  defp command_loop do
    input = IO.gets("🏢 > ") |> String.trim()
    
    case String.split(input, " ", parts: 4) do
      ["quit"] ->
        IO.puts("Goodbye!")
        
      ["help"] ->
        start()
        
      ["status"] ->
        show_status()
        command_loop()
        
      ["metrics"] ->
        show_metrics()
        command_loop()
        
      ["locations"] ->
        show_locations()
        command_loop()
        
      ["location", location_id] ->
        show_location(location_id)
        command_loop()
        
      ["regions"] ->
        show_regions()
        command_loop()
        
      ["region", region] ->
        show_region(String.to_atom(region))
        command_loop()
        
      ["surge"] ->
        RealtimeFranchiseNetwork.trigger_network_event(:market_surge)
        command_loop()
        
      ["disruption", region, severity] ->
        RealtimeFranchiseNetwork.trigger_network_event(:supply_disruption, %{
          region: String.to_atom(region),
          severity: String.to_atom(severity)
        })
        command_loop()
        
      ["competition"] ->
        RealtimeFranchiseNetwork.trigger_network_event(:competitive_pressure)
        command_loop()
        
      ["expand", region] ->
        RealtimeFranchiseNetwork.trigger_network_event(:expansion_opportunity, %{region: String.to_atom(region)})
        command_loop()
        
      ["crisis", crisis_type] ->
        RealtimeFranchiseNetwork.trigger_network_event(:crisis_management, %{type: String.to_atom(crisis_type)})
        command_loop()
        
      ["allocate", from, to, resources] ->
        RealtimeFranchiseNetwork.allocate_resources(from, to, resources)
        command_loop()
        
      ["promotion", name] ->
        RealtimeFranchiseNetwork.launch_promotion(%{name: name, type: :discount, value: 0.15})
        command_loop()
        
      ["emergency", emergency_type, locations] ->
        location_list = String.split(locations, ",")
        RealtimeFranchiseNetwork.emergency_coordination(String.to_atom(emergency_type), location_list)
        command_loop()
        
      ["stop"] ->
        RealtimeFranchiseNetwork.stop()
        IO.puts("Franchise network stopped.")
        command_loop()
        
      _ ->
        IO.puts("Unknown command. Type 'help' for available commands.")
        command_loop()
    end
  end
  
  defp show_status do
    case RealtimeFranchiseNetwork.status() do
      status when is_map(status) ->
        business_status = if status.business_hours, do: "BUSINESS HOURS", else: "AFTER HOURS"
        
        IO.puts("\n🏢 Franchise Network Status: #{business_status}")
        IO.puts("   Uptime: #{status.uptime} seconds")
        IO.puts("   Current Time: #{Calendar.strftime(status.current_time, "%H:%M")}")
        IO.puts("   Total Locations: #{status.total_locations}")
        IO.puts("   Total Regions: #{status.total_regions}")
        IO.puts("   Active Operations: #{status.active_operations}")
        IO.puts("   Network Health: #{Float.round(status.network_health * 100, 1)}%")
        IO.puts("   Performance Dashboard:")
        IO.puts("     Coordination Speed: #{Float.round(status.performance.coordination_speed, 1)} ops/sec")
        IO.puts("     Decision Quality: #{Float.round(status.performance.decision_quality * 100, 1)}%")
        IO.puts("     Resource Utilization: #{Float.round(status.performance.resource_utilization * 100, 1)}%")
        IO.puts("     Network Resilience: #{Float.round(status.performance.network_resilience * 100, 1)}%")
        
      error ->
        IO.puts("Error getting status: #{inspect(error)}")
    end
  end
  
  defp show_metrics do
    case RealtimeFranchiseNetwork.get_network_metrics() do
      metrics when is_map(metrics) ->
        IO.puts("\n📊 Network Metrics:")
        IO.puts("   Total Revenue: $#{trunc(metrics.total_network_revenue)}")
        IO.puts("   Average Satisfaction: #{Float.round(metrics.average_satisfaction, 1)}/5.0")
        IO.puts("   Operational Efficiency: #{Float.round(metrics.operational_efficiency * 100, 1)}%")
        IO.puts("   Brand Consistency: #{Float.round(metrics.brand_consistency * 100, 1)}%")
        IO.puts("   Market Penetration: #{Float.round(metrics.market_penetration * 100, 1)}%")
        IO.puts("   Customer Retention: #{Float.round(metrics.customer_retention * 100, 1)}%")
        IO.puts("   Innovation Index: #{Float.round(metrics.innovation_index * 100, 1)}%")
        IO.puts("   Sustainability Score: #{Float.round(metrics.sustainability_score * 100, 1)}%")
        
      error ->
        IO.puts("Error getting metrics: #{inspect(error)}")
    end
  end
  
  defp show_locations do
    location_ids = [
      "store_north_metro", "store_north_suburban", "store_north_mall",
      "store_south_downtown", "store_south_plaza", "store_south_outlet",
      "store_west_coast", "store_west_valley"
    ]
    
    IO.puts("\n🏪 Store Locations:")
    for location_id <- location_ids do
      case RealtimeFranchiseNetwork.get_location(location_id) do
        nil -> IO.puts("   #{location_id}: Not found")
        location ->
          performance = location.goal.(location.state)
          revenue = get_in(location.state, [:daily_metrics, :revenue]) || 0
          satisfaction = get_in(location.state, [:daily_metrics, :satisfaction]) || 0
          store_type = get_in(location.state, [:store_type])
          region = get_in(location.state, [:region])
          
          IO.puts("   #{location_id} (#{store_type}, #{region}):")
          IO.puts("     Performance: #{Float.round(performance * 100, 1)}%")
          IO.puts("     Revenue: $#{trunc(revenue)}")
          IO.puts("     Satisfaction: #{Float.round(satisfaction, 1)}/5.0")
      end
    end
  end
  
  defp show_location(location_id) do
    case RealtimeFranchiseNetwork.get_location(location_id) do
      nil ->
        IO.puts("Location '#{location_id}' not found")
        
      location ->
        IO.puts("\n🏪 Location: #{location_id}")
        IO.puts("   Type: #{get_in(location.state, [:store_type])}")
        IO.puts("   Region: #{get_in(location.state, [:region])}")
        IO.puts("   Performance: #{Float.round(location.goal.(location.state) * 100, 1)}%")
        IO.puts("   Status: #{get_in(location.state, [:operational_status])}")
        
        metrics = get_in(location.state, [:daily_metrics]) || %{}
        IO.puts("   Daily Metrics:")
        IO.puts("     Revenue: $#{trunc(Map.get(metrics, :revenue, 0))}")
        IO.puts("     Customers: #{Map.get(metrics, :customers, 0)}")
        IO.puts("     Satisfaction: #{Float.round(Map.get(metrics, :satisfaction, 0), 1)}/5.0")
        IO.puts("     Efficiency: #{Float.round(Map.get(metrics, :efficiency, 0) * 100, 1)}%")
        
        characteristics = get_in(location.state, [:characteristics]) || %{}
        IO.puts("   Characteristics:")
        IO.puts("     Location Type: #{Map.get(characteristics, :location_type)}")
        IO.puts("     Daily Traffic: #{Map.get(characteristics, :daily_traffic)}")
        IO.puts("     Specializations: #{Enum.join(Map.get(characteristics, :specializations, []), ", ")}")
    end
  end
  
  defp show_regions do
    regions = [:north, :south, :west]
    
    IO.puts("\n🌎 Regional Managers:")
    for region <- regions do
      case RealtimeFranchiseNetwork.get_regional_manager(region) do
        nil -> IO.puts("   #{region}: Not found")
        {id, manager} ->
          performance = manager.goal.(manager.state)
          revenue = get_in(manager.state, [:regional_performance, :revenue]) || 0
          stores = length(get_in(manager.state, [:managed_stores]) || [])
          
          IO.puts("   #{String.capitalize(to_string(region))} Region (#{id}):")
          IO.puts("     Performance: #{Float.round(performance * 100, 1)}%")
          IO.puts("     Revenue: $#{trunc(revenue)}")
          IO.puts("     Managed Stores: #{stores}")
          IO.puts("     Leadership Style: #{get_in(manager.state, [:leadership_style])}")
      end
    end
  end
  
  defp show_region(region) do
    case RealtimeFranchiseNetwork.get_regional_manager(region) do
      nil ->
        IO.puts("Region '#{region}' not found")
        
      {id, manager} ->
        IO.puts("\n🌎 Region: #{String.capitalize(to_string(region))}")
        IO.puts("   Manager ID: #{id}")
        IO.puts("   Performance: #{Float.round(manager.goal.(manager.state) * 100, 1)}%")
        IO.puts("   Leadership Style: #{get_in(manager.state, [:leadership_style])}")
        
        regional_perf = get_in(manager.state, [:regional_performance]) || %{}
        IO.puts("   Regional Performance:")
        IO.puts("     Revenue: $#{trunc(Map.get(regional_perf, :revenue, 0))}")
        IO.puts("     Satisfaction: #{Float.round(Map.get(regional_perf, :customer_satisfaction, 0), 1)}/5.0")
        IO.puts("     Efficiency: #{Float.round(Map.get(regional_perf, :operational_efficiency, 0) * 100, 1)}%")
        IO.puts("     Market Share: #{Float.round(Map.get(regional_perf, :market_share, 0) * 100, 1)}%")
        
        managed_stores = get_in(manager.state, [:managed_stores]) || []
        IO.puts("   Managed Stores: #{Enum.join(managed_stores, ", ")}")
        
        tasks = get_in(manager.state, [:coordination_tasks]) || []
        if length(tasks) > 0 do
          IO.puts("   Current Tasks:")
          for task <- Enum.take(tasks, 3) do
            IO.puts("     • #{task}")
          end
        end
    end
  end
end

# Run the system
case System.argv() do
  ["start"] ->
    {:ok, _pid} = RealtimeFranchiseNetwork.start_link()
    IO.puts("\n🎮 Starting interactive controller...")
    RealtimeFranchiseNetwork.Controller.start()
    
  ["demo"] ->
    {:ok, _pid} = RealtimeFranchiseNetwork.start_link()
    IO.puts("🏢 Running franchise network demo for 90 seconds...")
    
    # Trigger events during demo
    spawn(fn ->
      :timer.sleep(10000)
      RealtimeFranchiseNetwork.trigger_network_event(:market_surge)
      
      :timer.sleep(15000)
      RealtimeFranchiseNetwork.trigger_network_event(:supply_disruption, %{region: :north, severity: :moderate})
      
      :timer.sleep(20000)
      RealtimeFranchiseNetwork.launch_promotion(%{name: "Summer Sale", type: :discount, value: 0.2})
      
      :timer.sleep(15000)
      RealtimeFranchiseNetwork.trigger_network_event(:competitive_pressure)
      
      :timer.sleep(15000)
      RealtimeFranchiseNetwork.trigger_network_event(:expansion_opportunity, %{region: :central})
      
      :timer.sleep(10000)
      RealtimeFranchiseNetwork.emergency_coordination(:system_outage, ["store_west_coast"])
    end)
    
    :timer.sleep(90000)
    RealtimeFranchiseNetwork.stop()
    IO.puts("Franchise network demo complete!")
    
  _ ->
    IO.puts("Usage:")
    IO.puts("  elixir examples/realtime_franchise_network.exs start   # Start interactive network")
    IO.puts("  elixir examples/realtime_franchise_network.exs demo    # Run 90-second demo")
end
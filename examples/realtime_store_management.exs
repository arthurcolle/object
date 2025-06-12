#!/usr/bin/env elixir

# Realtime Store Management System
# Living store with continuous operations, real customer flow, and dynamic responses

defmodule RealtimeStoreManagement do
  @moduledoc """
  Realtime store management system that operates as a living store.
  
  Features:
  - Continuous customer flow simulation
  - Real-time inventory management
  - Dynamic pricing and promotions
  - Live staff performance monitoring
  - Autonomous store operations
  - Interactive management interface
  """
  
  use GenServer
  require Logger
  
  defstruct [
    :store_objects,
    :active_customers,
    :daily_metrics,
    :running,
    :start_time,
    :current_time,
    :store_hours,
    :event_log,
    :performance_stats,
    :alerts
  ]
  
  # Store configuration
  @store_open_hour 8
  @store_close_hour 22
  @customer_spawn_rate 0.3  # Probability per tick
  @tick_interval 2000       # 2 seconds
  
  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end
  
  def stop do
    GenServer.call(__MODULE__, :stop)
  end
  
  def status do
    GenServer.call(__MODULE__, :status)
  end
  
  def metrics do
    GenServer.call(__MODULE__, :metrics)
  end
  
  def get_object(object_id) do
    GenServer.call(__MODULE__, {:get_object, object_id})
  end
  
  def trigger_event(event_type, params \\ %{}) do
    GenServer.cast(__MODULE__, {:trigger_event, event_type, params})
  end
  
  def emergency_alert(alert_type, message) do
    GenServer.cast(__MODULE__, {:emergency_alert, alert_type, message})
  end
  
  def adjust_pricing(category, adjustment) do
    GenServer.cast(__MODULE__, {:adjust_pricing, category, adjustment})
  end
  
  def get_customers do
    GenServer.call(__MODULE__, :get_customers)
  end
  
  def get_alerts do
    GenServer.call(__MODULE__, :get_alerts)
  end
  
  @impl true
  def init(_opts) do
    IO.puts("🏪 Initializing Realtime Store Management System...")
    
    # Create store objects
    store_objects = create_store_objects()
    
    # Initialize state
    state = %__MODULE__{
      store_objects: store_objects |> Enum.map(&{&1.id, &1}) |> Map.new(),
      active_customers: %{},
      daily_metrics: initialize_daily_metrics(),
      running: true,
      start_time: DateTime.utc_now(),
      current_time: DateTime.utc_now(),
      store_hours: %{open: @store_open_hour, close: @store_close_hour},
      event_log: [],
      performance_stats: %{
        customers_per_hour: 0,
        average_transaction_value: 0,
        staff_efficiency: 0,
        inventory_turnover: 0
      },
      alerts: []
    }
    
    # Start the store operations
    schedule_tick()
    
    IO.puts("✅ Realtime store system started!")
    IO.puts("   • Store objects: #{map_size(state.store_objects)}")
    IO.puts("   • Store hours: #{@store_open_hour}:00 - #{@store_close_hour}:00")
    IO.puts("   • Tick interval: #{@tick_interval}ms")
    
    {:ok, state}
  end
  
  @impl true
  def handle_call(:stop, _from, state) do
    IO.puts("🛑 Closing store...")
    {:stop, :normal, :ok, %{state | running: false}}
  end
  
  @impl true
  def handle_call(:status, _from, state) do
    store_status = %{
      running: state.running,
      store_open: is_store_open?(state.current_time),
      uptime: DateTime.diff(DateTime.utc_now(), state.start_time),
      active_customers: map_size(state.active_customers),
      store_objects: map_size(state.store_objects),
      alerts: length(state.alerts),
      current_time: state.current_time,
      performance: state.performance_stats
    }
    {:reply, store_status, state}
  end
  
  @impl true
  def handle_call(:metrics, _from, state) do
    {:reply, state.daily_metrics, state}
  end
  
  @impl true
  def handle_call({:get_object, object_id}, _from, state) do
    object = Map.get(state.store_objects, object_id)
    {:reply, object, state}
  end
  
  @impl true
  def handle_call(:get_customers, _from, state) do
    {:reply, state.active_customers, state}
  end
  
  @impl true
  def handle_call(:get_alerts, _from, state) do
    {:reply, state.alerts, state}
  end
  
  @impl true
  def handle_cast({:trigger_event, event_type, params}, state) do
    {new_state, result} = handle_triggered_event(state, event_type, params)
    log_event(new_state, event_type, result)
    {:noreply, new_state}
  end
  
  @impl true
  def handle_cast({:emergency_alert, alert_type, message}, state) do
    alert = %{
      type: alert_type,
      message: message,
      timestamp: DateTime.utc_now(),
      severity: :high
    }
    
    new_alerts = [alert | state.alerts]
    IO.puts("🚨 ALERT: #{alert_type} - #{message}")
    
    {:noreply, %{state | alerts: new_alerts}}
  end
  
  @impl true
  def handle_cast({:adjust_pricing, category, adjustment}, state) do
    # Update pricing in store objects
    updated_objects = state.store_objects
                     |> Enum.map(fn {id, object} ->
                       if object.subtype == :actuator_object do  # Sales associate
                         pricing_update = %{
                           "#{category}_price_adjustment" => adjustment
                         }
                         updated_object = Object.update_state(object, pricing_update)
                         {id, updated_object}
                       else
                         {id, object}
                       end
                     end)
                     |> Map.new()
    
    event = %{
      type: :pricing_adjustment,
      timestamp: DateTime.utc_now(),
      data: %{category: category, adjustment: adjustment}
    }
    
    new_state = %{state |
      store_objects: updated_objects,
      event_log: [event | state.event_log]
    }
    
    IO.puts("💰 Pricing adjusted for #{category}: #{adjustment}%")
    {:noreply, new_state}
  end
  
  @impl true
  def handle_info(:tick, %{running: false} = state) do
    {:noreply, state}
  end
  
  @impl true
  def handle_info(:tick, state) do
    tick_start = System.monotonic_time(:millisecond)
    
    # Update current time (accelerated - 1 tick = 1 minute)
    new_time = DateTime.add(state.current_time, 60, :second)
    
    new_state = %{state | current_time: new_time}
                |> process_store_operations()
                |> spawn_customers()
                |> process_customer_activities()
                |> update_store_objects()
                |> process_inventory_management()
                |> update_daily_metrics()
                |> cleanup_old_data()
    
    # Calculate performance
    tick_duration = System.monotonic_time(:millisecond) - tick_start
    updated_stats = calculate_performance_stats(new_state, tick_duration)
    
    final_state = %{new_state | performance_stats: updated_stats}
    
    # Log hourly status
    if rem(DateTime.to_time(final_state.current_time).minute, 60) == 0 do
      log_hourly_status(final_state)
    end
    
    schedule_tick()
    {:noreply, final_state}
  end
  
  # Private functions
  
  defp create_store_objects do
    # Store Manager
    store_manager = Object.new(
      id: "store_manager_001",
      subtype: :coordinator_object,
      state: %{
        role: :store_manager,
        daily_targets: %{sales: 10000, satisfaction: 4.5},
        current_performance: %{sales: 0, satisfaction: 0},
        management_decisions: [],
        stress_level: 0.2,
        shift_start: @store_open_hour
      },
      methods: [:manage_operations, :make_decisions, :coordinate_staff, :handle_emergencies],
      goal: fn state ->
        sales_performance = min(1.0, state.current_performance.sales / state.daily_targets.sales)
        satisfaction_performance = state.current_performance.satisfaction / 5.0
        stress_penalty = 1.0 - state.stress_level
        (sales_performance + satisfaction_performance + stress_penalty) / 3
      end
    )
    
    # Inventory Manager
    inventory_manager = Object.new(
      id: "inventory_manager_002",
      subtype: :sensor_object,
      state: %{
        role: :inventory_manager,
        inventory_levels: %{
          electronics: %{current: 45, min: 20, max: 100, reorder_point: 25},
          clothing: %{current: 78, min: 30, max: 150, reorder_point: 40},
          home_goods: %{current: 23, min: 15, max: 80, reorder_point: 20},
          books: %{current: 156, min: 50, max: 200, reorder_point: 60}
        },
        pending_orders: [],
        last_audit: DateTime.utc_now(),
        efficiency_score: 0.95
      },
      methods: [:monitor_inventory, :generate_orders, :receive_shipments, :audit_stock],
      goal: fn state ->
        # Goal: maintain optimal inventory levels
        levels = state.inventory_levels
        total_health = Enum.reduce(levels, 0, fn {_, level}, acc ->
          if level.current >= level.min do
            acc + 1
          else
            acc + 0.5  # Penalty for low stock
          end
        end)
        (total_health / map_size(levels)) * state.efficiency_score
      end
    )
    
    # Customer Service Representative
    customer_service = Object.new(
      id: "customer_service_003",
      subtype: :ai_agent,
      state: %{
        role: :customer_service,
        active_interactions: 0,
        response_times: [],
        satisfaction_scores: [],
        energy_level: 1.0,
        break_needed: false,
        expertise_areas: [:general, :electronics, :returns]
      },
      methods: [:assist_customer, :handle_complaint, :process_return, :answer_questions],
      goal: fn state ->
        avg_satisfaction = if length(state.satisfaction_scores) > 0 do
          Enum.sum(state.satisfaction_scores) / length(state.satisfaction_scores) / 5.0
        else
          0.8
        end
        
        response_efficiency = if length(state.response_times) > 0 do
          avg_response = Enum.sum(state.response_times) / length(state.response_times)
          max(0.0, 1.0 - avg_response / 300)  # 5 minutes max response time
        else
          0.8
        end
        
        (avg_satisfaction + response_efficiency + state.energy_level) / 3
      end
    )
    
    # Sales Associate
    sales_associate = Object.new(
      id: "sales_associate_004",
      subtype: :actuator_object,
      state: %{
        role: :sales_associate,
        transactions_today: 0,
        total_sales: 0,
        upsell_success_rate: 0.35,
        customer_queue: [],
        commission_earned: 0,
        product_knowledge: %{electronics: 0.9, clothing: 0.7, home_goods: 0.8, books: 0.6}
      },
      methods: [:process_sale, :suggest_products, :handle_payment, :update_inventory],
      goal: fn state ->
        sales_performance = min(1.0, state.total_sales / 2000)  # Daily target $2000
        upsell_performance = state.upsell_success_rate
        queue_efficiency = max(0.0, 1.0 - length(state.customer_queue) / 10)
        (sales_performance + upsell_performance + queue_efficiency) / 3
      end
    )
    
    # Security System
    security_system = Object.new(
      id: "security_system_005",
      subtype: :sensor_object,
      state: %{
        role: :security,
        cameras_active: 8,
        motion_detectors: 6,
        alerts_today: 0,
        false_alarms: 0,
        visitor_count: 0,
        suspicious_activities: [],
        system_health: 1.0
      },
      methods: [:monitor_premises, :detect_threats, :count_visitors, :generate_alerts],
      goal: fn state ->
        detection_accuracy = if state.alerts_today + state.false_alarms > 0 do
          state.alerts_today / (state.alerts_today + state.false_alarms)
        else
          1.0
        end
        
        (detection_accuracy + state.system_health) / 2
      end
    )
    
    [store_manager, inventory_manager, customer_service, sales_associate, security_system]
  end
  
  defp initialize_daily_metrics do
    %{
      total_sales: 0,
      transaction_count: 0,
      customer_count: 0,
      average_transaction_value: 0,
      customer_satisfaction: 4.2,
      inventory_alerts: 0,
      security_incidents: 0,
      staff_performance: 0.8,
      hour_by_hour_sales: %{}
    }
  end
  
  defp is_store_open?(current_time) do
    hour = DateTime.to_time(current_time).hour
    hour >= @store_open_hour and hour < @store_close_hour
  end
  
  defp process_store_operations(state) do
    if is_store_open?(state.current_time) do
      # Store is open - normal operations
      state
    else
      # Store is closed - night operations
      process_night_operations(state)
    end
  end
  
  defp process_night_operations(state) do
    # Overnight restocking, cleaning, maintenance
    if DateTime.to_time(state.current_time).hour == 2 do
      # 2 AM - major restocking
      restock_inventory(state)
    else
      state
    end
  end
  
  defp restock_inventory(state) do
    inventory_manager = Map.get(state.store_objects, "inventory_manager_002")
    
    updated_inventory = inventory_manager.state.inventory_levels
                       |> Enum.map(fn {category, levels} ->
                         # Restock to 80% of max capacity
                         target_level = trunc(levels.max * 0.8)
                         new_current = max(levels.current, target_level)
                         {category, %{levels | current: new_current}}
                       end)
                       |> Map.new()
    
    updated_manager = Object.update_state(inventory_manager, %{inventory_levels: updated_inventory})
    updated_objects = Map.put(state.store_objects, "inventory_manager_002", updated_manager)
    
    event = %{
      type: :overnight_restocking,
      timestamp: DateTime.utc_now(),
      data: %{restocked_categories: Map.keys(updated_inventory)}
    }
    
    %{state |
      store_objects: updated_objects,
      event_log: [event | state.event_log]
    }
  end
  
  defp spawn_customers(state) do
    if is_store_open?(state.current_time) and :rand.uniform() < @customer_spawn_rate do
      customer = create_random_customer(state.current_time)
      updated_customers = Map.put(state.active_customers, customer.id, customer)
      
      # Update visitor count in security system
      security = Map.get(state.store_objects, "security_system_005")
      updated_security = Object.update_state(security, %{visitor_count: security.state.visitor_count + 1})
      updated_objects = Map.put(state.store_objects, "security_system_005", updated_security)
      
      %{state |
        active_customers: updated_customers,
        store_objects: updated_objects
      }
    else
      state
    end
  end
  
  defp create_random_customer(current_time) do
    customer_types = [:casual_browser, :specific_shopper, :bargain_hunter, :loyal_customer, :difficult_customer]
    customer_type = Enum.random(customer_types)
    
    %{
      id: "customer_#{:rand.uniform(10000)}",
      type: customer_type,
      entered_at: current_time,
      budget: 50 + :rand.uniform(200),
      satisfaction: 0.8,
      items_wanted: generate_shopping_list(customer_type),
      patience: 0.7 + :rand.uniform() * 0.3,
      current_activity: :browsing,
      time_spent: 0,
      assistance_needed: false
    }
  end
  
  defp generate_shopping_list(customer_type) do
    all_items = [
      %{category: :electronics, item: "laptop", price: 800},
      %{category: :electronics, item: "phone", price: 400},
      %{category: :electronics, item: "headphones", price: 150},
      %{category: :clothing, item: "jacket", price: 80},
      %{category: :clothing, item: "shoes", price: 120},
      %{category: :home_goods, item: "lamp", price: 45},
      %{category: :home_goods, item: "pillow", price: 25},
      %{category: :books, item: "novel", price: 15},
      %{category: :books, item: "cookbook", price: 30}
    ]
    
    case customer_type do
      :casual_browser -> Enum.take_random(all_items, :rand.uniform(2))
      :specific_shopper -> Enum.take_random(all_items, 1 + :rand.uniform(2))
      :bargain_hunter -> Enum.filter(all_items, &(&1.price < 100)) |> Enum.take_random(3)
      :loyal_customer -> Enum.take_random(all_items, 2 + :rand.uniform(3))
      :difficult_customer -> Enum.take_random(all_items, 1 + :rand.uniform(2))
    end
  end
  
  defp process_customer_activities(state) do
    updated_customers = state.active_customers
                       |> Enum.map(fn {id, customer} ->
                         updated_customer = process_customer_tick(customer, state)
                         {id, updated_customer}
                       end)
                       |> Enum.reject(fn {_, customer} -> customer == :remove end)
                       |> Map.new()
    
    %{state | active_customers: updated_customers}
  end
  
  defp process_customer_tick(customer, state) do
    updated_customer = %{customer | time_spent: customer.time_spent + 1}
    
    case updated_customer.current_activity do
      :browsing ->
        if updated_customer.time_spent > 5 or :rand.uniform() < 0.3 do
          if length(updated_customer.items_wanted) > 0 and :rand.uniform() < 0.7 do
            %{updated_customer | current_activity: :ready_to_buy}
          else
            # Customer leaves without buying
            process_customer_exit(updated_customer, state, :no_purchase)
            :remove
          end
        else
          updated_customer
        end
        
      :ready_to_buy ->
        # Customer goes to checkout
        process_customer_purchase(updated_customer, state)
        :remove
        
      :needs_assistance ->
        if updated_customer.patience > 0 do
          if :rand.uniform() < 0.5 do
            # Customer gets assistance
            %{updated_customer | current_activity: :being_assisted}
          else
            %{updated_customer | patience: updated_customer.patience - 0.1}
          end
        else
          # Customer leaves frustrated
          process_customer_exit(updated_customer, state, :poor_service)
          :remove
        end
        
      :being_assisted ->
        # Customer completes interaction
        %{updated_customer | current_activity: :ready_to_buy, satisfaction: min(1.0, updated_customer.satisfaction + 0.2)}
        
      _ ->
        updated_customer
    end
  end
  
  defp process_customer_purchase(customer, state) do
    sales_associate = Map.get(state.store_objects, "sales_associate_004")
    
    # Calculate purchase
    items_to_buy = Enum.filter(customer.items_wanted, fn item ->
      item.price <= customer.budget and :rand.uniform() < 0.8
    end)
    
    total_amount = Enum.sum(Enum.map(items_to_buy, & &1.price))
    
    if total_amount > 0 do
      # Process successful sale
      process_successful_sale(customer, items_to_buy, total_amount, state)
    else
      # Customer leaves without buying
      process_customer_exit(customer, state, :no_suitable_items)
    end
  end
  
  defp process_successful_sale(customer, items, total_amount, state) do
    # Update sales associate
    sales_associate = Map.get(state.store_objects, "sales_associate_004")
    updated_sales = Object.update_state(sales_associate, %{
      transactions_today: sales_associate.state.transactions_today + 1,
      total_sales: sales_associate.state.total_sales + total_amount
    })
    
    # Update inventory
    inventory_manager = Map.get(state.store_objects, "inventory_manager_002")
    updated_inventory = reduce_inventory_for_sale(inventory_manager.state.inventory_levels, items)
    updated_inventory_manager = Object.update_state(inventory_manager, %{inventory_levels: updated_inventory})
    
    # Log sale event
    sale_event = %{
      type: :sale_completed,
      timestamp: DateTime.utc_now(),
      data: %{
        customer_id: customer.id,
        items: items,
        total: total_amount,
        customer_satisfaction: customer.satisfaction
      }
    }
    
    IO.puts("💰 Sale completed: $#{total_amount} (#{length(items)} items)")
  end
  
  defp process_customer_exit(customer, _state, reason) do
    exit_event = %{
      type: :customer_exit,
      timestamp: DateTime.utc_now(),
      data: %{
        customer_id: customer.id,
        reason: reason,
        time_spent: customer.time_spent,
        satisfaction: customer.satisfaction
      }
    }
    
    case reason do
      :no_purchase -> nil
      :poor_service -> IO.puts("😞 Customer left due to poor service")
      :no_suitable_items -> nil
    end
  end
  
  defp reduce_inventory_for_sale(inventory_levels, items) do
    Enum.reduce(items, inventory_levels, fn item, acc ->
      case Map.get(acc, item.category) do
        nil -> acc
        level_data ->
          updated_level = %{level_data | current: max(0, level_data.current - 1)}
          Map.put(acc, item.category, updated_level)
      end
    end)
  end
  
  defp update_store_objects(state) do
    updated_objects = state.store_objects
                     |> Enum.map(fn {id, object} ->
                       updated_object = update_object_autonomous_behavior(object, state)
                       {id, updated_object}
                     end)
                     |> Map.new()
    
    %{state | store_objects: updated_objects}
  end
  
  defp update_object_autonomous_behavior(object, state) do
    case get_in(object.state, [:role]) do
      :store_manager ->
        update_manager_behavior(object, state)
      
      :inventory_manager ->
        update_inventory_manager_behavior(object, state)
      
      :customer_service ->
        update_customer_service_behavior(object, state)
      
      :sales_associate ->
        update_sales_associate_behavior(object, state)
      
      :security ->
        update_security_behavior(object, state)
      
      _ ->
        object
    end
  end
  
  defp update_manager_behavior(manager, state) do
    # Manager monitors overall performance and makes adjustments
    current_hour = DateTime.to_time(state.current_time).hour
    sales_target_hour = manager.state.daily_targets.sales / 14  # 14 operating hours
    
    hour_sales = get_hour_sales(state.daily_metrics, current_hour)
    
    stress_adjustment = if hour_sales < sales_target_hour * 0.8 do
      0.1  # Increase stress if behind target
    else
      -0.05  # Reduce stress if meeting target
    end
    
    new_stress = max(0.0, min(1.0, manager.state.stress_level + stress_adjustment))
    
    Object.update_state(manager, %{stress_level: new_stress})
  end
  
  defp update_inventory_manager_behavior(inventory_manager, state) do
    # Check for low stock and generate alerts
    low_stock_items = inventory_manager.state.inventory_levels
                     |> Enum.filter(fn {_, level} -> level.current <= level.reorder_point end)
                     |> Enum.map(fn {category, _} -> category end)
    
    if length(low_stock_items) > 0 do
      # Generate reorder
      pending_orders = low_stock_items
                      |> Enum.map(fn category ->
                        %{category: category, quantity: 50, order_time: DateTime.utc_now()}
                      end)
      
      Object.update_state(inventory_manager, %{pending_orders: pending_orders})
    else
      inventory_manager
    end
  end
  
  defp update_customer_service_behavior(cs_rep, state) do
    # Adjust energy based on workload
    customers_needing_help = state.active_customers
                           |> Map.values()
                           |> Enum.count(&(&1.current_activity == :needs_assistance))
    
    energy_drain = min(0.1, customers_needing_help * 0.02)
    new_energy = max(0.0, cs_rep.state.energy_level - energy_drain)
    
    Object.update_state(cs_rep, %{
      energy_level: new_energy,
      active_interactions: customers_needing_help
    })
  end
  
  defp update_sales_associate_behavior(sales_associate, state) do
    # Update customer queue
    customers_ready = state.active_customers
                     |> Map.values()
                     |> Enum.filter(&(&1.current_activity == :ready_to_buy))
                     |> length()
    
    Object.update_state(sales_associate, %{customer_queue: customers_ready})
  end
  
  defp update_security_behavior(security, _state) do
    # Random security checks
    if :rand.uniform() < 0.1 do
      # Random security incident check
      if :rand.uniform() < 0.05 do
        Object.update_state(security, %{alerts_today: security.state.alerts_today + 1})
      else
        security
      end
    else
      security
    end
  end
  
  defp process_inventory_management(state) do
    inventory_manager = Map.get(state.store_objects, "inventory_manager_002")
    
    # Process pending orders (simulate delivery)
    completed_orders = inventory_manager.state.pending_orders
                      |> Enum.filter(fn order ->
                        DateTime.diff(DateTime.utc_now(), order.order_time, :minute) > 30  # 30 min delivery
                      end)
    
    if length(completed_orders) > 0 do
      # Update inventory with delivered items
      updated_inventory = Enum.reduce(completed_orders, inventory_manager.state.inventory_levels, fn order, acc ->
        case Map.get(acc, order.category) do
          nil -> acc
          level_data ->
            new_current = min(level_data.max, level_data.current + order.quantity)
            updated_level = %{level_data | current: new_current}
            Map.put(acc, order.category, updated_level)
        end
      end)
      
      remaining_orders = inventory_manager.state.pending_orders -- completed_orders
      
      updated_manager = Object.update_state(inventory_manager, %{
        inventory_levels: updated_inventory,
        pending_orders: remaining_orders
      })
      
      updated_objects = Map.put(state.store_objects, "inventory_manager_002", updated_manager)
      
      IO.puts("📦 Inventory restocked: #{length(completed_orders)} deliveries")
      
      %{state | store_objects: updated_objects}
    else
      state
    end
  end
  
  defp update_daily_metrics(state) do
    # Calculate metrics from current state
    sales_associate = Map.get(state.store_objects, "sales_associate_004")
    customer_service = Map.get(state.store_objects, "customer_service_003")
    
    current_hour = DateTime.to_time(state.current_time).hour
    
    updated_metrics = %{state.daily_metrics |
      total_sales: sales_associate.state.total_sales,
      transaction_count: sales_associate.state.transactions_today,
      customer_count: Map.get(state.store_objects, "security_system_005").state.visitor_count,
      average_transaction_value: if sales_associate.state.transactions_today > 0 do
        sales_associate.state.total_sales / sales_associate.state.transactions_today
      else
        0
      end,
      customer_satisfaction: if length(customer_service.state.satisfaction_scores) > 0 do
        Enum.sum(customer_service.state.satisfaction_scores) / length(customer_service.state.satisfaction_scores)
      else
        state.daily_metrics.customer_satisfaction
      end,
      hour_by_hour_sales: Map.put(state.daily_metrics.hour_by_hour_sales, current_hour, sales_associate.state.total_sales)
    }
    
    %{state | daily_metrics: updated_metrics}
  end
  
  defp cleanup_old_data(state) do
    # Remove old events (keep last 100)
    cleaned_events = Enum.take(state.event_log, 100)
    
    # Remove old alerts (keep last 20)
    cleaned_alerts = Enum.take(state.alerts, 20)
    
    %{state |
      event_log: cleaned_events,
      alerts: cleaned_alerts
    }
  end
  
  defp calculate_performance_stats(state, tick_duration) do
    customers_per_hour = map_size(state.active_customers) * 60  # Extrapolate from current
    
    sales_associate = Map.get(state.store_objects, "sales_associate_004")
    avg_transaction = if sales_associate.state.transactions_today > 0 do
      sales_associate.state.total_sales / sales_associate.state.transactions_today
    else
      0
    end
    
    # Calculate staff efficiency from all objects
    staff_scores = state.store_objects
                  |> Map.values()
                  |> Enum.map(&(&1.goal.(&1.state)))
                  |> Enum.filter(&is_number/1)
    
    staff_efficiency = if length(staff_scores) > 0 do
      Enum.sum(staff_scores) / length(staff_scores)
    else
      0.8
    end
    
    %{
      customers_per_hour: customers_per_hour,
      average_transaction_value: avg_transaction,
      staff_efficiency: staff_efficiency,
      inventory_turnover: 0.1  # Simplified
    }
  end
  
  defp log_hourly_status(state) do
    hour = DateTime.to_time(state.current_time).hour
    
    IO.puts("\n🕐 Store Status - #{hour}:00")
    IO.puts("   👥 Active Customers: #{map_size(state.active_customers)}")
    IO.puts("   💰 Sales Today: $#{trunc(state.daily_metrics.total_sales)}")
    IO.puts("   📊 Transactions: #{state.daily_metrics.transaction_count}")
    IO.puts("   ⭐ Satisfaction: #{Float.round(state.daily_metrics.customer_satisfaction, 1)}/5.0")
    
    if length(state.alerts) > 0 do
      IO.puts("   🚨 Active Alerts: #{length(state.alerts)}")
    end
  end
  
  defp get_hour_sales(metrics, hour) do
    Map.get(metrics.hour_by_hour_sales, hour, 0)
  end
  
  defp handle_triggered_event(state, event_type, params) do
    case event_type do
      :rush_hour ->
        handle_rush_hour(state, params)
      
      :inventory_shortage ->
        handle_inventory_shortage(state, params)
      
      :difficult_customer ->
        handle_difficult_customer(state, params)
      
      :system_maintenance ->
        handle_system_maintenance(state, params)
      
      :staff_break ->
        handle_staff_break(state, params)
      
      _ ->
        {state, %{result: :unknown_event}}
    end
  end
  
  defp handle_rush_hour(state, _params) do
    # Spawn multiple customers rapidly
    new_customers = for i <- 1..5 do
      customer = create_random_customer(state.current_time)
      {customer.id, customer}
    end |> Map.new()
    
    updated_customers = Map.merge(state.active_customers, new_customers)
    
    IO.puts("🏃‍♂️ Rush hour! #{map_size(new_customers)} customers entered")
    
    {%{state | active_customers: updated_customers}, %{result: :rush_hour_started, customers_added: map_size(new_customers)}}
  end
  
  defp handle_inventory_shortage(state, params) do
    category = Map.get(params, :category, :electronics)
    
    inventory_manager = Map.get(state.store_objects, "inventory_manager_002")
    current_level = get_in(inventory_manager.state, [:inventory_levels, category, :current]) || 0
    
    # Reduce inventory to critical level
    updated_level = max(0, trunc(current_level * 0.3))
    
    updated_inventory = put_in(inventory_manager.state.inventory_levels[category].current, updated_level)
    updated_manager = %{inventory_manager | state: %{inventory_manager.state | inventory_levels: updated_inventory}}
    
    updated_objects = Map.put(state.store_objects, "inventory_manager_002", updated_manager)
    
    IO.puts("📉 Inventory shortage in #{category}")
    
    {%{state | store_objects: updated_objects}, %{result: :shortage_applied, category: category}}
  end
  
  defp handle_difficult_customer(state, _params) do
    difficult_customer = %{
      id: "difficult_customer_#{:rand.uniform(1000)}",
      type: :difficult_customer,
      entered_at: state.current_time,
      budget: 100,
      satisfaction: 0.3,
      items_wanted: [%{category: :electronics, item: "expensive_item", price: 2000}],
      patience: 0.2,
      current_activity: :needs_assistance,
      time_spent: 0,
      assistance_needed: true
    }
    
    updated_customers = Map.put(state.active_customers, difficult_customer.id, difficult_customer)
    
    IO.puts("😠 Difficult customer entered the store")
    
    {%{state | active_customers: updated_customers}, %{result: :difficult_customer_added}}
  end
  
  defp handle_system_maintenance(state, params) do
    duration = Map.get(params, :duration, 15)  # 15 minutes
    
    # Reduce all object efficiency temporarily
    updated_objects = state.store_objects
                     |> Enum.map(fn {id, object} ->
                       if object.subtype in [:sensor_object, :actuator_object] do
                         maintenance_state = Map.put(object.state, :maintenance_mode, true)
                         {id, %{object | state: maintenance_state}}
                       else
                         {id, object}
                       end
                     end)
                     |> Map.new()
    
    IO.puts("🔧 System maintenance started (#{duration} minutes)")
    
    {%{state | store_objects: updated_objects}, %{result: :maintenance_started, duration: duration}}
  end
  
  defp handle_staff_break(state, params) do
    staff_id = Map.get(params, :staff_id, "customer_service_003")
    
    case Map.get(state.store_objects, staff_id) do
      nil -> {state, %{result: :staff_not_found}}
      
      staff_object ->
        break_state = Map.put(staff_object.state, :on_break, true)
        updated_staff = %{staff_object | state: break_state}
        updated_objects = Map.put(state.store_objects, staff_id, updated_staff)
        
        IO.puts("☕ #{staff_id} is now on break")
        
        {%{state | store_objects: updated_objects}, %{result: :break_started, staff_id: staff_id}}
    end
  end
  
  defp log_event(state, event_type, result) do
    event = %{
      type: event_type,
      timestamp: DateTime.utc_now(),
      result: result
    }
    
    %{state | event_log: [event | state.event_log]}
  end
  
  defp schedule_tick do
    Process.send_after(self(), :tick, @tick_interval)
  end
end

# Interactive control module
defmodule RealtimeStoreManagement.Controller do
  @moduledoc """
  Interactive control interface for the realtime store
  """
  
  def start do
    IO.puts("\n🎮 Realtime Store Management Controller")
    IO.puts("=" |> String.duplicate(50))
    IO.puts("Commands:")
    IO.puts("  status        - Show store status")
    IO.puts("  metrics       - Show daily metrics")
    IO.puts("  customers     - List active customers")
    IO.puts("  objects       - List store objects")
    IO.puts("  object <id>   - Show object details")
    IO.puts("  alerts        - Show active alerts")
    IO.puts("  rush          - Trigger rush hour")
    IO.puts("  shortage      - Trigger inventory shortage")
    IO.puts("  difficult     - Add difficult customer")
    IO.puts("  maintenance   - Start system maintenance")
    IO.puts("  break <id>    - Send staff on break")
    IO.puts("  price <cat> <adj> - Adjust category pricing")
    IO.puts("  alert <type> <msg> - Create emergency alert")
    IO.puts("  stop          - Close store")
    IO.puts("  help          - Show this help")
    IO.puts("  quit          - Exit controller")
    IO.puts("")
    
    command_loop()
  end
  
  defp command_loop do
    input = IO.gets("🏪 > ") |> String.trim()
    
    case String.split(input, " ", parts: 3) do
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
        
      ["customers"] ->
        show_customers()
        command_loop()
        
      ["objects"] ->
        show_objects()
        command_loop()
        
      ["object", object_id] ->
        show_object(object_id)
        command_loop()
        
      ["alerts"] ->
        show_alerts()
        command_loop()
        
      ["rush"] ->
        RealtimeStoreManagement.trigger_event(:rush_hour)
        command_loop()
        
      ["shortage"] ->
        RealtimeStoreManagement.trigger_event(:inventory_shortage, %{category: :electronics})
        command_loop()
        
      ["difficult"] ->
        RealtimeStoreManagement.trigger_event(:difficult_customer)
        command_loop()
        
      ["maintenance"] ->
        RealtimeStoreManagement.trigger_event(:system_maintenance)
        command_loop()
        
      ["break", staff_id] ->
        RealtimeStoreManagement.trigger_event(:staff_break, %{staff_id: staff_id})
        command_loop()
        
      ["price", category, adjustment] ->
        {adj_num, _} = Float.parse(adjustment)
        cat_atom = String.to_atom(category)
        RealtimeStoreManagement.adjust_pricing(cat_atom, adj_num)
        command_loop()
        
      ["alert", alert_type, message] ->
        RealtimeStoreManagement.emergency_alert(String.to_atom(alert_type), message)
        command_loop()
        
      ["stop"] ->
        RealtimeStoreManagement.stop()
        IO.puts("Store closed.")
        command_loop()
        
      _ ->
        IO.puts("Unknown command. Type 'help' for available commands.")
        command_loop()
    end
  end
  
  defp show_status do
    case RealtimeStoreManagement.status() do
      status when is_map(status) ->
        store_status = if status.store_open, do: "OPEN", else: "CLOSED"
        
        IO.puts("\n🏪 Store Status: #{store_status}")
        IO.puts("   Uptime: #{status.uptime} seconds")
        IO.puts("   Active Customers: #{status.active_customers}")
        IO.puts("   Store Objects: #{status.store_objects}")
        IO.puts("   Active Alerts: #{status.alerts}")
        IO.puts("   Current Time: #{Calendar.strftime(status.current_time, "%H:%M")}")
        IO.puts("   Performance:")
        IO.puts("     Customers/hour: #{trunc(status.performance.customers_per_hour)}")
        IO.puts("     Avg Transaction: $#{Float.round(status.performance.average_transaction_value, 2)}")
        IO.puts("     Staff Efficiency: #{Float.round(status.performance.staff_efficiency * 100, 1)}%")
        
      error ->
        IO.puts("Error getting status: #{inspect(error)}")
    end
  end
  
  defp show_metrics do
    case RealtimeStoreManagement.metrics() do
      metrics when is_map(metrics) ->
        IO.puts("\n📊 Daily Metrics:")
        IO.puts("   Total Sales: $#{trunc(metrics.total_sales)}")
        IO.puts("   Transactions: #{metrics.transaction_count}")
        IO.puts("   Customers: #{metrics.customer_count}")
        IO.puts("   Avg Transaction: $#{Float.round(metrics.average_transaction_value, 2)}")
        IO.puts("   Customer Satisfaction: #{Float.round(metrics.customer_satisfaction, 1)}/5.0")
        IO.puts("   Inventory Alerts: #{metrics.inventory_alerts}")
        IO.puts("   Security Incidents: #{metrics.security_incidents}")
        
        if map_size(metrics.hour_by_hour_sales) > 0 do
          IO.puts("   Hourly Sales:")
          for {hour, sales} <- metrics.hour_by_hour_sales |> Enum.sort() do
            IO.puts("     #{hour}:00 - $#{trunc(sales)}")
          end
        end
        
      error ->
        IO.puts("Error getting metrics: #{inspect(error)}")
    end
  end
  
  defp show_customers do
    case RealtimeStoreManagement.get_customers() do
      customers when is_map(customers) ->
        IO.puts("\n👥 Active Customers (#{map_size(customers)}):")
        for {id, customer} <- customers do
          IO.puts("   #{id}: #{customer.type} - #{customer.current_activity}")
          IO.puts("     Budget: $#{customer.budget}, Satisfaction: #{Float.round(customer.satisfaction, 2)}")
          IO.puts("     Time in store: #{customer.time_spent} minutes")
        end
        
      error ->
        IO.puts("Error getting customers: #{inspect(error)}")
    end
  end
  
  defp show_objects do
    IO.puts("\n🏪 Store Objects:")
    objects = ["store_manager_001", "inventory_manager_002", "customer_service_003", "sales_associate_004", "security_system_005"]
    
    for object_id <- objects do
      case RealtimeStoreManagement.get_object(object_id) do
        nil -> IO.puts("   #{object_id}: Not found")
        object ->
          role = get_in(object.state, [:role])
          performance = object.goal.(object.state)
          IO.puts("   #{object_id}: #{role} - #{Float.round(performance * 100, 1)}% performance")
      end
    end
  end
  
  defp show_object(object_id) do
    case RealtimeStoreManagement.get_object(object_id) do
      nil ->
        IO.puts("Object '#{object_id}' not found")
        
      object ->
        IO.puts("\n🔧 Object: #{object_id}")
        IO.puts("   Role: #{get_in(object.state, [:role])}")
        IO.puts("   Subtype: #{object.subtype}")
        IO.puts("   Performance: #{Float.round(object.goal.(object.state) * 100, 1)}%")
        IO.puts("   State keys: #{Map.keys(object.state) |> Enum.join(", ")}")
        
        # Show relevant metrics
        case get_in(object.state, [:role]) do
          :sales_associate ->
            IO.puts("   Sales today: $#{get_in(object.state, [:total_sales]) || 0}")
            IO.puts("   Transactions: #{get_in(object.state, [:transactions_today]) || 0}")
            
          :inventory_manager ->
            levels = get_in(object.state, [:inventory_levels]) || %{}
            IO.puts("   Inventory levels:")
            for {category, level} <- levels do
              IO.puts("     #{category}: #{level.current}/#{level.max}")
            end
            
          :customer_service ->
            IO.puts("   Active interactions: #{get_in(object.state, [:active_interactions]) || 0}")
            IO.puts("   Energy level: #{Float.round(get_in(object.state, [:energy_level]) || 0, 2)}")
            
          _ ->
            nil
        end
    end
  end
  
  defp show_alerts do
    case RealtimeStoreManagement.get_alerts() do
      alerts when is_list(alerts) ->
        IO.puts("\n🚨 Active Alerts (#{length(alerts)}):")
        for alert <- alerts do
          timestamp = Calendar.strftime(alert.timestamp, "%H:%M:%S")
          IO.puts("   [#{timestamp}] #{alert.type}: #{alert.message}")
        end
        
      error ->
        IO.puts("Error getting alerts: #{inspect(error)}")
    end
  end
end

# Run the system
case System.argv() do
  ["start"] ->
    {:ok, _pid} = RealtimeStoreManagement.start_link()
    IO.puts("\n🎮 Starting interactive controller...")
    RealtimeStoreManagement.Controller.start()
    
  ["demo"] ->
    {:ok, _pid} = RealtimeStoreManagement.start_link()
    IO.puts("🕐 Running store demo for 60 seconds...")
    
    # Trigger events during demo
    spawn(fn ->
      :timer.sleep(10000)
      RealtimeStoreManagement.trigger_event(:rush_hour)
      
      :timer.sleep(15000)
      RealtimeStoreManagement.trigger_event(:inventory_shortage, %{category: :electronics})
      
      :timer.sleep(15000)
      RealtimeStoreManagement.trigger_event(:difficult_customer)
      
      :timer.sleep(10000)
      RealtimeStoreManagement.emergency_alert(:fire_drill, "Monthly fire drill")
    end)
    
    :timer.sleep(60000)
    RealtimeStoreManagement.stop()
    IO.puts("Store demo complete!")
    
  _ ->
    IO.puts("Usage:")
    IO.puts("  elixir examples/realtime_store_management.exs start   # Start interactive store")
    IO.puts("  elixir examples/realtime_store_management.exs demo    # Run 60-second demo")
end
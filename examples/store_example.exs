#!/usr/bin/env elixir

# Store Management Demo
# Shows AAOS objects managing a retail store with inventory, customers, and sales

defmodule StoreExample do
  @moduledoc """
  Demonstration of AAOS objects managing a retail store ecosystem.
  
  This demo shows:
  1. Inventory management objects tracking stock levels
  2. Customer service objects handling interactions
  3. Sales objects processing transactions
  4. Manager objects coordinating operations
  5. Collaborative problem solving for store optimization
  """
  
  def run_demo do
    IO.puts("🏪 Store Management Demo")
    IO.puts("=" |> String.duplicate(40))
    
    # Create store management objects
    store_objects = create_store_objects()
    
    # Simulate store operations
    IO.puts("\n🌅 Morning Store Opening")
    {updated_objects, morning_results} = simulate_morning_opening(store_objects)
    
    IO.puts("\n👤 Customer Service Scenario")
    {updated_objects, customer_results} = simulate_customer_service(updated_objects)
    
    IO.puts("\n📦 Inventory Management")
    {updated_objects, inventory_results} = simulate_inventory_management(updated_objects)
    
    IO.puts("\n💰 Sales Processing")
    {updated_objects, sales_results} = simulate_sales_processing(updated_objects)
    
    IO.puts("\n🎯 Store Optimization")
    {final_objects, optimization_results} = simulate_store_optimization(updated_objects)
    
    # Generate store performance report
    IO.puts("\n📊 Store Performance Report")
    generate_store_report(final_objects, [morning_results, customer_results, inventory_results, sales_results, optimization_results])
    
    IO.puts("\n✅ Store Demo Complete!")
  end
  
  defp create_store_objects do
    IO.puts("Creating store management objects...")
    
    # Store Manager - Overall coordination and decision making
    store_manager = Object.new(
      id: "store_manager_001",
      subtype: :coordinator_object,
      state: %{
        store_metrics: %{
          daily_sales_target: 10000,
          customer_satisfaction: 0.85,
          inventory_turnover: 0.75,
          employee_efficiency: 0.9
        },
        active_promotions: ["summer_sale", "loyalty_program"],
        current_challenges: [],
        management_style: "collaborative",
        decision_making_speed: 0.8
      },
      methods: [:coordinate, :analyze_performance, :make_decisions, :optimize_operations, :handle_issues],
      goal: fn state ->
        metrics = state.store_metrics
        (metrics.customer_satisfaction + metrics.inventory_turnover + metrics.employee_efficiency) / 3
      end
    )
    
    # Inventory Manager - Stock tracking and reordering
    inventory_manager = Object.new(
      id: "inventory_mgr_002", 
      subtype: :sensor_object,
      state: %{
        inventory_levels: %{
          "electronics" => %{current: 45, minimum: 20, maximum: 100},
          "clothing" => %{current: 78, minimum: 30, maximum: 150},
          "home_goods" => %{current: 23, minimum: 15, maximum: 80},
          "books" => %{current: 156, minimum: 50, maximum: 200}
        },
        reorder_alerts: [],
        supplier_relationships: %{
          "TechSupply Co" => %{reliability: 0.95, lead_time: 3},
          "Fashion Dist" => %{reliability: 0.88, lead_time: 5},
          "Home Plus" => %{reliability: 0.92, lead_time: 4}
        },
        accuracy_rate: 0.97,
        last_audit: DateTime.utc_now()
      },
      methods: [:track_inventory, :generate_reorder_alerts, :audit_stock, :forecast_demand, :optimize_storage],
      goal: fn state ->
        levels = state.inventory_levels
        low_stock_penalty = Enum.count(levels, fn {_, data} -> data.current < data.minimum end) * 0.1
        max(0.0, state.accuracy_rate - low_stock_penalty)
      end
    )
    
    # Customer Service Representative - Customer interaction and satisfaction
    customer_service = Object.new(
      id: "customer_svc_003",
      subtype: :ai_agent,
      state: %{
        active_customers: [],
        interaction_history: [],
        service_metrics: %{
          response_time: 2.3,
          resolution_rate: 0.89,
          satisfaction_score: 4.2,
          escalation_rate: 0.12
        },
        knowledge_base: %{
          product_info: "comprehensive",
          policies: "current",
          promotions: "active"
        },
        communication_style: "friendly_professional",
        language_skills: ["english", "spanish", "mandarin"]
      },
      methods: [:greet_customer, :answer_questions, :resolve_issues, :process_returns, :gather_feedback],
      goal: fn state ->
        metrics = state.service_metrics
        (metrics.resolution_rate + (5 - metrics.response_time) / 5 + metrics.satisfaction_score / 5) / 3
      end
    )
    
    # Sales Associate - Transaction processing and upselling
    sales_associate = Object.new(
      id: "sales_assoc_004",
      subtype: :actuator_object,
      state: %{
        daily_sales: 0,
        transactions_processed: 0,
        upsell_success_rate: 0.35,
        payment_methods: ["cash", "credit", "debit", "mobile"],
        current_promotions: %{
          "buy_2_get_1" => %{active: true, categories: ["books"]},
          "summer_discount" => %{active: true, discount: 0.15}
        },
        sales_techniques: ["consultative", "solution_based"],
        product_knowledge_level: 0.85
      },
      methods: [:process_transaction, :suggest_products, :apply_discounts, :handle_payments, :update_loyalty_points],
      goal: fn state ->
        sales_performance = min(1.0, state.daily_sales / 2000.0)
        upsell_performance = state.upsell_success_rate
        (sales_performance + upsell_performance) / 2
      end
    )
    
    # Security System - Loss prevention and safety monitoring
    security_system = Object.new(
      id: "security_sys_005",
      subtype: :sensor_object,
      state: %{
        camera_feeds: %{
          "entrance" => :active,
          "electronics" => :active,
          "checkout" => :active,
          "storage" => :active
        },
        alarm_status: :normal,
        incident_log: [],
        loss_prevention: %{
          shrinkage_rate: 0.02,
          suspicious_activities: 0,
          false_alarms: 0
        },
        visitor_count: 0,
        peak_hours: [10, 11, 14, 15, 16]
      },
      methods: [:monitor_premises, :detect_anomalies, :count_visitors, :generate_alerts, :secure_facility],
      goal: fn state ->
        loss_prevention = 1.0 - state.loss_prevention.shrinkage_rate
        reliability = 1.0 - (state.loss_prevention.false_alarms / 10.0)
        (loss_prevention + reliability) / 2
      end
    )
    
    IO.puts("✅ Created store management ecosystem with 5 specialized objects")
    
    [store_manager, inventory_manager, customer_service, sales_associate, security_system]
  end
  
  defp simulate_morning_opening(objects) do
    IO.puts("🌅 8:00 AM - Store opening procedures...")
    
    store_manager = Enum.find(objects, &(&1.id == "store_manager_001"))
    inventory_manager = Enum.find(objects, &(&1.id == "inventory_mgr_002"))
    security_system = Enum.find(objects, &(&1.id == "security_sys_005"))
    
    # Manager initiates morning opening checklist
    IO.puts("  📋 Store Manager initiating opening procedures...")
    
    opening_message = %{
      sender: store_manager.id,
      content: "Good morning team! Beginning store opening procedures. Security, please confirm facility status. Inventory, provide overnight stock report. Let's ensure everything is ready for customers.",
      timestamp: DateTime.utc_now(),
      message_type: :coordination_request,
      priority: :high
    }
    
    # Security system responds with facility status
    {:ok, security_response, updated_security} = simulate_store_response(security_system, opening_message,
      context: "Morning security check"
    )
    
    IO.puts("  🛡️  Security: \"#{security_response.content}\"")
    
    # Inventory provides overnight report
    {:ok, inventory_response, updated_inventory} = simulate_store_response(inventory_manager, opening_message,
      context: "Morning inventory status"
    )
    
    IO.puts("  📦 Inventory: \"#{inventory_response.content}\"")
    
    # Manager coordinates final opening tasks
    final_coordination = %{
      sender: store_manager.id,
      content: "Excellent reports. Security status green, inventory levels acceptable. Activating customer service systems, enabling payment processing, and unlocking entrance. Store ready for business!",
      timestamp: DateTime.utc_now(),
      message_type: :status_update,
      priority: :normal
    }
    
    # Update store manager with opening coordination
    updated_manager = Object.interact(store_manager, %{
      type: :opening_coordination,
      tasks_completed: ["security_check", "inventory_review", "system_activation"],
      opening_time: DateTime.utc_now(),
      readiness_status: "ready_for_business"
    })
    
    updated_objects = Enum.map(objects, fn object ->
      case object.id do
        "store_manager_001" -> updated_manager
        "inventory_mgr_002" -> updated_inventory
        "security_sys_005" -> updated_security
        _ -> object
      end
    end)
    
    results = %{
      scenario: "morning_opening",
      opening_time: "8:00 AM",
      coordination_success: true,
      security_status: "green",
      inventory_status: "acceptable",
      readiness: "ready_for_business"
    }
    
    IO.puts("  ✅ Store successfully opened - all systems operational")
    
    {updated_objects, results}
  end
  
  defp simulate_customer_service(objects) do
    IO.puts("👤 10:30 AM - Customer service scenario...")
    
    customer_service = Enum.find(objects, &(&1.id == "customer_svc_003"))
    store_manager = Enum.find(objects, &(&1.id == "store_manager_001"))
    inventory_manager = Enum.find(objects, &(&1.id == "inventory_mgr_002"))
    
    # Customer arrives with a question about product availability
    IO.puts("  🛍️  Customer inquiry about product availability...")
    
    customer_inquiry = %{
      sender: "customer_sarah",
      content: "Hi, I'm looking for a specific wireless headphone model - the TechSound Pro X1. Do you have it in stock? Also, are there any current promotions I should know about?",
      timestamp: DateTime.utc_now(),
      message_type: :customer_inquiry,
      priority: :normal,
      customer_data: %{
        customer_id: "sarah_001",
        loyalty_member: true,
        previous_purchases: ["electronics", "accessories"]
      }
    }
    
    # Customer service responds and coordinates with inventory
    {:ok, service_response, updated_service} = simulate_store_response(customer_service, customer_inquiry,
      context: "Product availability inquiry"
    )
    
    IO.puts("  👨‍💼 Customer Service: \"#{service_response.content}\"")
    
    # Customer service requests inventory check
    inventory_check_request = %{
      sender: customer_service.id,
      content: "Inventory check needed for TechSound Pro X1 wireless headphones. Customer is loyalty member with electronics purchase history. Please provide availability and any related product suggestions.",
      timestamp: DateTime.utc_now(),
      message_type: :inventory_check,
      priority: :normal,
      product_code: "TSP-X1-WH"
    }
    
    {:ok, inventory_check, updated_inventory} = simulate_store_response(inventory_manager, inventory_check_request,
      context: "Product availability check"
    )
    
    IO.puts("  📦 Inventory: \"#{inventory_check.content}\"")
    
    # Customer service provides comprehensive response to customer
    final_customer_response = %{
      sender: customer_service.id,
      content: "Great news Sarah! We have the TechSound Pro X1 in stock - 3 units available. As a loyalty member, you qualify for our 15% summer discount. I can also show you the new TechSound Pro X2 model that just arrived. Would you like to see both options?",
      timestamp: DateTime.utc_now(),
      message_type: :customer_response,
      product_recommendations: ["TSP-X1-WH", "TSP-X2-WH"],
      discounts_applied: ["loyalty_15_percent", "summer_promotion"]
    }
    
    IO.puts("  👨‍💼 Customer Service (to customer): \"#{final_customer_response.content}\"")
    
    # Customer service escalates positive interaction to manager
    escalation_message = %{
      sender: customer_service.id,
      content: "Manager, excellent customer interaction with loyalty member Sarah. Successfully provided product info, applied appropriate discounts, and suggested upgrades. Customer appears satisfied and likely to purchase.",
      timestamp: DateTime.utc_now(),
      message_type: :positive_escalation,
      priority: :low
    }
    
    {:ok, manager_response, updated_manager} = simulate_store_response(store_manager, escalation_message,
      context: "Positive customer interaction"
    )
    
    IO.puts("  👔 Store Manager: \"#{manager_response.content}\"")
    
    # Update objects with customer service interaction
    service_interaction = %{
      type: :customer_service_interaction,
      customer_id: "sarah_001",
      inquiry_type: "product_availability",
      resolution_time: 3.5,
      satisfaction_score: 4.8,
      upsell_opportunity: true,
      discount_applied: true
    }
    
    updated_objects = Enum.map(objects, fn object ->
      case object.id do
        "customer_svc_003" -> Object.interact(updated_service, service_interaction)
        "store_manager_001" -> updated_manager
        "inventory_mgr_002" -> updated_inventory
        _ -> object
      end
    end)
    
    results = %{
      scenario: "customer_service",
      customer_type: "loyalty_member",
      inquiry_resolution_time: 3.5,
      satisfaction_score: 4.8,
      upsell_presented: true,
      coordination_quality: "excellent"
    }
    
    IO.puts("  ✅ Customer service interaction successful - high satisfaction achieved")
    
    {updated_objects, results}
  end
  
  defp simulate_inventory_management(objects) do
    IO.puts("📦 2:00 PM - Inventory management and reordering...")
    
    inventory_manager = Enum.find(objects, &(&1.id == "inventory_mgr_002"))
    store_manager = Enum.find(objects, &(&1.id == "store_manager_001"))
    
    # Inventory system detects low stock levels
    IO.puts("  📊 Automated inventory analysis in progress...")
    
    # Simulate inventory system detecting low stock
    low_stock_alert = %{
      sender: "inventory_system",
      content: "LOW STOCK ALERT: Home goods category at 23 units (minimum: 15). Electronics category decreasing rapidly due to summer promotion. Recommend immediate reorder for both categories.",
      timestamp: DateTime.utc_now(),
      message_type: :low_stock_alert,
      priority: :high,
      affected_categories: ["home_goods", "electronics"],
      reorder_urgency: "high"
    }
    
    {:ok, inventory_response, updated_inventory} = simulate_store_response(inventory_manager, low_stock_alert,
      context: "Low stock management"
    )
    
    IO.puts("  📦 Inventory Manager: \"#{inventory_response.content}\"")
    
    # Inventory manager escalates to store manager for approval
    reorder_request = %{
      sender: inventory_manager.id,
      content: "Manager, requesting approval for emergency reorder: Home goods ($3,500, 2-day delivery), Electronics restocking ($8,200, next-day delivery). Total investment $11,700 to prevent stockouts during peak sales period.",
      timestamp: DateTime.utc_now(),
      message_type: :reorder_approval_request,
      priority: :high,
      financial_impact: 11700,
      suppliers: ["Home Plus", "TechSupply Co"]
    }
    
    {:ok, manager_approval, updated_manager} = simulate_store_response(store_manager, reorder_request,
      context: "Reorder approval decision"
    )
    
    IO.puts("  👔 Store Manager: \"#{manager_approval.content}\"")
    
    # Inventory executes approved reorders
    IO.puts("  📋 Executing approved reorders...")
    
    reorder_execution = %{
      "home_goods_reorder" => %{
        supplier: "Home Plus",
        amount: 3500,
        delivery_date: Date.add(Date.utc_today(), 2),
        items: 45,
        status: "confirmed"
      },
      "electronics_reorder" => %{
        supplier: "TechSupply Co", 
        amount: 8200,
        delivery_date: Date.add(Date.utc_today(), 1),
        items: 35,
        status: "confirmed"
      }
    }
    
    IO.puts("  ✅ Home goods reorder: $3,500, 45 items, 2-day delivery")
    IO.puts("  ✅ Electronics reorder: $8,200, 35 items, next-day delivery")
    
    # Update inventory predictions
    demand_forecast = %{
      "electronics" => %{trend: "increasing", confidence: 0.92, seasonal_factor: 1.3},
      "home_goods" => %{trend: "stable", confidence: 0.85, seasonal_factor: 1.1},
      "clothing" => %{trend: "decreasing", confidence: 0.78, seasonal_factor: 0.9},
      "books" => %{trend: "stable", confidence: 0.88, seasonal_factor: 1.0}
    }
    
    IO.puts("  📈 Updated demand forecasts based on current trends")
    
    inventory_management_record = %{
      type: :inventory_management,
      alerts_processed: 2,
      reorders_executed: 2,
      total_investment: 11700,
      prevention_achieved: "stockout_prevention",
      forecast_updated: true
    }
    
    updated_objects = Enum.map(objects, fn object ->
      case object.id do
        "inventory_mgr_002" -> Object.interact(updated_inventory, inventory_management_record)
        "store_manager_001" -> updated_manager
        _ -> object
      end
    end)
    
    results = %{
      scenario: "inventory_management",
      alerts_triggered: 2,
      reorders_approved: 2,
      investment_amount: 11700,
      stockout_prevention: true,
      manager_approval_time: "immediate"
    }
    
    IO.puts("  ✅ Inventory management complete - stockouts prevented")
    
    {updated_objects, results}
  end
  
  defp simulate_sales_processing(objects) do
    IO.puts("💰 4:00 PM - Peak sales processing period...")
    
    sales_associate = Enum.find(objects, &(&1.id == "sales_assoc_004"))
    customer_service = Enum.find(objects, &(&1.id == "customer_svc_003"))
    store_manager = Enum.find(objects, &(&1.id == "store_manager_001"))
    
    # Sales associate handles multiple transactions
    IO.puts("  💳 Processing multiple customer transactions...")
    
    # Transaction 1: Regular purchase with upsell opportunity
    transaction_1 = %{
      customer_id: "mike_002",
      items: [
        %{product: "TechSound Pro X1", price: 199.99, quantity: 1},
        %{product: "Phone Case", price: 24.99, quantity: 1}
      ],
      payment_method: "credit_card",
      suggested_upsell: "TechSound Pro X2 (newer model)"
    }
    
    # Transaction 2: Loyalty member with promotion
    transaction_2 = %{
      customer_id: "sarah_001",
      items: [
        %{product: "TechSound Pro X2", price: 249.99, quantity: 1},
        %{product: "Wireless Charger", price: 39.99, quantity: 1}
      ],
      payment_method: "mobile_pay",
      loyalty_discount: 0.15,
      promotion_applied: "summer_discount"
    }
    
    # Sales associate processes Transaction 1
    IO.puts("  🛒 Transaction 1: Mike purchasing TechSound Pro X1...")
    
    sales_message_1 = %{
      sender: "customer_mike",
      content: "I'd like to purchase the TechSound Pro X1 and this phone case. Are there any warranties or accessories I should consider?",
      timestamp: DateTime.utc_now(),
      message_type: :purchase_request,
      transaction_data: transaction_1
    }
    
    {:ok, sales_response_1, updated_sales} = simulate_store_response(sales_associate, sales_message_1,
      context: "Transaction processing with upsell"
    )
    
    IO.puts("  💼 Sales Associate: \"#{sales_response_1.content}\"")
    
    # Sales associate processes Transaction 2  
    IO.puts("  🛒 Transaction 2: Sarah (loyalty member) purchasing upgraded model...")
    
    sales_message_2 = %{
      sender: "customer_sarah",
      content: "I'll take the TechSound Pro X2 and the wireless charger. Please apply my loyalty discount and any current promotions.",
      timestamp: DateTime.utc_now(),
      message_type: :purchase_request,
      transaction_data: transaction_2
    }
    
    {:ok, sales_response_2, updated_sales_2} = simulate_store_response(updated_sales, sales_message_2,
      context: "Loyalty transaction with promotions"
    )
    
    IO.puts("  💼 Sales Associate: \"#{sales_response_2.content}\"")
    
    # Sales associate reports to manager on successful upselling
    sales_report = %{
      sender: sales_associate.id,
      content: "Manager, excellent sales period! Successfully upsold Mike to warranty package (+$49), processed Sarah's loyalty purchase with X2 upgrade. Daily sales target 65% achieved by 4 PM. Strong momentum continuing.",
      timestamp: DateTime.utc_now(),
      message_type: :sales_report,
      priority: :normal,
      performance_metrics: %{
        transactions_completed: 2,
        upsell_success: 1,
        loyalty_transactions: 1,
        daily_progress: 0.65
      }
    }
    
    {:ok, manager_sales_response, updated_manager} = simulate_store_response(store_manager, sales_report,
      context: "Sales performance review"
    )
    
    IO.puts("  👔 Store Manager: \"#{manager_sales_response.content}\"")
    
    # Customer service follows up on transactions
    follow_up_message = %{
      sender: customer_service.id,
      content: "Following up on recent transactions. Mike's warranty activated successfully, Sarah's loyalty points updated (+50 points). Both customers received satisfaction surveys. Maintaining high service standards.",
      timestamp: DateTime.utc_now(),
      message_type: :post_sale_follow_up,
      priority: :low
    }
    
    IO.puts("  👨‍💼 Customer Service: \"#{follow_up_message.content}\"")
    
    # Calculate transaction results
    transaction_results = %{
      transaction_1: %{
        total: 274.97,  # Original + warranty upsell
        upsell_value: 49.00,
        customer_satisfaction: 4.5
      },
      transaction_2: %{
        total: 246.49,  # With 15% loyalty discount
        discount_applied: 43.50,
        loyalty_points_earned: 50,
        customer_satisfaction: 4.9
      }
    }
    
    sales_performance_record = %{
      type: :sales_performance,
      transactions_processed: 2,
      total_revenue: 521.46,
      upsell_success_rate: 0.5,
      loyalty_engagement: true,
      customer_satisfaction_avg: 4.7
    }
    
    updated_objects = Enum.map(objects, fn object ->
      case object.id do
        "sales_assoc_004" -> Object.interact(updated_sales_2, sales_performance_record)
        "store_manager_001" -> updated_manager
        _ -> object
      end
    end)
    
    results = %{
      scenario: "sales_processing",
      transactions_completed: 2,
      total_revenue: 521.46,
      upsell_success: true,
      loyalty_engagement: true,
      customer_satisfaction: 4.7
    }
    
    IO.puts("  ✅ Sales processing successful - strong revenue and satisfaction")
    
    {updated_objects, results}
  end
  
  defp simulate_store_optimization(objects) do
    IO.puts("🎯 6:00 PM - End-of-day store optimization analysis...")
    
    store_manager = Enum.find(objects, &(&1.id == "store_manager_001"))
    inventory_manager = Enum.find(objects, &(&1.id == "inventory_mgr_002"))
    sales_associate = Enum.find(objects, &(&1.id == "sales_assoc_004"))
    customer_service = Enum.find(objects, &(&1.id == "customer_svc_003"))
    security_system = Enum.find(objects, &(&1.id == "security_sys_005"))
    
    # Store manager initiates optimization analysis
    IO.puts("  📊 Conducting comprehensive store performance analysis...")
    
    optimization_request = %{
      sender: store_manager.id,
      content: "Team, let's analyze today's performance for optimization opportunities. Please provide key metrics and improvement suggestions for tomorrow's operations.",
      timestamp: DateTime.utc_now(),
      message_type: :optimization_analysis,
      priority: :normal
    }
    
    # Each system provides optimization insights
    {:ok, inventory_insights, _} = simulate_store_response(inventory_manager, optimization_request,
      context: "Inventory optimization analysis"
    )
    
    {:ok, sales_insights, _} = simulate_store_response(sales_associate, optimization_request,
      context: "Sales performance optimization"
    )
    
    {:ok, service_insights, _} = simulate_store_response(customer_service, optimization_request,
      context: "Customer service optimization"
    )
    
    {:ok, security_insights, _} = simulate_store_response(security_system, optimization_request,
      context: "Security and traffic optimization"
    )
    
    IO.puts("  📦 Inventory: \"#{inventory_insights.content}\"")
    IO.puts("  💼 Sales: \"#{sales_insights.content}\"")
    IO.puts("  👨‍💼 Service: \"#{service_insights.content}\"")
    IO.puts("  🛡️  Security: \"#{security_insights.content}\"")
    
    # Store manager synthesizes optimization plan
    IO.puts("  🧠 Manager synthesizing optimization strategy...")
    
    optimization_strategy = %{
      inventory_optimizations: [
        "Increase electronics buffer stock by 20%",
        "Implement predictive reordering for seasonal items",
        "Negotiate faster delivery terms with TechSupply Co"
      ],
      sales_optimizations: [
        "Enhance upselling training for warranty products",
        "Create bundled promotions for complementary items",
        "Implement dynamic pricing based on demand patterns"
      ],
      service_optimizations: [
        "Deploy AI chatbot for basic inquiries",
        "Create specialized consultation areas for electronics",
        "Implement customer feedback loop automation"
      ],
      security_optimizations: [
        "Adjust camera coverage for high-traffic areas",
        "Implement predictive crowd management",
        "Enhance loss prevention during peak hours"
      ]
    }
    
    IO.puts("  📋 Optimization Strategy Developed:")
    IO.puts("    • Inventory: Increase buffer stock, predictive reordering")
    IO.puts("    • Sales: Enhanced training, bundle promotions, dynamic pricing")
    IO.puts("    • Service: AI support, specialized areas, automated feedback")
    IO.puts("    • Security: Improved coverage, crowd management, peak hour focus")
    
    # Implement learning from today's operations
    IO.puts("  📚 Capturing operational learnings...")
    
    daily_learnings = %{
      successful_strategies: [
        "Proactive inventory management prevented stockouts",
        "Upselling techniques increased transaction value",
        "Loyalty program drove repeat purchases",
        "Collaborative coordination improved efficiency"
      ],
      improvement_areas: [
        "Faster response to inventory alerts needed",
        "More aggressive promotion of warranty products",
        "Enhanced customer education on loyalty benefits",
        "Better integration between service and sales"
      ],
      performance_metrics: %{
        daily_sales: 8500,
        customer_satisfaction: 4.6,
        inventory_efficiency: 0.92,
        security_incidents: 0
      }
    }
    
    # Update all objects with optimization learnings
    optimization_record = %{
      type: :daily_optimization,
      performance_analysis: daily_learnings,
      strategy_updates: optimization_strategy,
      learning_captured: true,
      next_day_preparations: true
    }
    
    updated_objects = Enum.map(objects, fn object ->
      Object.interact(object, optimization_record)
    end)
    
    results = %{
      scenario: "store_optimization",
      daily_sales: 8500,
      optimization_areas: 4,
      learning_points: 8,
      strategy_improvements: length(optimization_strategy.inventory_optimizations) + 
                           length(optimization_strategy.sales_optimizations) + 
                           length(optimization_strategy.service_optimizations) +
                           length(optimization_strategy.security_optimizations),
      readiness_next_day: true
    }
    
    IO.puts("  ✅ Store optimization complete - ready for enhanced operations tomorrow")
    
    {updated_objects, results}
  end
  
  defp generate_store_report(objects, scenario_results) do
    IO.puts("=" |> String.duplicate(50))
    IO.puts("📊 STORE PERFORMANCE REPORT")
    IO.puts("-" |> String.duplicate(30))
    
    # Aggregate results
    total_revenue = Enum.reduce(scenario_results, 0, fn result, acc ->
      acc + Map.get(result, :total_revenue, 0)
    end)
    
    customer_satisfaction = Enum.reduce(scenario_results, 0, fn result, acc ->
      acc + Map.get(result, :customer_satisfaction, 0)
    end) / length(scenario_results)
    
    IO.puts("Daily Revenue: $#{Float.round(total_revenue, 2)}")
    IO.puts("Customer Satisfaction: #{Float.round(customer_satisfaction, 1)}/5.0")
    IO.puts("Scenarios Completed: #{length(scenario_results)}")
    
    # Individual object performance
    IO.puts("\n🏪 Department Performance:")
    for object <- objects do
      performance_score = object.goal.(object.state)
      IO.puts("  #{String.replace(object.id, "_", " ") |> String.capitalize()}:")
      IO.puts("    Performance Score: #{Float.round(performance_score * 100, 1)}%")
      IO.puts("    Object Type: #{object.subtype}")
      
      # Department-specific metrics
      case object.subtype do
        :coordinator_object ->
          IO.puts("    Coordination Efficiency: High")
        :sensor_object ->
          accuracy = Map.get(object.state, :accuracy_rate, 0.95)
          IO.puts("    Data Accuracy: #{Float.round(accuracy * 100, 1)}%")
        :ai_agent ->
          resolution_rate = get_in(object.state, [:service_metrics, :resolution_rate]) || 0.89
          IO.puts("    Issue Resolution: #{Float.round(resolution_rate * 100, 1)}%")
        :actuator_object ->
          upsell_rate = Map.get(object.state, :upsell_success_rate, 0.35)
          IO.puts("    Upsell Success: #{Float.round(upsell_rate * 100, 1)}%")
        _ ->
          IO.puts("    Status: Operational")
      end
    end
    
    # Collaboration analysis
    IO.puts("\n🤝 Collaboration Insights:")
    IO.puts("  Cross-Department Coordination: Excellent")
    IO.puts("  Information Sharing: Real-time and accurate")
    IO.puts("  Problem Resolution Speed: Immediate")
    IO.puts("  Decision Making: Collaborative and data-driven")
    
    # Key achievements
    IO.puts("\n🎯 Key Achievements:")
    IO.puts("  • Prevented stockouts through proactive inventory management")
    IO.puts("  • Achieved high customer satisfaction through coordinated service")
    IO.puts("  • Implemented successful upselling strategies")
    IO.puts("  • Maintained zero security incidents")
    IO.puts("  • Developed comprehensive optimization strategy")
    
    # Recommendations
    IO.puts("\n💡 Optimization Opportunities:")
    IO.puts("  • Implement predictive analytics for demand forecasting")
    IO.puts("  • Enhance cross-training between departments")
    IO.puts("  • Deploy automated customer feedback systems")
    IO.puts("  • Create dynamic pricing algorithms")
    IO.puts("  • Develop advanced loyalty program features")
    
    IO.puts("=" |> String.duplicate(50))
  end
  
  # Helper functions for store-specific responses
  
  defp simulate_store_response(object, message, opts \\ []) do
    context = Keyword.get(opts, :context, "general_store_operation")
    
    response_content = case {object.subtype, context} do
      {:coordinator_object, "Morning security check"} ->
        "All systems green. Facility secure, cameras operational, no overnight incidents detected. Store ready for opening. Team coordination systems activated."
      
      {:sensor_object, "Morning inventory status"} ->
        "Overnight inventory audit complete. Current stock levels: Electronics 45 units, Clothing 78 units, Home goods 23 units (approaching minimum), Books 156 units. Recommend monitoring home goods category today."
      
      {:ai_agent, "Product availability inquiry"} ->
        "Hello! I'll be happy to help you find the TechSound Pro X1. Let me check our current inventory and any applicable promotions for loyalty members like yourself."
      
      {:sensor_object, "Product availability check"} ->
        "TechSound Pro X1: 3 units in stock, aisle 4 electronics section. Related items available: TechSound Pro X2 (newer model), wireless charging accessories, extended warranties. Customer's purchase history suggests high-value electronics preference."
      
      {:coordinator_object, "Positive customer interaction"} ->
        "Excellent work! Customer satisfaction is our priority. I notice Sarah is a valued loyalty member - ensure she's aware of our exclusive member events. This kind of service builds long-term relationships."
      
      {:sensor_object, "Low stock management"} ->
        "Confirmed low stock alert. Home goods at critical level (23/15 minimum), electronics depleting due to summer promotion success. Immediate reordering required to prevent stockouts. I have supplier contacts ready."
      
      {:coordinator_object, "Reorder approval decision"} ->
        "Approved. The investment is justified - stockouts would cost more in lost sales and customer dissatisfaction. Priority delivery authorized. Excellent proactive management."
      
      {:actuator_object, "Transaction processing with upsell"} ->
        "Excellent choice on the TechSound Pro X1! I'd recommend our premium warranty - it covers accidental damage and extends coverage to 3 years. With your phone case, that's great protection. Total with warranty: $274.97. Shall I add that?"
      
      {:actuator_object, "Loyalty transaction with promotions"} ->
        "Perfect choice upgrading to the X2 model, Sarah! Your loyalty discount gives you 15% off, plus the wireless charger complements perfectly. Total: $246.49, and you'll earn 50 loyalty points. Payment ready when you are!"
      
      {:coordinator_object, "Sales performance review"} ->
        "Outstanding performance! The upselling success and loyalty engagement show excellent customer relationship skills. We're well positioned to exceed today's targets. Keep up the excellent work team!"
      
      {:sensor_object, "Inventory optimization analysis"} ->
        "Analysis complete: Electronics buffer should increase 20% due to promotion success. Implement predictive reordering for seasonal patterns. Negotiate faster delivery with TechSupply Co for peak demand responsiveness."
      
      {:actuator_object, "Sales performance optimization"} ->
        "Today's insights: Warranty upselling highly successful (50% acceptance), bundle opportunities with accessories underutilized. Recommend enhanced product knowledge training and dynamic promotional displays."
      
      {:ai_agent, "Customer service optimization"} ->
        "Service metrics strong today - 4.6/5 satisfaction average. Opportunities: Deploy AI chatbot for basic FAQs, create specialized electronics consultation zone, automate post-purchase satisfaction tracking."
      
      {:sensor_object, "Security and traffic optimization"} ->
        "Traffic analysis: Peak periods 10-11 AM and 2-4 PM. Zero security incidents today. Recommend enhanced camera coverage in electronics during promotions, predictive crowd management for weekend sales."
      
      _ ->
        "Processing #{context} request. Analyzing current state and providing appropriate response based on store operations requirements."
    end
    
    response = %{
      content: response_content,
      tone: determine_store_tone(object, message),
      confidence: 0.9,
      timestamp: DateTime.utc_now(),
      context: context
    }
    
    # Update object with interaction
    updated_object = Object.interact(object, %{
      type: :store_operation,
      context: context,
      message: message,
      response: response,
      success: true
    })
    
    {:ok, response, updated_object}
  end
  
  defp determine_store_tone(object, message) do
    priority = Map.get(message, :priority, :normal)
    case {object.subtype, priority} do
      {_, :high} -> "urgent"
      {:coordinator_object, _} -> "authoritative"
      {:ai_agent, _} -> "helpful"
      {:sensor_object, _} -> "precise"
      {:actuator_object, _} -> "professional"
      _ -> "business_professional"
    end
  end
end

# Run the store demo
IO.puts("🏪 Starting Store Management Demo...")
StoreExample.run_demo()
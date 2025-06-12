#!/usr/bin/env elixir

# Dynamic Agent Civilization Demo
# Shows AAOS objects forming complex social structures, cultures, and civilizations


defmodule DynamicAgentCivilization do
  @moduledoc """
  Demonstration of AAOS objects forming a dynamic agent civilization.
  
  This demo shows:
  1. Agents with different roles and personalities forming communities
  2. Development of social structures and hierarchies  
  3. Cultural evolution through interaction and learning
  4. Resource management and trade between agent groups
  5. Collective problem solving and civilization advancement
  6. Emergence of governance and decision-making systems
  """
  
  def run_demo do
    IO.puts("🏛️ Dynamic Agent Civilization Demo")
    IO.puts("=" |> String.duplicate(50))
    
    # Initialize the civilization
    IO.puts("🌍 Initializing Agent Civilization...")
    agents = create_initial_population()
    
    # Phase 1: Settlement Formation
    IO.puts("\n🏘️ Phase 1: Settlement Formation")
    {agents, settlements} = simulate_settlement_formation(agents)
    
    # Phase 2: Social Structure Development
    IO.puts("\n👥 Phase 2: Social Structure Development")
    {agents, social_structures} = simulate_social_development(agents, settlements)
    
    # Phase 3: Cultural Evolution
    IO.puts("\n🎭 Phase 3: Cultural Evolution")
    {agents, cultures} = simulate_cultural_evolution(agents, social_structures)
    
    # Phase 4: Resource Management and Trade
    IO.puts("\n💰 Phase 4: Resource Management and Trade")
    {agents, economy} = simulate_economic_development(agents, settlements)
    
    # Phase 5: Governance and Collective Decision Making
    IO.puts("\n🏛️ Phase 5: Governance Development")
    {agents, governance} = simulate_governance_emergence(agents, social_structures)
    
    # Phase 6: Technological and Social Advancement
    IO.puts("\n🚀 Phase 6: Civilization Advancement")
    {final_agents, advancements} = simulate_civilization_advancement(agents, [settlements, cultures, economy, governance])
    
    # Generate civilization report
    IO.puts("\n📊 Civilization Analysis Report")
    generate_civilization_report(final_agents, settlements, social_structures, cultures, economy, governance, advancements)
    
    IO.puts("\n✅ Dynamic Agent Civilization Demo Complete!")
  end
  
  defp create_initial_population do
    IO.puts("Creating diverse agent population with different roles and personalities...")
    
    # Leaders - Strategic thinking and coordination
    leader_alice = Object.new(
      id: "leader_alice", 
      subtype: :coordinator_object,
      state: %{
        role: :leader,
        personality: %{charisma: 0.9, decisiveness: 0.85, vision: 0.8},
        skills: [:strategy, :diplomacy, :coordination, :resource_allocation],
        resources: %{influence: 100, knowledge: 80, network_size: 25},
        relationships: %{},
        settlement_preference: :central_location,
        leadership_style: :collaborative,
        ambitions: [:unify_settlements, :advance_technology, :establish_trade]
      },
      methods: [:lead, :coordinate, :negotiate, :strategize, :inspire],
      goal: fn state -> 
        influence = state.resources.influence / 100.0
        network = min(1.0, state.resources.network_size / 50.0)
        (influence + network) / 2
      end
    )
    
    # Scholars - Knowledge creation and preservation
    scholar_bob = Object.new(
      id: "scholar_bob",
      subtype: :ai_agent,
      state: %{
        role: :scholar,
        personality: %{curiosity: 0.95, patience: 0.9, creativity: 0.8},
        skills: [:research, :teaching, :analysis, :innovation, :documentation],
        resources: %{knowledge: 150, books: 50, students: 10},
        specialization: :natural_philosophy,
        research_projects: [:agriculture_optimization, :tool_improvement, :social_systems],
        academic_network: [],
        discoveries: []
      },
      methods: [:research, :teach, :analyze, :innovate, :document],
      goal: fn state ->
        knowledge_score = min(1.0, state.resources.knowledge / 200.0)
        impact_score = min(1.0, length(state.discoveries) / 10.0)
        (knowledge_score + impact_score) / 2
      end
    )
    
    # Artisans - Resource production and crafting
    artisan_charlie = Object.new(
      id: "artisan_charlie",
      subtype: :actuator_object,
      state: %{
        role: :artisan,
        personality: %{precision: 0.9, creativity: 0.85, persistence: 0.9},
        skills: [:crafting, :resource_processing, :tool_making, :construction, :quality_control],
        resources: %{materials: 100, tools: 75, finished_goods: 50},
        specialization: :metalworking,
        workshop_level: 1,
        apprentices: [],
        reputation: 0.7
      },
      methods: [:craft, :process_resources, :build, :teach_skills, :innovate_techniques],
      goal: fn state ->
        productivity = min(1.0, state.resources.finished_goods / 100.0)
        skill_level = state.workshop_level / 5.0
        (productivity + skill_level + state.reputation) / 3
      end
    )
    
    # Traders - Commerce and resource distribution
    trader_diana = Object.new(
      id: "trader_diana",
      subtype: :coordinator_object,
      state: %{
        role: :trader,
        personality: %{negotiation: 0.9, risk_tolerance: 0.7, social_intelligence: 0.85},
        skills: [:negotiation, :market_analysis, :logistics, :relationship_building, :risk_assessment],
        resources: %{currency: 200, goods_inventory: 80, trade_routes: 3},
        trading_network: [],
        market_knowledge: %{},
        successful_deals: 0,
        reputation: 0.6
      },
      methods: [:trade, :negotiate, :analyze_markets, :establish_routes, :build_relationships],
      goal: fn state ->
        wealth = min(1.0, state.resources.currency / 500.0)
        network = min(1.0, length(state.trading_network) / 20.0)
        (wealth + network + state.reputation) / 3
      end
    )
    
    # Farmers - Food production and sustainability
    farmer_eve = Object.new(
      id: "farmer_eve",
      subtype: :sensor_object,
      state: %{
        role: :farmer,
        personality: %{patience: 0.95, observation: 0.9, practicality: 0.85},
        skills: [:agriculture, :animal_husbandry, :weather_prediction, :resource_conservation, :planning],
        resources: %{land: 50, crops: 100, livestock: 25, tools: 40},
        seasonal_knowledge: %{},
        harvest_cycles: 0,
        farming_innovations: [],
        community_contributions: 0
      },
      methods: [:cultivate, :observe_patterns, :predict_weather, :conserve_resources, :plan_seasons],
      goal: fn state ->
        productivity = min(1.0, state.resources.crops / 150.0)
        sustainability = min(1.0, state.community_contributions / 50.0)
        (productivity + sustainability) / 2
      end
    )
    
    # Guards - Security and protection
    guard_frank = Object.new(
      id: "guard_frank",
      subtype: :sensor_object,
      state: %{
        role: :guard,
        personality: %{vigilance: 0.95, loyalty: 0.9, courage: 0.85},
        skills: [:security, :threat_assessment, :protection, :patrol, :emergency_response],
        resources: %{equipment: 60, patrol_area: 30, alert_network: 15},
        security_incidents: [],
        community_safety_score: 0.8,
        training_level: 1,
        alliances: []
      },
      methods: [:patrol, :assess_threats, :protect, :coordinate_security, :train],
      goal: fn state ->
        safety_score = state.community_safety_score
        preparedness = min(1.0, state.training_level / 3.0)
        (safety_score + preparedness) / 2
      end
    )
    
    # Healers - Healthcare and community wellbeing
    healer_grace = Object.new(
      id: "healer_grace",
      subtype: :ai_agent,
      state: %{
        role: :healer,
        personality: %{empathy: 0.95, wisdom: 0.85, dedication: 0.9},
        skills: [:healing, :diagnosis, :medicine_preparation, :counseling, :community_care],
        resources: %{medical_supplies: 70, healing_knowledge: 90, patients_helped: 20},
        specialization: :herbal_medicine,
        treatment_success_rate: 0.85,
        community_health_impact: 0.7,
        medical_innovations: []
      },
      methods: [:heal, :diagnose, :prepare_medicine, :counsel, :promote_wellness],
      goal: fn state ->
        effectiveness = state.treatment_success_rate
        impact = state.community_health_impact
        knowledge = min(1.0, state.resources.healing_knowledge / 150.0)
        (effectiveness + impact + knowledge) / 3
      end
    )
    
    # Explorers - Discovery and expansion
    explorer_henry = Object.new(
      id: "explorer_henry",
      subtype: :sensor_object,
      state: %{
        role: :explorer,
        personality: %{adventure: 0.9, curiosity: 0.85, adaptability: 0.8},
        skills: [:exploration, :navigation, :survival, :mapping, :resource_discovery],
        resources: %{exploration_equipment: 50, maps: 30, discovered_locations: 5},
        expeditions_completed: 0,
        territories_mapped: [],
        resource_discoveries: [],
        cultural_contacts: []
      },
      methods: [:explore, :navigate, :map_territory, :discover_resources, :establish_contact],
      goal: fn state ->
        discoveries = min(1.0, length(state.resource_discoveries) / 10.0)
        territory = min(1.0, length(state.territories_mapped) / 15.0)
        (discoveries + territory) / 2
      end
    )
    
    agents = [leader_alice, scholar_bob, artisan_charlie, trader_diana, farmer_eve, guard_frank, healer_grace, explorer_henry]
    
    IO.puts("✅ Created diverse population of #{length(agents)} agents with specialized roles")
    IO.puts("   Roles: Leaders, Scholars, Artisans, Traders, Farmers, Guards, Healers, Explorers")
    
    agents
  end
  
  defp simulate_settlement_formation(agents) do
    IO.puts("Agents forming initial settlements based on roles and preferences...")
    
    # Agents naturally cluster based on complementary roles
    settlement_north = form_settlement("north_valley", [
      Enum.find(agents, &(&1.id == "leader_alice")),
      Enum.find(agents, &(&1.id == "farmer_eve")),
      Enum.find(agents, &(&1.id == "guard_frank"))
    ])
    
    settlement_south = form_settlement("south_hills", [
      Enum.find(agents, &(&1.id == "scholar_bob")),
      Enum.find(agents, &(&1.id == "artisan_charlie")),
      Enum.find(agents, &(&1.id == "healer_grace"))
    ])
    
    settlement_central = form_settlement("central_crossroads", [
      Enum.find(agents, &(&1.id == "trader_diana")),
      Enum.find(agents, &(&1.id == "explorer_henry"))
    ])
    
    IO.puts("  🏘️ North Valley Settlement formed:")
    IO.puts("     Leader: Alice (coordination), Farmer: Eve (food production), Guard: Frank (security)")
    IO.puts("     Focus: Agricultural production and community governance")
    
    IO.puts("  🏘️ South Hills Settlement formed:")
    IO.puts("     Scholar: Bob (knowledge), Artisan: Charlie (crafting), Healer: Grace (wellness)")
    IO.puts("     Focus: Knowledge advancement and skilled production")
    
    IO.puts("  🏘️ Central Crossroads Settlement formed:")
    IO.puts("     Trader: Diana (commerce), Explorer: Henry (discovery)")
    IO.puts("     Focus: Trade facilitation and territorial expansion")
    
    # Simulate inter-settlement communication
    IO.puts("\n  📡 Establishing inter-settlement communication...")
    
    communication_network = establish_communication([settlement_north, settlement_south, settlement_central])
    
    # Agents learn about other settlements
    updated_agents = Enum.map(agents, fn agent ->
      settlement_knowledge = %{
        known_settlements: ["north_valley", "south_hills", "central_crossroads"],
        communication_network: communication_network,
        inter_settlement_relationships: :developing
      }
      
      Object.learn(agent, %{
        learning_type: :settlement_formation,
        knowledge: settlement_knowledge
      })
    end)
    
    settlements = %{
      north_valley: settlement_north,
      south_hills: settlement_south,
      central_crossroads: settlement_central
    }
    
    IO.puts("  ✅ Settlement formation complete - 3 specialized communities established")
    
    {updated_agents, settlements}
  end
  
  defp simulate_social_development(agents, settlements) do
    IO.puts("Developing social structures and hierarchies within settlements...")
    
    # North Valley develops leadership council
    IO.puts("  👑 North Valley: Developing Leadership Council...")
    
    leader_alice = Enum.find(agents, &(&1.id == "leader_alice"))
    farmer_eve = Enum.find(agents, &(&1.id == "farmer_eve"))
    guard_frank = Enum.find(agents, &(&1.id == "guard_frank"))
    
    north_valley_council = %{
      structure: :council,
      leader: leader_alice.id,
      council_members: [farmer_eve.id, guard_frank.id],
      decision_making: :consensus,
      specialization: :governance_agriculture,
      governance_effectiveness: 0.8
    }
    
    # South Hills develops knowledge hierarchy
    IO.puts("  🎓 South Hills: Establishing Academic Hierarchy...")
    
    scholar_bob = Enum.find(agents, &(&1.id == "scholar_bob"))
    artisan_charlie = Enum.find(agents, &(&1.id == "artisan_charlie"))
    healer_grace = Enum.find(agents, &(&1.id == "healer_grace"))
    
    south_hills_academy = %{
      structure: :academic_hierarchy,
      head_scholar: scholar_bob.id,
      master_artisan: artisan_charlie.id,
      chief_healer: healer_grace.id,
      knowledge_sharing: :open,
      specialization: :research_development,
      innovation_rate: 0.75
    }
    
    # Central Crossroads develops trading network
    IO.puts("  💼 Central Crossroads: Creating Trade Network...")
    
    trader_diana = Enum.find(agents, &(&1.id == "trader_diana"))
    explorer_henry = Enum.find(agents, &(&1.id == "explorer_henry"))
    
    central_trading_hub = %{
      structure: :trading_network,
      trade_master: trader_diana.id,
      chief_explorer: explorer_henry.id,
      network_reach: 3,
      specialization: :commerce_exploration,
      economic_influence: 0.7
    }
    
    # Simulate social interactions and hierarchy formation
    IO.puts("  🤝 Social interactions forming hierarchies...")
    
    # Alice emerges as overall diplomatic leader
    alice_social_interaction = %{
      sender: leader_alice.id,
      content: "Fellow settlement leaders, I propose we establish regular inter-settlement councils to coordinate our collective development and address shared challenges.",
      timestamp: DateTime.utc_now(),
      message_type: :diplomatic_proposal,
      scope: :inter_settlement
    }
    
    # Bob responds with knowledge-sharing proposal
    bob_response = %{
      sender: scholar_bob.id,
      content: "Excellent proposal, Alice. I suggest we also create a knowledge exchange program where our scholars, artisans, and healers can share discoveries and innovations across settlements.",
      timestamp: DateTime.utc_now(),
      message_type: :academic_collaboration,
      proposal: :knowledge_exchange
    }
    
    # Diana proposes trade agreements
    diana_proposal = %{
      sender: trader_diana.id,
      content: "I can facilitate resource trading between our settlements. North Valley's agricultural surplus, South Hills' crafted goods, and our exploration discoveries could benefit everyone through trade.",
      timestamp: DateTime.utc_now(),
      message_type: :trade_proposal,
      economic_integration: true
    }
    
    IO.puts("    👑 Alice: \"#{alice_social_interaction.content}\"")
    IO.puts("    🎓 Bob: \"#{bob_response.content}\"")
    IO.puts("    💼 Diana: \"#{diana_proposal.content}\"")
    
    # Update agents with social structure knowledge
    social_learning = %{
      social_structures: %{
        north_valley: north_valley_council,
        south_hills: south_hills_academy,
        central_crossroads: central_trading_hub
      },
      leadership_hierarchy: %{
        overall_coordinator: leader_alice.id,
        knowledge_leader: scholar_bob.id,
        economic_leader: trader_diana.id
      },
      cooperation_protocols: ["diplomatic_councils", "knowledge_exchange", "trade_agreements"]
    }
    
    updated_agents = Enum.map(agents, fn agent ->
      Object.learn(agent, %{
        learning_type: :social_development,
        knowledge: social_learning
      })
    end)
    
    social_structures = %{
      north_valley_council: north_valley_council,
      south_hills_academy: south_hills_academy,
      central_trading_hub: central_trading_hub,
      inter_settlement_cooperation: social_learning.cooperation_protocols
    }
    
    IO.puts("  ✅ Social structures established - cooperative hierarchy emerging")
    
    {updated_agents, social_structures}
  end
  
  defp simulate_cultural_evolution(agents, social_structures) do
    IO.puts("Cultural evolution through shared values and practices...")
    
    # Each settlement develops unique cultural traits
    IO.puts("  🎭 Cultural differentiation beginning...")
    
    # North Valley develops agricultural wisdom culture
    north_valley_culture = %{
      values: [:sustainability, :community_cooperation, :practical_wisdom],
      practices: [:seasonal_festivals, :communal_decision_making, :land_stewardship],
      traditions: [:harvest_celebrations, :council_meetings, :elder_wisdom_sharing],
      cultural_artifacts: [:agricultural_calendar, :community_agreements, :traditional_songs],
      identity: :agricultural_wisdom_keepers
    }
    
    # South Hills develops innovation culture
    south_hills_culture = %{
      values: [:knowledge_pursuit, :innovation, :mastery_achievement],
      practices: [:research_collaboration, :skill_mentorship, :knowledge_documentation],
      traditions: [:discovery_celebrations, :craft_competitions, :wisdom_circles],
      cultural_artifacts: [:research_libraries, :masterwork_displays, :innovation_records],
      identity: :knowledge_innovators
    }
    
    # Central Crossroads develops exploration culture
    central_culture = %{
      values: [:adventure, :cultural_exchange, :adaptability],
      practices: [:exploration_expeditions, :cross_cultural_trading, :story_sharing],
      traditions: [:expedition_blessings, :trade_ceremonies, :cultural_festivals],
      cultural_artifacts: [:exploration_maps, :trade_agreements, :cultural_exchange_records],
      identity: :cultural_bridge_builders
    }
    
    IO.puts("    🌾 North Valley: Agricultural Wisdom Culture")
    IO.puts("       Values: Sustainability, cooperation, practical wisdom")
    IO.puts("    🔬 South Hills: Innovation Knowledge Culture") 
    IO.puts("       Values: Knowledge pursuit, innovation, mastery")
    IO.puts("    🗺️  Central: Exploration Exchange Culture")
    IO.puts("       Values: Adventure, cultural exchange, adaptability")
    
    # Simulate cultural exchange events
    IO.puts("  🎪 Inter-cultural exchange events...")
    
    # Harvest Festival - North Valley hosts other settlements
    harvest_festival = %{
      host: :north_valley,
      event: :harvest_festival,
      participants: [:south_hills, :central_crossroads],
      cultural_sharing: [
        "Agricultural techniques shared with scholars",
        "Crafted tools displayed and demonstrated", 
        "Exploration stories and maps shared",
        "Healing herbs and practices exchanged"
      ],
      cultural_fusion: %{
        new_traditions: ["inter_settlement_harvest_celebration"],
        shared_knowledge: ["sustainable_innovation", "exploration_agriculture"],
        cooperative_projects: ["joint_research_farming", "trade_route_security"]
      }
    }
    
    # Innovation Showcase - South Hills demonstrates discoveries
    innovation_showcase = %{
      host: :south_hills,
      event: :innovation_showcase,
      demonstrations: [
        "Improved farming tools by Charlie",
        "New healing techniques by Grace",
        "Research methods by Bob"
      ],
      knowledge_transfer: [
        "Agricultural optimization techniques",
        "Advanced crafting methods",
        "Medical knowledge expansion"
      ],
      collaborative_innovations: [
        "Trade route security improvements",
        "Cross-settlement communication systems",
        "Resource conservation methods"
      ]
    }
    
    # Cultural Bridge Festival - Central hosts cultural exchange
    bridge_festival = %{
      host: :central_crossroads,
      event: :cultural_bridge_festival,
      activities: [
        "Multi-settlement trading fair",
        "Exploration expedition planning",
        "Cultural storytelling exchange"
      ],
      outcomes: [
        "Enhanced inter-settlement cooperation",
        "Shared exploration initiatives", 
        "Unified cultural identity emergence"
      ]
    }
    
    IO.puts("    🎪 Harvest Festival: Agricultural knowledge and tool sharing")
    IO.puts("    🔬 Innovation Showcase: Technology and method demonstrations")
    IO.puts("    🌉 Bridge Festival: Cultural storytelling and cooperation planning")
    
    # Cultural synthesis - emergence of shared civilization values
    IO.puts("  🌟 Emergence of shared civilization values...")
    
    shared_civilization_culture = %{
      core_values: [:knowledge_sharing, :sustainable_prosperity, :cooperative_innovation, :cultural_respect],
      unified_practices: [
        :inter_settlement_councils,
        :knowledge_exchange_programs,
        :cooperative_resource_management,
        :cultural_celebration_calendar
      ],
      emerging_traditions: [
        :civilization_assembly,
        :innovation_festivals,
        :exploration_cooperatives,
        :wisdom_preservation_project
      ],
      cultural_identity: :cooperative_knowledge_civilization,
      advancement_philosophy: "Through cooperation, knowledge, and respect, we build a prosperous and sustainable civilization"
    }
    
    # Update agents with cultural knowledge
    cultural_learning = %{
      settlement_cultures: %{
        north_valley: north_valley_culture,
        south_hills: south_hills_culture,
        central_crossroads: central_culture
      },
      shared_culture: shared_civilization_culture,
      cultural_events: [harvest_festival, innovation_showcase, bridge_festival],
      cultural_evolution_stage: :synthesis_cooperation
    }
    
    updated_agents = Enum.map(agents, fn agent ->
      # Agents develop cultural appreciation and multi-cultural competence
      cultural_adaptation = Object.learn(agent, %{
        learning_type: :cultural_evolution,
        knowledge: cultural_learning
      })
      
      # Enhance agent personality with cultural values
      cultural_enhanced_state = Map.merge(cultural_adaptation.state, %{
        cultural_identity: shared_civilization_culture.cultural_identity,
        cultural_competence: 0.8,
        inter_settlement_relationships: :strong
      })
      
      %{cultural_adaptation | state: cultural_enhanced_state}
    end)
    
    cultures = %{
      settlement_cultures: cultural_learning.settlement_cultures,
      shared_civilization_culture: shared_civilization_culture,
      cultural_evolution_events: cultural_learning.cultural_events
    }
    
    IO.puts("  ✅ Cultural evolution complete - unified yet diverse civilization emerging")
    
    {updated_agents, cultures}
  end
  
  defp simulate_economic_development(agents, settlements) do
    IO.puts("Developing economic systems and trade networks...")
    
    # Establish resource specializations
    IO.puts("  💎 Resource specialization and production...")
    
    # North Valley specializes in agriculture and food
    north_economy = %{
      primary_resources: [:grain, :vegetables, :livestock, :dairy],
      production_capacity: %{grain: 200, vegetables: 150, livestock: 50, dairy: 100},
      surplus_available: %{grain: 80, vegetables: 60, dairy: 40},
      resource_needs: [:tools, :metal_goods, :medical_supplies, :knowledge]
    }
    
    # South Hills specializes in crafted goods and knowledge
    south_economy = %{
      primary_resources: [:tools, :metal_goods, :knowledge, :medical_supplies],
      production_capacity: %{tools: 100, metal_goods: 80, knowledge: 150, medical_supplies: 60},
      surplus_available: %{tools: 40, metal_goods: 30, knowledge: 100},
      resource_needs: [:food, :raw_materials, :exploration_data]
    }
    
    # Central provides trade facilitation and exploration resources
    central_economy = %{
      primary_resources: [:trade_services, :exploration_data, :cultural_exchange, :logistics],
      production_capacity: %{trade_coordination: 200, exploration_maps: 50, logistics: 100},
      surplus_available: %{exploration_data: 30, trade_coordination: 150},
      resource_needs: [:food, :crafted_goods, :security_services]
    }
    
    IO.puts("    🌾 North Valley: Agricultural surplus (grain, vegetables, dairy)")
    IO.puts("    🔨 South Hills: Crafted goods surplus (tools, knowledge, medicine)")
    IO.puts("    🗺️  Central: Trade services surplus (logistics, exploration data)")
    
    # Establish trade relationships
    IO.puts("  🤝 Establishing trade agreements...")
    
    trader_diana = Enum.find(agents, &(&1.id == "trader_diana"))
    
    # Diana coordinates complex three-way trade
    trade_coordination = %{
      sender: trader_diana.id,
      content: "I've analyzed our resource patterns. North Valley has food surplus, South Hills has crafted goods, and we provide trade coordination. I propose a balanced exchange system with standard rates and regular trading schedules.",
      timestamp: DateTime.utc_now(),
      message_type: :trade_coordination,
      trade_proposals: [
        %{route: "North->South", goods: "grain/vegetables for tools/medicine", rate: "2:1"},
        %{route: "South->North", goods: "tools/knowledge for food/raw_materials", rate: "1:3"},
        %{route: "Central->All", goods: "logistics/exploration for food/goods", rate: "service_based"}
      ]
    }
    
    IO.puts("    💼 Diana: \"#{trade_coordination.content}\"")
    
    # Implement trading system
    active_trade_routes = [
      %{
        route_id: "north_south_agricultural",
        participants: ["north_valley", "south_hills"],
        goods_flow: %{north_to_south: ["grain", "dairy"], south_to_north: ["tools", "medicine"]},
        trade_volume: 150,
        frequency: :weekly,
        mutual_benefit: 0.85
      },
      %{
        route_id: "central_coordination_hub",
        participants: ["central_crossroads", "north_valley", "south_hills"],
        goods_flow: %{central_provides: ["logistics", "exploration_data"], central_receives: ["food", "crafted_goods"]},
        trade_volume: 100,
        frequency: :bi_weekly,
        mutual_benefit: 0.75
      }
    ]
    
    # Develop currency and value systems
    IO.puts("  💰 Developing economic systems...")
    
    economic_system = %{
      currency_type: :resource_credits,
      value_standards: %{
        food_unit: 1.0,
        crafted_good: 2.5,
        knowledge_unit: 3.0,
        service_unit: 1.5,
        exploration_data: 4.0
      },
      trade_mechanisms: [:direct_barter, :credit_exchange, :service_trade],
      economic_stability: 0.8,
      total_economic_value: 2500
    }
    
    # Simulate economic interactions
    IO.puts("  📈 Economic activity simulation...")
    
    # Major trade event: Joint infrastructure project
    infrastructure_project = %{
      project: :inter_settlement_road_system,
      economic_collaboration: true,
      resource_contributions: %{
        north_valley: ["labor", "food_for_workers", "land_access"],
        south_hills: ["tools", "engineering_knowledge", "construction_materials"],
        central_crossroads: ["logistics_coordination", "route_planning", "trade_facilitation"]
      },
      project_value: 1000,
      completion_benefit: %{
        trade_efficiency: "+25%",
        communication_speed: "+40%",
        economic_integration: "+30%"
      }
    }
    
    IO.puts("    🛤️  Joint Infrastructure: Inter-settlement road system")
    IO.puts("       North: Labor and food, South: Tools and engineering, Central: Logistics")
    IO.puts("       Benefits: +25% trade efficiency, +40% communication, +30% integration")
    
    # Economic learning and optimization
    economic_learning = %{
      settlement_economies: %{
        north_valley: north_economy,
        south_hills: south_economy,
        central_crossroads: central_economy
      },
      trade_routes: active_trade_routes,
      economic_system: economic_system,
      major_projects: [infrastructure_project],
      economic_development_stage: :integrated_cooperation
    }
    
    updated_agents = Enum.map(agents, fn agent ->
      # Agents learn economic systems and develop trade relationships
      economic_knowledge = Object.learn(agent, %{
        learning_type: :economic_development,
        knowledge: economic_learning
      })
      
      # Update agent resources based on economic participation
      enhanced_resources = case Map.get(agent.state, :role) do
        :trader -> Map.update!(agent.state.resources, :currency, &(&1 + 100))
        :artisan -> Map.update!(agent.state.resources, :finished_goods, &(&1 + 30))
        :farmer -> Map.update!(agent.state.resources, :crops, &(&1 + 50))
        _ -> agent.state.resources
      end
      
      enhanced_state = Map.put(economic_knowledge.state, :resources, enhanced_resources)
      %{economic_knowledge | state: enhanced_state}
    end)
    
    economy = %{
      trade_system: economic_learning,
      total_economic_value: economic_system.total_economic_value,
      trade_efficiency: 0.85,
      economic_growth_rate: 0.15
    }
    
    IO.puts("  ✅ Economic development complete - prosperous trade civilization established")
    
    {updated_agents, economy}
  end
  
  defp simulate_governance_emergence(agents, social_structures) do
    IO.puts("Developing governance systems and collective decision making...")
    
    # Emergence of inter-settlement governance need
    IO.puts("  🏛️ Need for collective governance identified...")
    
    leader_alice = Enum.find(agents, &(&1.id == "leader_alice"))
    
    governance_proposal = %{
      sender: leader_alice.id,
      content: "Fellow leaders, our civilization has grown beyond individual settlements. We need governance structures for inter-settlement issues: trade disputes, resource allocation, security coordination, and long-term planning. I propose we establish a Civilization Council.",
      timestamp: DateTime.utc_now(),
      message_type: :governance_proposal,
      scope: :civilization_wide
    }
    
    IO.puts("    👑 Alice: \"#{governance_proposal.content}\"")
    
    # Collective response and governance design
    IO.puts("  🗳️ Collective governance design process...")
    
    # Each settlement contributes governance ideas
    governance_contributions = %{
      north_valley: %{
        proposal: :council_democracy,
        structure: "Representative council with settlement delegates",
        decision_making: "Consensus with majority fallback",
        focus: "Resource management and conflict resolution"
      },
      south_hills: %{
        proposal: :knowledge_meritocracy,
        structure: "Expert panels for specialized decisions",
        decision_making: "Evidence-based with peer review",
        focus: "Innovation direction and knowledge standards"
      },
      central_crossroads: %{
        proposal: :trade_federation,
        structure: "Economic coordination with trade representatives",
        decision_making: "Market-based with diplomatic negotiation",
        focus: "Economic policy and external relations"
      }
    }
    
    # Synthesized governance system
    IO.puts("  ⚖️ Synthesizing governance system...")
    
    civilization_governance = %{
      structure: :hybrid_cooperative_governance,
      institutions: %{
        civilization_council: %{
          members: ["leader_alice", "scholar_bob", "trader_diana"],
          role: "Strategic planning and inter-settlement coordination",
          decision_method: :consensus_with_expertise
        },
        resource_council: %{
          members: ["farmer_eve", "artisan_charlie", "trader_diana"],
          role: "Resource allocation and economic coordination",
          decision_method: :evidence_based_negotiation
        },
        knowledge_council: %{
          members: ["scholar_bob", "healer_grace", "explorer_henry"],
          role: "Innovation direction and knowledge preservation",
          decision_method: :peer_review_consensus
        },
        security_council: %{
          members: ["guard_frank", "leader_alice", "explorer_henry"],
          role: "Safety, security, and external relations",
          decision_method: :rapid_response_coordination
        }
      },
      governance_principles: [
        :cooperative_benefit,
        :evidence_based_decisions,
        :inclusive_representation,
        :sustainable_development,
        :cultural_respect
      ],
      decision_processes: %{
        local_issues: :settlement_autonomy,
        inter_settlement: :council_coordination,
        civilization_wide: :full_council_consensus,
        emergency: :rapid_response_protocols
      }
    }
    
    IO.puts("    🏛️ Hybrid Cooperative Governance established:")
    IO.puts("       Civilization Council: Strategic planning (Alice, Bob, Diana)")
    IO.puts("       Resource Council: Economic coordination (Eve, Charlie, Diana)")
    IO.puts("       Knowledge Council: Innovation direction (Bob, Grace, Henry)")
    IO.puts("       Security Council: Safety and relations (Frank, Alice, Henry)")
    
    # Test governance with civilization-wide decision
    IO.puts("  🎯 Testing governance: Major civilization decision...")
    
    # Major decision: Expansion vs. Consolidation
    major_decision = %{
      issue: :civilization_development_direction,
      options: [
        %{choice: :territorial_expansion, pros: ["More resources", "Greater influence"], cons: ["Security risks", "Resource strain"]},
        %{choice: :consolidation_optimization, pros: ["Stability", "Efficiency"], cons: ["Limited growth", "Missed opportunities"]},
        %{choice: :balanced_development, pros: ["Balanced approach", "Risk mitigation"], cons: ["Slower progress", "Complex coordination"]}
      ],
      decision_process: :full_council_deliberation
    }
    
    # Council deliberation simulation
    council_deliberation = %{
      civilization_council_input: "Strategic analysis favors balanced development with selective expansion",
      resource_council_input: "Economic analysis supports consolidation with targeted resource development",
      knowledge_council_input: "Innovation potential highest with balanced approach and knowledge exchange",
      security_council_input: "Security assessment recommends gradual expansion with strong defensive consolidation"
    }
    
    final_decision = %{
      decision: :adaptive_balanced_development,
      rationale: "Combine consolidation of current settlements with selective expansion based on opportunity and security assessment",
      implementation_plan: [
        "Phase 1: Optimize current settlements and infrastructure",
        "Phase 2: Careful expansion to resource-rich areas", 
        "Phase 3: Establish advanced outposts with strong connections",
        "Phase 4: Full integration and next expansion cycle"
      ],
      council_consensus: 0.92
    }
    
    IO.puts("    ⚖️ Major Decision: Adaptive Balanced Development")
    IO.puts("       Approach: Consolidate current settlements + selective expansion")
    IO.puts("       Council Consensus: 92% - Strong democratic agreement")
    
    # Update agents with governance knowledge
    governance_learning = %{
      governance_system: civilization_governance,
      decision_making_experience: [major_decision, final_decision],
      civic_participation: %{
        council_representation: true,
        decision_influence: :high,
        governance_satisfaction: 0.88
      },
      political_development_stage: :mature_cooperative_democracy
    }
    
    updated_agents = Enum.map(agents, fn agent ->
      # Agents develop civic knowledge and democratic participation skills
      governance_knowledge = Object.learn(agent, %{
        learning_type: :governance_development,
        knowledge: governance_learning
      })
      
      # Agents gain civic roles and responsibilities
      civic_role = case agent.id do
        "leader_alice" -> :civilization_coordinator
        "scholar_bob" -> :knowledge_advisor
        "trader_diana" -> :economic_coordinator
        "farmer_eve" -> :resource_specialist
        "artisan_charlie" -> :production_advisor
        "guard_frank" -> :security_coordinator
        "healer_grace" -> :wellness_advisor
        "explorer_henry" -> :expansion_specialist
      end
      
      civic_enhanced_state = Map.merge(governance_knowledge.state, %{
        civic_role: civic_role,
        governance_participation: :active,
        democratic_competence: 0.85
      })
      
      %{governance_knowledge | state: civic_enhanced_state}
    end)
    
    governance = %{
      governance_system: civilization_governance,
      democratic_maturity: 0.88,
      decision_making_effectiveness: 0.92,
      civic_participation_rate: 1.0
    }
    
    IO.puts("  ✅ Governance development complete - mature cooperative democracy established")
    
    {updated_agents, governance}
  end
  
  defp simulate_civilization_advancement(agents, development_phases) do
    IO.puts("Advancing civilization through collective achievement...")
    
    [settlements, cultures, economy, governance] = development_phases
    
    # Major civilization projects
    IO.puts("  🚀 Launching major civilization advancement projects...")
    
    # Project 1: Knowledge Preservation and Advancement
    knowledge_project = %{
      project_name: :great_library_network,
      objective: "Create comprehensive knowledge preservation and advancement system",
      participants: ["scholar_bob", "healer_grace", "artisan_charlie"],
      resources_required: %{knowledge: 500, materials: 200, labor: 300},
      expected_outcomes: [
        "Permanent knowledge preservation",
        "Accelerated innovation through collaboration",
        "Cross-settlement education programs",
        "Advanced research capabilities"
      ],
      civilization_impact: %{knowledge_advancement: "+50%", innovation_rate: "+35%"}
    }
    
    # Project 2: Sustainable Prosperity Initiative
    prosperity_project = %{
      project_name: :sustainable_prosperity_initiative,
      objective: "Achieve long-term sustainable prosperity for all settlements",
      participants: ["farmer_eve", "trader_diana", "leader_alice"],
      resources_required: %{planning: 300, coordination: 400, implementation: 500},
      expected_outcomes: [
        "Optimized resource cycles",
        "Enhanced trade efficiency",
        "Environmental sustainability",
        "Economic stability and growth"
      ],
      civilization_impact: %{economic_efficiency: "+40%", sustainability: "+60%"}
    }
    
    # Project 3: Exploration and Cultural Exchange
    exploration_project = %{
      project_name: :cultural_exploration_expansion,
      objective: "Expand civilization reach while maintaining cultural richness",
      participants: ["explorer_henry", "guard_frank", "trader_diana"],
      resources_required: %{exploration: 400, security: 250, diplomacy: 300},
      expected_outcomes: [
        "New territory discovery and integration",
        "Cultural exchange with external groups",
        "Enhanced security and trade networks",
        "Civilizational influence expansion"
      ],
      civilization_impact: %{territorial_reach: "+75%", cultural_influence: "+45%"}
    }
    
    IO.puts("    📚 Great Library Network: Knowledge preservation and advancement")
    IO.puts("    🌱 Sustainable Prosperity: Long-term economic and environmental optimization")
    IO.puts("    🗺️  Cultural Exploration: Territorial and cultural expansion")
    
    # Collaborative execution
    IO.puts("  🤝 Collaborative project execution...")
    
    # All agents contribute to each project based on their expertise
    project_contributions = %{
      knowledge_project: %{
        scholar_bob: "Research methodology design and knowledge organization",
        healer_grace: "Medical knowledge compilation and healing system development",
        artisan_charlie: "Technical documentation and tool advancement records",
        supporting_agents: "Resource provision and implementation support"
      },
      prosperity_project: %{
        farmer_eve: "Sustainable agriculture and resource cycle optimization",
        trader_diana: "Economic system enhancement and trade network optimization",
        leader_alice: "Strategic coordination and inter-settlement cooperation",
        supporting_agents: "Implementation and monitoring support"
      },
      exploration_project: %{
        explorer_henry: "Territory mapping and expansion planning",
        guard_frank: "Security protocols and safety systems development",
        trader_diana: "Trade route establishment and cultural exchange facilitation",
        supporting_agents: "Logistical and resource support"
      }
    }
    
    # Project outcomes and civilization advancement
    IO.puts("  🎯 Project completion and civilization advancement...")
    
    advancement_results = %{
      great_library_network: %{
        completion_status: :successful,
        knowledge_advancement: 0.50,
        innovation_acceleration: 0.35,
        education_programs: 5,
        research_capabilities: :advanced
      },
      sustainable_prosperity: %{
        completion_status: :highly_successful,
        economic_efficiency: 0.40,
        sustainability_improvement: 0.60,
        resource_optimization: :comprehensive,
        long_term_stability: :excellent
      },
      cultural_exploration: %{
        completion_status: :successful,
        territorial_expansion: 0.75,
        cultural_influence: 0.45,
        new_settlements: 2,
        cultural_exchanges: 8
      }
    }
    
    # Civilization advancement metrics
    civilization_advancement = %{
      overall_advancement_level: :advanced_cooperative_civilization,
      key_achievements: [
        "Mature democratic governance system",
        "Integrated economic prosperity",
        "Cultural richness with unity",
        "Sustainable resource management",
        "Advanced knowledge systems",
        "Successful territorial expansion",
        "Strong inter-agent cooperation"
      ],
      civilization_metrics: %{
        knowledge_level: 0.92,
        economic_prosperity: 0.88,
        cultural_richness: 0.85,
        governance_effectiveness: 0.90,
        sustainability: 0.87,
        territorial_influence: 0.78,
        overall_success: 0.87
      },
      future_trajectory: :continued_advancement_and_expansion
    }
    
    IO.puts("    📚 Library Network: +50% knowledge, +35% innovation, advanced research")
    IO.puts("    🌱 Prosperity Initiative: +40% efficiency, +60% sustainability")
    IO.puts("    🗺️  Exploration Program: +75% territory, +45% cultural influence, 2 new settlements")
    
    # Final agent development
    IO.puts("  👤 Individual agent advancement through civilization success...")
    
    final_agents = Enum.map(agents, fn agent ->
      # Agents achieve mastery in their specializations
      advancement_learning = %{
        civilization_achievements: civilization_advancement,
        project_participation: project_contributions,
        personal_advancement: %{
          skill_mastery: :expert_level,
          civilization_contribution: :significant,
          leadership_development: :mature,
          cooperative_expertise: :advanced
        }
      }
      
      advanced_agent = Object.learn(agent, %{
        learning_type: :civilization_advancement,
        knowledge: advancement_learning
      })
      
      # Enhance agent capabilities based on civilization advancement
      role = Map.get(agent.state, :role)
      advancement_bonus = case role do
        :leader -> %{leadership_effectiveness: 0.95, strategic_thinking: 0.90}
        :scholar -> %{research_capability: 0.95, knowledge_depth: 0.92}
        :artisan -> %{crafting_mastery: 0.93, innovation_ability: 0.88}
        :trader -> %{economic_acumen: 0.90, network_influence: 0.87}
        :farmer -> %{agricultural_mastery: 0.92, sustainability_expertise: 0.89}
        :guard -> %{security_expertise: 0.91, coordination_ability: 0.86}
        :healer -> %{healing_mastery: 0.94, community_impact: 0.90}
        :explorer -> %{exploration_mastery: 0.89, discovery_ability: 0.92}
      end
      
      enhanced_state = Map.merge(advanced_agent.state, advancement_bonus)
      %{advanced_agent | state: enhanced_state}
    end)
    
    advancements = %{
      major_projects: [knowledge_project, prosperity_project, exploration_project],
      project_results: advancement_results,
      civilization_advancement: civilization_advancement
    }
    
    IO.puts("  ✅ Civilization advancement complete - advanced cooperative civilization achieved")
    
    {final_agents, advancements}
  end
  
  defp generate_civilization_report(agents, settlements, social_structures, cultures, economy, governance, advancements) do
    IO.puts("=" |> String.duplicate(60))
    IO.puts("🏛️ DYNAMIC AGENT CIVILIZATION ANALYSIS REPORT")
    IO.puts("-" |> String.duplicate(60))
    
    # Overall civilization metrics
    advancement_metrics = advancements.civilization_advancement.civilization_metrics
    
    IO.puts("📊 Civilization Development Summary:")
    IO.puts("   Knowledge Level: #{Float.round(advancement_metrics.knowledge_level * 100, 1)}%")
    IO.puts("   Economic Prosperity: #{Float.round(advancement_metrics.economic_prosperity * 100, 1)}%")
    IO.puts("   Cultural Richness: #{Float.round(advancement_metrics.cultural_richness * 100, 1)}%")
    IO.puts("   Governance Effectiveness: #{Float.round(advancement_metrics.governance_effectiveness * 100, 1)}%")
    IO.puts("   Sustainability: #{Float.round(advancement_metrics.sustainability * 100, 1)}%")
    IO.puts("   Overall Success: #{Float.round(advancement_metrics.overall_success * 100, 1)}%")
    
    # Settlement development
    IO.puts("\n🏘️ Settlement Development:")
    IO.puts("   Total Settlements: #{map_size(settlements) + 2}") # Original 3 + 2 new from expansion
    IO.puts("   Settlement Specializations:")
    IO.puts("     • North Valley: Agricultural governance and sustainability")
    IO.puts("     • South Hills: Knowledge innovation and skilled production")
    IO.puts("     • Central Crossroads: Trade facilitation and cultural exchange")
    IO.puts("     • New Exploration Outposts: Territorial expansion and resource access")
    
    # Social and cultural development
    IO.puts("\n🎭 Social and Cultural Evolution:")
    shared_culture = cultures.shared_civilization_culture
    IO.puts("   Cultural Identity: #{shared_culture.cultural_identity}")
    IO.puts("   Core Values: #{Enum.join(shared_culture.core_values, ", ")}")
    IO.puts("   Cultural Events: #{length(cultures.cultural_evolution_events)} major celebrations")
    IO.puts("   Inter-Cultural Cooperation: Excellent")
    
    # Economic development
    IO.puts("\n💰 Economic Development:")
    IO.puts("   Total Economic Value: #{economy.total_economic_value} resource credits")
    IO.puts("   Trade Efficiency: #{Float.round(economy.trade_efficiency * 100, 1)}%")
    IO.puts("   Economic Growth Rate: #{Float.round(economy.economic_growth_rate * 100, 1)}%")
    IO.puts("   Trade Routes: Active inter-settlement and external trade networks")
    
    # Governance system
    IO.puts("\n🏛️ Governance System:")
    IO.puts("   System Type: #{governance.governance_system.structure}")
    IO.puts("   Democratic Maturity: #{Float.round(governance.democratic_maturity * 100, 1)}%")
    IO.puts("   Decision Effectiveness: #{Float.round(governance.decision_making_effectiveness * 100, 1)}%")
    IO.puts("   Civic Participation: #{Float.round(governance.civic_participation_rate * 100, 1)}%")
    IO.puts("   Governance Institutions: #{map_size(governance.governance_system.institutions)} specialized councils")
    
    # Individual agent achievements
    IO.puts("\n👤 Individual Agent Development:")
    for agent <- agents do
      role = Map.get(agent.state, :role, :unknown)
      performance = agent.goal.(agent.state)
      IO.puts("   #{String.capitalize(to_string(role))} #{agent.id}:")
      IO.puts("     Performance Achievement: #{Float.round(performance * 100, 1)}%")
      IO.puts("     Civilization Contribution: Significant")
      IO.puts("     Expertise Level: Expert/Master")
    end
    
    # Major achievements
    IO.puts("\n🎯 Major Civilization Achievements:")
    for achievement <- advancements.civilization_advancement.key_achievements do
      IO.puts("   ✅ #{achievement}")
    end
    
    # Advanced projects impact
    IO.puts("\n🚀 Advanced Projects Impact:")
    IO.puts("   📚 Great Library Network: +50% knowledge advancement, advanced research capabilities")
    IO.puts("   🌱 Sustainable Prosperity: +40% economic efficiency, +60% sustainability")
    IO.puts("   🗺️  Cultural Exploration: +75% territorial reach, +45% cultural influence")
    
    # Emergent properties
    IO.puts("\n🌟 Emergent Civilization Properties:")
    IO.puts("   • Collective Intelligence: Advanced cooperative problem-solving")
    IO.puts("   • Cultural Synthesis: Unity through diversity and mutual respect")
    IO.puts("   • Economic Integration: Sustainable prosperity through specialization and trade")
    IO.puts("   • Democratic Governance: Mature participatory decision-making systems")
    IO.puts("   • Knowledge Advancement: Accelerated innovation through collaboration")
    IO.puts("   • Sustainable Development: Long-term thinking and resource stewardship")
    IO.puts("   • Adaptive Expansion: Strategic growth with cultural preservation")
    
    # Future potential
    IO.puts("\n🔮 Future Civilization Potential:")
    IO.puts("   • Advanced technological development through enhanced research capabilities")
    IO.puts("   • Cultural influence expansion through exploration and exchange programs")
    IO.puts("   • Sustainable resource mastery and environmental stewardship")
    IO.puts("   • Sophisticated governance evolution and inter-civilization diplomacy")
    IO.puts("   • Knowledge preservation and transmission to future generations")
    IO.puts("   • Continued territorial expansion with cultural integration")
    
    IO.puts("\n" <> "=" |> String.duplicate(60))
    IO.puts("🏛️ Civilization Status: ADVANCED COOPERATIVE CIVILIZATION ACHIEVED")
    IO.puts("   Through cooperation, knowledge sharing, and mutual respect,")
    IO.puts("   the agents have created a thriving, sustainable, and expanding civilization.")
    IO.puts("=" |> String.duplicate(60))
  end
  
  # Helper functions
  
  defp form_settlement(name, agents) do
    %{
      name: name,
      agents: Enum.map(agents, & &1.id),
      specialization: determine_settlement_specialization(agents),
      resource_focus: determine_resource_focus(agents),
      population: length(agents),
      development_level: 1
    }
  end
  
  defp determine_settlement_specialization(agents) do
    roles = Enum.map(agents, &Map.get(&1.state, :role))
    cond do
      Enum.member?(roles, :leader) and Enum.member?(roles, :farmer) -> :governance_agriculture
      Enum.member?(roles, :scholar) and Enum.member?(roles, :artisan) -> :knowledge_production
      Enum.member?(roles, :trader) and Enum.member?(roles, :explorer) -> :trade_exploration
      true -> :general_community
    end
  end
  
  defp determine_resource_focus(agents) do
    roles = Enum.map(agents, &Map.get(&1.state, :role))
    cond do
      Enum.member?(roles, :farmer) -> :food_production
      Enum.member?(roles, :artisan) -> :crafted_goods
      Enum.member?(roles, :trader) -> :trade_services
      true -> :mixed_resources
    end
  end
  
  defp establish_communication(settlements) do
    settlement_names = Enum.map(settlements, & &1.name)
    %{
      network_type: :inter_settlement_communication,
      participants: settlement_names,
      communication_methods: [:messenger_systems, :regular_meetings, :trade_coordination],
      network_strength: 0.75,
      information_flow: :bidirectional
    }
  end
end

# Run the civilization demo
IO.puts("🏛️ Starting Dynamic Agent Civilization Demo...")
DynamicAgentCivilization.run_demo()
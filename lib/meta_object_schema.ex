defmodule MetaObjectSchema do
  @moduledoc """
  Meta-schema system enabling objects to reason about and modify their own schemas.
  Implements hierarchical schema inheritance with runtime evolution capabilities.
  """

  defstruct [
    :schema_id,
    :version,
    :parent_schemas,
    :core_attributes,
    :behavioral_patterns,
    :interaction_protocols,
    :learning_parameters,
    :evolution_constraints,
    :compatibility_matrix,
    :reflection_capabilities,
    :meta_operations
  ]

  @type compatibility_score :: float()
  @type constraint :: map()
  @type inheritance_spec :: map()
  @type trigger_condition :: map()
  @type action_spec :: map()
  @type state_condition :: map()
  @type response_spec :: map()
  @type compatibility_rule :: map()
  @type learning_config :: map()
  @type evolution_rules :: map()
  @type reflection_capability :: map()
  @type meta_operation :: map()

  @type t :: %__MODULE__{
    schema_id: atom(),
    version: String.t(),
    parent_schemas: [atom()],
    core_attributes: %{atom() => schema_attribute()},
    behavioral_patterns: %{atom() => behavior_pattern()},
    interaction_protocols: %{atom() => protocol_spec()},
    learning_parameters: learning_config(),
    evolution_constraints: evolution_rules(),
    compatibility_matrix: %{atom() => compatibility_score()},
    reflection_capabilities: reflection_capability(),
    meta_operations: %{atom() => meta_operation()}
  }

  @type schema_attribute :: %{
    type: :primitive | :composite | :emergent,
    constraints: [constraint()],
    mutability: :immutable | :mutable | :evolvable,
    inheritance_rules: inheritance_spec()
  }

  @type behavior_pattern :: %{
    triggers: [trigger_condition()],
    actions: [action_spec()],
    learning_weight: float(),
    adaptation_rate: float(),
    effectiveness_threshold: float()
  }

  @type protocol_spec :: %{
    message_types: [atom()],
    state_requirements: [state_condition()],
    response_patterns: %{atom() => response_spec()},
    compatibility_requirements: [compatibility_rule()]
  }

  # Core Meta-Schema Types
  
  @doc """
  Returns the base agent schema that defines fundamental agent capabilities.
  
  Provides a foundational schema for autonomous agents with core attributes,
  behavioral patterns, interaction protocols, and meta-cognitive capabilities.
  This schema serves as the parent for all specialized agent schemas.
  
  ## Returns
  
  - Base agent schema struct with:
    - Core attributes: identity, state, goals, memory
    - Behavioral patterns: goal pursuit, social interaction
    - Interaction protocols: basic messaging
    - Learning parameters and evolution constraints
    - Reflection capabilities and meta-operations
  
  ## Examples
  
      iex> MetaObjectSchema.base_agent_schema()
      %MetaObjectSchema{schema_id: :base_agent, version: "1.0.0", ...}
  """
  def base_agent_schema do
    %__MODULE__{
      schema_id: :base_agent,
      version: "1.0.0",
      parent_schemas: [],
      core_attributes: %{
        identity: %{type: :primitive, constraints: [:unique, :immutable], mutability: :immutable},
        state: %{type: :composite, constraints: [:observable], mutability: :evolvable},
        goals: %{type: :composite, constraints: [:prioritizable], mutability: :mutable},
        memory: %{type: :emergent, constraints: [:persistent], mutability: :evolvable}
      },
      behavioral_patterns: %{
        goal_pursuit: %{
          triggers: [:goal_activation, :environmental_change],
          actions: [:plan_formation, :action_execution, :progress_monitoring],
          learning_weight: 0.8,
          adaptation_rate: 0.1,
          effectiveness_threshold: 0.6
        },
        social_interaction: %{
          triggers: [:message_received, :dyad_formation],
          actions: [:intent_analysis, :response_generation, :relationship_update],
          learning_weight: 0.6,
          adaptation_rate: 0.15,
          effectiveness_threshold: 0.5
        }
      },
      interaction_protocols: %{
        basic_messaging: %{
          message_types: [:request, :response, :notification, :heartbeat],
          state_requirements: [:active, :responsive],
          response_patterns: %{
            request: %{required: true, timeout: 5000, retry_count: 3},
            notification: %{required: false, timeout: 1000, retry_count: 1}
          }
        }
      },
      learning_parameters: %{
        exploration_rate: 0.2,
        exploitation_threshold: 0.7,
        forgetting_rate: 0.05,
        consolidation_interval: 3600,
        meta_learning_enabled: true
      },
      evolution_constraints: %{
        core_immutables: [:identity, :base_protocols],
        evolution_rate_limit: 0.1,
        compatibility_preservation: 0.8,
        rollback_capability: true
      },
      reflection_capabilities: %{
        self_monitoring: [:performance_tracking, :goal_assessment, :behavior_analysis],
        meta_cognition: [:schema_awareness, :adaptation_planning, :effectiveness_evaluation],
        introspection_frequency: 60000
      },
      meta_operations: %{
        schema_fork: %{conditions: [:major_adaptation_needed], cost: :high},
        schema_merge: %{conditions: [:compatible_schemas_detected], cost: :medium},
        schema_evolve: %{conditions: [:performance_threshold_exceeded], cost: :low}
      }
    }
  end

  @doc """
  Creates a specialized agent schema based on the provided specialization type.
  
  Extends the base agent schema with domain-specific capabilities and behaviors
  tailored to the specialization. Currently supports researcher, coordinator,
  and creative agent specializations.
  
  ## Parameters
  
  - `specialization` - The type of specialization (`:researcher`, `:coordinator`, `:creative`)
  
  ## Returns
  
  - Specialized agent schema struct with extended capabilities:
    - `:researcher` - Enhanced with knowledge acquisition and hypothesis formation
    - `:coordinator` - Enhanced with resource allocation and optimization strategies  
    - `:creative` - Enhanced with creative synthesis and aesthetic evaluation
  
  ## Examples
  
      iex> MetaObjectSchema.specialist_agent_schema(:researcher)
      %MetaObjectSchema{schema_id: :researcher_agent, ...}
      
      iex> MetaObjectSchema.specialist_agent_schema(:coordinator)
      %MetaObjectSchema{schema_id: :coordinator_agent, ...}
  """
  def specialist_agent_schema(specialization) do
    base = base_agent_schema()
    
    case specialization do
      :researcher -> 
        %{base | 
          schema_id: :researcher_agent,
          parent_schemas: [:base_agent],
          core_attributes: Map.merge(base.core_attributes, %{
            knowledge_domains: %{type: :composite, constraints: [:expandable], mutability: :evolvable},
            research_methodology: %{type: :emergent, constraints: [:systematic], mutability: :mutable},
            hypothesis_space: %{type: :emergent, constraints: [:bounded], mutability: :evolvable}
          }),
          behavioral_patterns: Map.merge(base.behavioral_patterns, %{
            knowledge_acquisition: %{
              triggers: [:information_gap_detected, :curiosity_threshold],
              actions: [:hypothesis_formation, :experiment_design, :data_collection],
              learning_weight: 0.9,
              adaptation_rate: 0.05,
              effectiveness_threshold: 0.75
            }
          })
        }
      
      :coordinator ->
        %{base | 
          schema_id: :coordinator_agent,
          parent_schemas: [:base_agent],
          core_attributes: Map.merge(base.core_attributes, %{
            managed_objects: %{type: :composite, constraints: [:hierarchical], mutability: :mutable},
            coordination_strategies: %{type: :emergent, constraints: [:adaptive], mutability: :evolvable},
            resource_allocation: %{type: :composite, constraints: [:optimizable], mutability: :mutable}
          }),
          behavioral_patterns: Map.merge(base.behavioral_patterns, %{
            resource_optimization: %{
              triggers: [:resource_scarcity, :efficiency_threshold],
              actions: [:load_balancing, :priority_adjustment, :delegation],
              learning_weight: 0.7,
              adaptation_rate: 0.2,
              effectiveness_threshold: 0.65
            }
          })
        }
      
      :creative ->
        %{base |
          schema_id: :creative_agent, 
          parent_schemas: [:base_agent],
          core_attributes: Map.merge(base.core_attributes, %{
            creative_domains: %{type: :emergent, constraints: [:diverse], mutability: :evolvable},
            inspiration_sources: %{type: :composite, constraints: [:varied], mutability: :mutable},
            aesthetic_preferences: %{type: :emergent, constraints: [:personal], mutability: :evolvable}
          }),
          behavioral_patterns: Map.merge(base.behavioral_patterns, %{
            creative_synthesis: %{
              triggers: [:creative_challenge, :inspiration_trigger],
              actions: [:concept_combination, :iterative_refinement, :aesthetic_evaluation],
              learning_weight: 0.6,
              adaptation_rate: 0.25,
              effectiveness_threshold: 0.4
            }
          })
        }
    end
  end

  @doc """
  Returns the rules and constraints governing meta-schema evolution.
  
  Defines the framework for how schemas can evolve over time while maintaining
  system stability and compatibility. Includes compatibility requirements,
  adaptation triggers, evolution strategies, and validation requirements.
  
  ## Returns
  
  - Map containing evolution rules with:
    - `:compatibility_preservation` - Backward compatibility requirements
    - `:adaptation_triggers` - Conditions that trigger schema evolution
    - `:evolution_strategies` - Available strategies (incremental, branching, revolutionary)
    - `:validation_requirements` - Testing and validation requirements
  
  ## Examples
  
      iex> MetaObjectSchema.meta_schema_evolution_rules()
      %{compatibility_preservation: %{minimum_backward_compatibility: 0.8}, ...}
  """
  def meta_schema_evolution_rules do
    %{
      compatibility_preservation: %{
        minimum_backward_compatibility: 0.8,
        interface_stability_requirement: true,
        gradual_deprecation_policy: true
      },
      adaptation_triggers: %{
        performance_degradation: 0.3,
        environmental_drift: 0.5,
        goal_misalignment: 0.4,
        social_pressure: 0.6
      },
      evolution_strategies: %{
        incremental: %{cost: :low, risk: :low, frequency: :high},
        branching: %{cost: :medium, risk: :medium, frequency: :medium},
        revolutionary: %{cost: :high, risk: :high, frequency: :low}
      },
      validation_requirements: %{
        simulation_testing: true,
        peer_review_consensus: 0.75,
        rollback_capability: true,
        performance_benchmarking: true
      }
    }
  end
end
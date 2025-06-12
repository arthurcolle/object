# AAOS Benchmark Configuration
# This file configures the comprehensive baseline testing suite

import Config

config :aaos_benchmarks,
  # General benchmark settings
  runs: 1000,
  warmup_runs: 100,
  parallel: true,
  save_raw_data: true,
  output_format: [:console, :html, :csv, :json],
  results_directory: "benchmarks/results",
  
  # Statistical analysis settings
  confidence_level: 0.95,
  statistical_tests: [:mann_whitney_u, :t_test, :anova],
  effect_size_calculation: :cohens_d,
  outlier_removal: :iqr_method,
  
  # Performance test configuration
  performance_tests: [
    object_creation: %{
      iterations: 10_000,
      batch_sizes: [1, 10, 100, 1000],
      measure: [:time, :memory, :reductions]
    },
    
    message_throughput: %{
      duration: 60,  # seconds
      message_sizes: [100, 1_000, 10_000],  # bytes
      patterns: [:unicast, :broadcast, :multicast]
    },
    
    coordination_latency: %{
      coalition_sizes: [2, 5, 10, 20, 50],
      coordination_types: [:consensus, :voting, :auction]
    },
    
    memory_usage: %{
      object_counts: [100, 1_000, 10_000, 100_000],
      state_complexity: [:minimal, :moderate, :complex]
    }
  ],
  
  # Learning algorithm test configuration
  learning_tests: [
    convergence_rate: %{
      environments: [:gridworld, :cartpole, :multi_agent_tag],
      algorithms: [:oorl, :q_learning, :ppo, :a3c],
      max_episodes: 10_000,
      success_threshold: 0.95
    },
    
    sample_efficiency: %{
      metrics: [:steps_to_threshold, :sample_complexity, :data_efficiency],
      environments: [:navigation, :resource_allocation, :coordination]
    },
    
    transfer_learning: %{
      source_tasks: [:maze_navigation, :object_manipulation, :planning],
      target_tasks: [:new_maze, :different_objects, :complex_planning],
      evaluation_metrics: [:zero_shot, :few_shot, :fine_tuning]
    },
    
    social_learning: %{
      population_sizes: [5, 10, 20, 50],
      communication_types: [:full, :local, :hierarchical],
      knowledge_sharing: [:experience_replay, :policy_distillation, :gradient_sharing]
    }
  ],
  
  # Scalability test configuration
  scalability_tests: [
    horizontal_scaling: %{
      node_counts: [1, 2, 4, 8, 16],
      objects_per_node: [1_000, 10_000, 50_000],
      deployment_patterns: [:homogeneous, :heterogeneous],
      network_conditions: [:lan, :wan, :cloud]
    },
    
    vertical_scaling: %{
      cpu_cores: [2, 4, 8, 16, 32],
      memory_sizes: [4, 8, 16, 32, 64],  # GB
      object_counts: [1_000, 10_000, 100_000, 1_000_000]
    },
    
    network_impact: %{
      topologies: [:full_mesh, :star, :ring, :hierarchical, :random],
      latencies: [1, 10, 50, 100, 500],  # ms
      packet_loss: [0, 0.1, 1, 5],  # percentage
      bandwidth: [:unlimited, :gigabit, :fast_ethernet, :constrained]
    }
  ],
  
  # Fault tolerance test configuration
  fault_tolerance_tests: [
    failure_recovery: %{
      failure_types: [:object_crash, :node_failure, :network_partition, :storage_failure],
      failure_rates: [0.01, 0.05, 0.1, 0.25],  # percentage
      recovery_strategies: [:restart, :failover, :replication, :consensus]
    },
    
    byzantine_resistance: %{
      byzantine_ratios: [0.1, 0.2, 0.3, 0.4],  # percentage of byzantine nodes
      attack_types: [:state_corruption, :message_tampering, :denial_of_service],
      detection_mechanisms: [:voting, :cryptographic, :behavioral]
    },
    
    cascade_prevention: %{
      initial_failure_counts: [1, 3, 5, 10],
      propagation_models: [:direct, :transitive, :epidemic],
      mitigation_strategies: [:circuit_breaker, :bulkhead, :rate_limiting]
    },
    
    resilience_under_load: %{
      load_levels: [0.5, 0.8, 1.0, 1.2, 1.5],  # relative to capacity
      failure_injection_rates: [0, 0.1, 0.25, 0.5],
      graceful_degradation: [:enabled, :disabled]
    }
  ],
  
  # Baseline comparison configuration
  baseline_comparisons: [
    actor_systems: [
      {:erlang_otp, "Native Erlang/OTP actors"},
      {:akka, "Akka actor system"},
      {:orleans, "Microsoft Orleans"},
      {:ray, "Ray distributed actors"}
    ],
    
    multi_agent_systems: [
      {:jade, "Java Agent DEvelopment framework"},
      {:netlogo, "NetLogo agent-based modeling"},
      {:mason, "Multi-Agent Simulator Of Neighborhoods"},
      {:repast, "Recursive Porous Agent Simulation Toolkit"}
    ],
    
    rl_frameworks: [
      {:ray_rllib, "Ray RLlib"},
      {:stable_baselines3, "Stable Baselines3"},
      {:openai_gym, "OpenAI Gym"},
      {:pettingzoo, "PettingZoo multi-agent"}
    ]
  ],
  
  # Continuous monitoring configuration
  continuous_monitoring: %{
    enabled: true,
    regression_threshold: 0.05,  # 5% performance regression triggers alert
    metrics_retention_days: 90,
    dashboards: [
      {:grafana, "http://localhost:3000"},
      {:prometheus, "http://localhost:9090"}
    ],
    alerts: [
      performance_regression: %{
        threshold: 0.05,
        channels: [:email, :slack]
      },
      
      memory_leak: %{
        growth_rate: 0.1,  # 10% per hour
        channels: [:pagerduty]
      },
      
      convergence_failure: %{
        max_attempts: 3,
        channels: [:email]
      }
    ]
  },
  
  # Reproducibility configuration
  reproducibility: %{
    random_seed: 42,
    capture_system_info: true,
    git_commit_hash: true,
    environment_variables: true,
    dependency_versions: true,
    hardware_specs: true,
    
    archive_format: :tar_gz,
    archive_location: "benchmarks/archives",
    
    docker_image: "aaos/benchmark:latest",
    docker_compose: "benchmarks/docker-compose.yml"
  }

# Environment-specific overrides
if config_env() == :ci do
  config :aaos_benchmarks,
    runs: 100,  # Fewer runs in CI
    warmup_runs: 10,
    parallel: false,  # Sequential in CI for consistency
    output_format: [:json],  # Machine-readable for CI parsing
    
    performance_tests: [
      object_creation: %{iterations: 1_000},
      message_throughput: %{duration: 10}
    ],
    
    continuous_monitoring: %{enabled: false}
end

if config_env() == :dev do
  config :aaos_benchmarks,
    runs: 10,  # Quick runs for development
    warmup_runs: 2,
    save_raw_data: false,
    
    performance_tests: [
      object_creation: %{iterations: 100},
      message_throughput: %{duration: 5}
    ]
end
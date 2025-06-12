# Test Failure Analysis Report

**Generated**: 2025-06-12 05:36:07.780060Z
**Total Failures**: 50

## Summary by Category

- **unknown**: 50 failures

## Critical Issues Requiring Immediate Attention



## Recommendations

1. Implement comprehensive error handling patterns
1. Add defensive programming checks throughout the codebase
1. Review and fix service startup dependencies
1. Enhance test reliability with proper setup/teardown
1. Consider using property-based testing for edge cases

## Detailed Failure Analysis

#### 1. CollectiveLearning module performs distributed policy optimization (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:114
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.CollectiveLearning.distributed_policy_optimization/1 is undefined or private

#### 2. PolicyLearning module interaction dyad learning processes dyad experiences (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:74
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.PolicyLearning.interaction_dyad_learning/2 is undefined or private

#### 3. PolicyLearning module performs social imitation learning (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:62
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.PolicyLearning.social_imitation_learning/3 is undefined or private

#### 4. PolicyLearning module updates policy with experiences and social context (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:31
- **Category**: unknown
- **Error Type**: MatchError
- **Severity**: low
- **Message**:      ** (MatchError) no match of right hand side value: "policy_test_object"

#### 5. CollectiveLearning module forms learning coalition with compatible objects (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:99
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.CollectiveLearning.form_learning_coalition/2 is undefined or private

#### 6. CollectiveLearning module detects emergent behaviors in coalition (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:128
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.CollectiveLearning.emergence_detection/1 is undefined or private

#### 7. OORL framework edge cases handles zero-member coalition formation (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:243
- **Category**: unknown
- **Error Type**: UndefinedFunctionError
- **Severity**: low
- **Message**:      ** (UndefinedFunctionError) function OORL.CollectiveLearning.form_learning_coalition/2 is undefined or private

#### 8. OORL framework edge cases handles malformed social context (OORLFrameworkTest)

- **Location**: oorl_framework_test.exs:228
- **Category**: unknown
- **Error Type**: MatchError
- **Severity**: low
- **Message**: ** (MatchError) no match of right hand side value: {:error, :closed}

#### 9. location refresh refreshes remote node location (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:146
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.626.0>) an exception was raised:

#### 10. statistics tracks call statistics (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:116
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.638.0>) an exception was raised:

#### 11. method calls clears cache on demand (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:88
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.641.0>) an exception was raised:

#### 12. method calls handles casts (fire-and-forget) (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:77
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.644.0>) an exception was raised:

#### 13. method calls caches method results (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:61
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.647.0>) an exception was raised:

#### 14. statistics calculates average latency (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:132
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.650.0>) an exception was raised:

#### 15. proxy creation handles unknown object gracefully (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:51
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.653.0>) an exception was raised:

#### 16. proxy creation creates proxy for remote object (Object.NetworkProxyTest)

- **Location**: object_network_proxy_test.exs:46
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.656.0>) an exception was raised:

#### 17. DiLoCo algorithm components performs outer optimization step (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:93
- **Category**: unknown
- **Error Type**: ExitError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.682.0>) exited in: GenServer.call(Object.DistributedRegistry, {:register_object, "test_trainer", %Object.DistributedTraining{object_id: "test_trainer", worker_id: "test_worker", global_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.229660Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, local_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.229661Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, optimizer_config: %{learning_rate: 0.001, epsilon: 1.0e-8, inner_optimizer: :adamw, outer_optimizer: :nesterov_momentum, momentum: 0.9, weight_decay: 0.01, beta1: 0.9, beta2: 0.999}, training_config: %{batch_size: 8, inner_steps: 10, outer_steps: 5, communication_frequency: 500, gradient_clipping: 1.0, fault_tolerance_threshold: 0.33, checkpoint_frequency: 10}, data_shard: %{shard_id: "73686172645fb9aec474", data_path: "/tmp/training_data", total_samples: 10000, current_position: 0, preprocessing_config: %{}}, communication_state: %{sync_barrier_count: 0, pending_gradients: %{}, communication_overhead: 0.0, last_sync: ~U[2025-06-12 05:36:02.229662Z], bandwidth_usage: 0.0}, performance_metrics: %{throughput: 0.0, convergence_rate: 0.0, training_loss: 0.0, communication_efficiency: 1.0, validation_loss: 0.0, wall_clock_time: 0, compute_utilization: 0.0}, fault_tolerance_state: %{health_status: :healthy, consensus_state: %{algorithm: :pbft, votes: %{}, view_number: 0, leader: nil, committed_steps: 0}, failed_workers: [], backup_checkpoints: %{}}, synchronization_barrier: nil, outer_optimizer_state: %{type: :nesterov_momentum, state: %{}, step_count: 0, accumulated_gradients: %{}}, inner_optimizer_state: %{type: :adamw, state: %{}, step_count: 0, accumulated_gradients: %{}}, step_counters: %{inner_step: 0, outer_step: 0, total_steps: 0, communication_rounds: 0}, coordination_service: nil}}, 5000)

#### 18. DiLoCo algorithm components performs inner optimization steps (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:85
- **Category**: unknown
- **Error Type**: ExitError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.684.0>) exited in: GenServer.call(Object.DistributedRegistry, {:register_object, "test_trainer", %Object.DistributedTraining{object_id: "test_trainer", worker_id: "test_worker", global_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.229742Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, local_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.229743Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, optimizer_config: %{learning_rate: 0.001, epsilon: 1.0e-8, inner_optimizer: :adamw, outer_optimizer: :nesterov_momentum, momentum: 0.9, weight_decay: 0.01, beta1: 0.9, beta2: 0.999}, training_config: %{batch_size: 8, inner_steps: 10, outer_steps: 5, communication_frequency: 500, gradient_clipping: 1.0, fault_tolerance_threshold: 0.33, checkpoint_frequency: 10}, data_shard: %{shard_id: "73686172645f3d34db26", data_path: "/tmp/training_data", total_samples: 10000, current_position: 0, preprocessing_config: %{}}, communication_state: %{sync_barrier_count: 0, pending_gradients: %{}, communication_overhead: 0.0, last_sync: ~U[2025-06-12 05:36:02.229744Z], bandwidth_usage: 0.0}, performance_metrics: %{throughput: 0.0, convergence_rate: 0.0, training_loss: 0.0, communication_efficiency: 1.0, validation_loss: 0.0, wall_clock_time: 0, compute_utilization: 0.0}, fault_tolerance_state: %{health_status: :healthy, consensus_state: %{algorithm: :pbft, votes: %{}, view_number: 0, leader: nil, committed_steps: 0}, failed_workers: [], backup_checkpoints: %{}}, synchronization_barrier: nil, outer_optimizer_state: %{type: :nesterov_momentum, state: %{}, step_count: 0, accumulated_gradients: %{}}, inner_optimizer_state: %{type: :adamw, state: %{}, step_count: 0, accumulated_gradients: %{}}, step_counters: %{inner_step: 0, outer_step: 0, total_steps: 0, communication_rounds: 0}, coordination_service: nil}}, 5000)

#### 19. type preservation preserves tuples in JSON format (Object.SerializationTest)

- **Location**: object_serialization_test.exs:112
- **Category**: unknown
- **Error Type**: MatchError
- **Severity**: low
- **Message**:      ** (MatchError) no match of right hand side value: {:error, {:json_serialization_failed, %Protocol.UndefinedError{protocol: Enumerable, value: ~U[2025-06-12 05:36:02.202812Z], description: ""}}}

#### 20. partial serialization respects max_depth option (Object.SerializationTest)

- **Location**: object_serialization_test.exs:92
- **Category**: unknown
- **Error Type**: MatchError
- **Severity**: low
- **Message**:      ** (MatchError) no match of right hand side value: {:error, {:json_serialization_failed, %Protocol.UndefinedError{protocol: Enumerable, value: ~U[2025-06-12 05:36:02.254683Z], description: ""}}}

#### 21. DiLoCo algorithm components collects performance metrics (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:99
- **Category**: unknown
- **Error Type**: ExitError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.711.0>) exited in: GenServer.call(Object.DistributedRegistry, {:register_object, "test_trainer", %Object.DistributedTraining{object_id: "test_trainer", worker_id: "test_worker", global_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.314927Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, local_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.314934Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, optimizer_config: %{learning_rate: 0.001, epsilon: 1.0e-8, inner_optimizer: :adamw, outer_optimizer: :nesterov_momentum, momentum: 0.9, weight_decay: 0.01, beta1: 0.9, beta2: 0.999}, training_config: %{batch_size: 8, inner_steps: 10, outer_steps: 5, communication_frequency: 500, gradient_clipping: 1.0, fault_tolerance_threshold: 0.33, checkpoint_frequency: 10}, data_shard: %{shard_id: "73686172645f539cc2c5", data_path: "/tmp/training_data", total_samples: 10000, current_position: 0, preprocessing_config: %{}}, communication_state: %{sync_barrier_count: 0, pending_gradients: %{}, communication_overhead: 0.0, last_sync: ~U[2025-06-12 05:36:02.314937Z], bandwidth_usage: 0.0}, performance_metrics: %{throughput: 0.0, convergence_rate: 0.0, training_loss: 0.0, communication_efficiency: 1.0, validation_loss: 0.0, wall_clock_time: 0, compute_utilization: 0.0}, fault_tolerance_state: %{health_status: :healthy, consensus_state: %{algorithm: :pbft, votes: %{}, view_number: 0, leader: nil, committed_steps: 0}, failed_workers: [], backup_checkpoints: %{}}, synchronization_barrier: nil, outer_optimizer_state: %{type: :nesterov_momentum, state: %{}, step_count: 0, accumulated_gradients: %{}}, inner_optimizer_state: %{type: :adamw, state: %{}, step_count: 0, accumulated_gradients: %{}}, step_counters: %{inner_step: 0, outer_step: 0, total_steps: 0, communication_rounds: 0}, coordination_service: nil}}, 5000)

#### 22. communication and synchronization handles synchronization without coordination service (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:183
- **Category**: unknown
- **Error Type**: ExitError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.720.0>) exited in: GenServer.call(Object.DistributedRegistry, {:register_object, "64696c6f636f5f10a816bcedfa8406", %Object.DistributedTraining{object_id: "64696c6f636f5f10a816bcedfa8406", worker_id: "776f726b65725fcfa5cc03", global_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.315936Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, local_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.315937Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, optimizer_config: %{learning_rate: 0.001, epsilon: 1.0e-8, inner_optimizer: :adamw, outer_optimizer: :nesterov_momentum, momentum: 0.9, weight_decay: 0.01, beta1: 0.9, beta2: 0.999}, training_config: %{batch_size: 32, inner_steps: 500, outer_steps: 100, communication_frequency: 500, gradient_clipping: 1.0, fault_tolerance_threshold: 0.33, checkpoint_frequency: 10}, data_shard: %{shard_id: "73686172645f6409206f", data_path: "/tmp/training_data", total_samples: 10000, current_position: 0, preprocessing_config: %{}}, communication_state: %{sync_barrier_count: 0, pending_gradients: %{}, communication_overhead: 0.0, last_sync: ~U[2025-06-12 05:36:02.315938Z], bandwidth_usage: 0.0}, performance_metrics: %{throughput: 0.0, convergence_rate: 0.0, training_loss: 0.0, communication_efficiency: 1.0, validation_loss: 0.0, wall_clock_time: 0, compute_utilization: 0.0}, fault_tolerance_state: %{health_status: :healthy, consensus_state: %{algorithm: :pbft, votes: %{}, view_number: 0, leader: nil, committed_steps: 0}, failed_workers: [], backup_checkpoints: %{}}, synchronization_barrier: nil, outer_optimizer_state: %{type: :nesterov_momentum, state: %{}, step_count: 0, accumulated_gradients: %{}}, inner_optimizer_state: %{type: :adamw, state: %{}, step_count: 0, accumulated_gradients: %{}}, step_counters: %{inner_step: 0, outer_step: 0, total_steps: 0, communication_rounds: 0}, coordination_service: nil}}, 5000)

#### 23. communication and synchronization initializes communication state (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:175
- **Category**: unknown
- **Error Type**: ExitError
- **Severity**: low
- **Message**:      ** (EXIT from #PID<0.725.0>) exited in: GenServer.call(Object.DistributedRegistry, {:register_object, "64696c6f636f5f6134c6f6cbb1870d", %Object.DistributedTraining{object_id: "64696c6f636f5f6134c6f6cbb1870d", worker_id: "776f726b65725fec93117f", global_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.316365Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, local_model_state: %{metadata: %{last_updated: ~U[2025-06-12 05:36:02.316366Z], architecture: "transformer", parameter_count: 150000000, layer_count: 12}, parameters: %{}}, optimizer_config: %{learning_rate: 0.001, epsilon: 1.0e-8, inner_optimizer: :adamw, outer_optimizer: :nesterov_momentum, momentum: 0.9, weight_decay: 0.01, beta1: 0.9, beta2: 0.999}, training_config: %{batch_size: 32, inner_steps: 500, outer_steps: 100, communication_frequency: 500, gradient_clipping: 1.0, fault_tolerance_threshold: 0.33, checkpoint_frequency: 10}, data_shard: %{shard_id: "73686172645fb20eb51d", data_path: "/tmp/training_data", total_samples: 10000, current_position: 0, preprocessing_config: %{}}, communication_state: %{sync_barrier_count: 0, pending_gradients: %{}, communication_overhead: 0.0, last_sync: ~U[2025-06-12 05:36:02.316367Z], bandwidth_usage: 0.0}, performance_metrics: %{throughput: 0.0, convergence_rate: 0.0, training_loss: 0.0, communication_efficiency: 1.0, validation_loss: 0.0, wall_clock_time: 0, compute_utilization: 0.0}, fault_tolerance_state: %{health_status: :healthy, consensus_state: %{algorithm: :pbft, votes: %{}, view_number: 0, leader: nil, committed_steps: 0}, failed_workers: [], backup_checkpoints: %{}}, synchronization_barrier: nil, outer_optimizer_state: %{type: :nesterov_momentum, state: %{}, step_count: 0, accumulated_gradients: %{}}, inner_optimizer_state: %{type: :adamw, state: %{}, step_count: 0, accumulated_gradients: %{}}, step_counters: %{inner_step: 0, outer_step: 0, total_steps: 0, communication_rounds: 0}, coordination_service: nil}}, 5000)

#### 24. DiLoCo distributed training object creation creates a new distributed training object with default configuration (Object.DistributedTrainingTest)

- **Location**: object_distributed_training_test.exs:9
- **Category**: unknown
- **Error Type**: UnknownError
- **Severity**: low
- **Message**: Unknown error

#### 25. compression compresses serialized data (Object.SerializationTest)

- **Location**: object_serialization_test.exs:63
- **Category**: unknown
- **Error Type**: MatchError
- **Severity**: low
- **Message**:      ** (MatchError) no match of right hand side value: {:error, {:msgpack_decoding_failed, %Msgpax.UnpackError{reason: {:invalid_format, 218}}}}


## Architectural Concerns

Based on the failure patterns, the following architectural issues were identified:

1. **Service Startup Dependencies**: Multiple failures indicate improper service initialization order
2. **Race Conditions**: ETS table access before initialization suggests timing issues
3. **Type Safety**: Mathematical operations creating floats where integers expected
4. **Error Handling**: Insufficient defensive programming and error boundary patterns

## Next Steps

1. **Phase 1** (Critical): Fix race conditions and service dependencies
2. **Phase 2** (High): Address type safety and mathematical errors  
3. **Phase 3** (Medium): Improve timeout handling and performance
4. **Phase 4** (Low): Enhance overall error handling and test reliability

---
*Generated by automated test failure analysis*

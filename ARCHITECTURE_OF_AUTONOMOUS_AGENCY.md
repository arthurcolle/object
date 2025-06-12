# Architecture of Autonomous Agency (AAOS)

This document gives a *code-level* view of how the Autonomous AI Object System is organised.  It maps major conceptual layers to concrete Elixir modules in `lib/`.

---

## 1. Layered Overview

```
┌──────────────────────────────────────────────┐
│            Human & External World           │
└──────────────────────────────────────────────┘
            ▲                ▲
            │                │
┌────────────┴─────┐  ┌───────┴────────────┐
│  Interface Layer │  │  Action Layer      │
│  (HumanClient)   │  │  (ActuatorObject)  │
└────────┬─────────┘  └────────┬──────────┘
         │                     │
┌────────┴──────────────┐┌─────┴────────────┐
│   Cognitive Layer     ││  Sensing Layer   │
│   (AIAgent etc.)      ││  (SensorObject)  │
└────────┬──────────────┘└─────┬────────────┘
         │                     │
         ▼                     ▼
             Coordination Layer
┌──────────────────────────────────────────────┐
│CoordinatorObject ▸ CoordinationService ▸ ACL │
└──────────────────────────────────────────────┘
                       │
                       ▼
                Core Infrastructure
┌──────────────────────────────────────────────────────────────┐
│MessageRouter ▸ Mailboxes ▸ Persistence ▸ SchemaRegistry …    │
└──────────────────────────────────────────────────────────────┘
                       │
                       ▼
              Erlang/OTP Supervision Tree

```


## 2. Module Map

| Layer | Key Modules | Purpose |
|-------|-------------|---------|
| **Infrastructure** | `object_supervisor.ex`, `object_server.ex`, `object_mailbox.ex`, `object_message_router.ex`, `object_resource_manager.ex` | Fault tolerance, process isolation, message routing, resource quotas |
| **Persistence & Schema** | `object_schema_registry.ex`, `object_schema_evolution_manager.ex`, `lmstudio.persistence.*` | Versioned schemas, state snapshots, hot migrations |
| **Coordination** | `object_coordination_service.ex`, `object_interaction_patterns.ex` | Contract-net, auctions, consensus algorithms |
| **Cognitive / Reasoning** | `object_ai_reasoning.ex`, `object_dspy_bridge.ex`, `lmstudio.neural_architecture.*` | Chain-of-thought reasoning, LLM integration, vector attention |
| **Learning** | `oorl_framework.ex`, `object_exploration.ex`, `object_transfer_learning.ex` | Reinforcement learning, exploration bonuses, federated gradient sharing |
| **Meta** | `object_meta_dsl.ex`, `meta_object_schema.ex` | Self-modification, reflective DSL, object type system |
| **Sub-types** | `object_subtypes.ex` + nested modules | Concrete archetypes (AI Agent, Sensor, Actuator, Human Client, Coordinator) |
| **System Orchestration / Demos** | `object_system_orchestrator.ex`, `object_demo_runner.ex`, notebooks in `notebooks/` | Bootstraps multi-object ecosystems for experimentation |


## 3. Supervisory Structure

`Object.Supervisor` implements a **rest-for-one** tree with nested supervisors for infrastructure, core services, object management and integrations.  It supports:

• *Exponential Back-off* restart policy.

• *Circuit Breakers* to isolate flapping sub-systems.

• *Health Monitoring* via periodic telemetry (`:object, :health`).


## 4. Messaging & Back-Pressure

• **Mailbox (per object)** – GenServer holding a bounded queue.

• **Router** – Built on `GenStage`, exposes Demand-driven, Broadcast or Partition dispatchers.  Consumers register dynamic subscriptions, enabling location-transparent communication inside a cluster.

• **Dead Letter Queue** – Messages that exceed TTL are sent to a DLQ for analysis.  (See `object_message_router.ex`.)


## 5. Reasoning & LLM Bridge

`Object.AIReasoning` exposes *signatures* (structured prompts) consumed by `Dspy` and executed by an LLM provider (OpenAI, LMStudio, …).  Results return as structured maps, making the bridge deterministic at the protocol level even if the model itself is stochastic.


## 6. Data Flow Summary

```mermaid
flowchart LR
    subgraph Cluster
        direction LR
        Mailbox --> Router
        Router --msg--> Mailbox2[Mailbox]
    end

    Mailbox2 -->|experience| Learner(OORL)
    Learner -->|policy update| AIAgent
    AIAgent -->|invoke| AIReasoning
    AIReasoning -->|prompt| LLM[LLM Provider]
    LLM -->|structured result| AIAgent
```


## 7. Extensibility Points

1. **Custom Sub-Types** – Implement behaviour in a new module, embed in `Object.Subtypes`.
2. **New Message Protocols** – Add router strategy or ACL performatives.
3. **Alternative Learners** – `oorl_framework.ex` exposes a behaviour you can replace with, e.g., graph-based RL.


## 8. Code Footprint

Total **Elixir source**: ≈ *28k lines*  
Core AAOS modules: 25 files under `lib/`  
Tests: 40+ exhaustive property & chaos tests ensuring safety.


## 9. Relationship to External Deps

• **Dspy** – Prompt engineering and chain-of-thought orchestration.

• **LMStudio** – Local LLM runtime and multi-agent simulations.

• **GenStage / Telemetry / Jason** – Reactive streams, metrics and JSON encoding.


## 10. Conclusion

The architecture deliberately blends *Erlang/OTP* concurrency primitives with *symbolic* (meta-DSL) and *statistical* (RL & LLM) AI components.  The result is a modular, resilient substrate on which sophisticated autonomous societies can be safely prototyped and deployed.

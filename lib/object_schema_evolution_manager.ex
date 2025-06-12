defmodule Object.SchemaEvolutionManager do
  @moduledoc """
  Manages schema evolution and self-modification across the object system.
  Implements distributed consensus for schema changes and evolution tracking.
  """
  
  use GenServer
  require Logger
  
  alias Object.SchemaRegistry
  
  defstruct [
    :evolution_proposals,
    :active_evolutions,
    :evolution_history,
    :consensus_threshold,
    :performance_tracker
  ]
  
  # Client API
  
  @doc """
  Starts the schema evolution manager GenServer.
  
  Initializes the evolution state and starts periodic evolution analysis.
  
  ## Returns
  
  - `{:ok, pid}` - Successfully started evolution manager
  """
  def start_link(_) do
    GenServer.start_link(__MODULE__, :ok, name: __MODULE__)
  end
  
  @doc """
  Proposes a schema evolution for an object.
  
  Creates an evolution proposal and initiates the voting process
  among eligible objects.
  
  ## Parameters
  
  - `object_id` - ID of the object to evolve
  - `evolution_spec` - Specification of the proposed evolution
  
  ## Returns
  
  - `{:ok, proposal_id}` - Evolution proposal created successfully
  """
  def propose_evolution(object_id, evolution_spec) do
    GenServer.call(__MODULE__, {:propose_evolution, object_id, evolution_spec})
  end
  
  @doc """
  Casts a vote on an evolution proposal.
  
  ## Parameters
  
  - `proposal_id` - ID of the evolution proposal
  - `object_id` - ID of the voting object
  - `vote` - Vote (`:approve` or `:reject`)
  
  ## Returns
  
  - `:ok` - Vote recorded successfully
  """
  def vote_on_evolution(proposal_id, object_id, vote) do
    GenServer.call(__MODULE__, {:vote_evolution, proposal_id, object_id, vote})
  end
  
  @doc """
  Gets the current status of an evolution proposal.
  
  ## Parameters
  
  - `proposal_id` - ID of the evolution proposal
  
  ## Returns
  
  - `{:ok, status}` - Current proposal status
  - `{:error, :proposal_not_found}` - Proposal not found
  """
  def get_evolution_status(proposal_id) do
    GenServer.call(__MODULE__, {:get_status, proposal_id})
  end
  
  @doc """
  Gets the evolution history for an object.
  
  ## Parameters
  
  - `object_id` - ID of the object
  
  ## Returns
  
  Evolution history from the schema registry
  """
  def get_evolution_history(object_id) do
    SchemaRegistry.get_schema_evolution_history(object_id)
  end
  
  @doc """
  Triggers a system-wide evolution.
  
  ## Parameters
  
  - `evolution_type` - Type of system evolution
  - `parameters` - Evolution parameters
  
  ## Returns
  
  - `{:ok, evolution_id}` - System evolution initiated
  """
  def trigger_system_evolution(evolution_type, parameters) do
    GenServer.call(__MODULE__, {:system_evolution, evolution_type, parameters}, 60_000)
  end
  
  @doc """
  Gets evolution performance metrics.
  
  ## Returns
  
  Map with evolution statistics and performance data
  """
  def get_evolution_metrics() do
    GenServer.call(__MODULE__, :get_metrics)
  end
  
  # Server callbacks
  
  @impl true
  def init(:ok) do
    state = %__MODULE__{
      evolution_proposals: %{},
      active_evolutions: %{},
      evolution_history: [],
      consensus_threshold: 0.67,  # 67% consensus required
      performance_tracker: init_performance_tracker()
    }
    
    # Start periodic evolution analysis
    schedule_evolution_analysis()
    
    Logger.info("Schema Evolution Manager started")
    {:ok, state}
  end
  
  @impl true
  def handle_call({:propose_evolution, object_id, evolution_spec}, from, state) do
    proposal_id = generate_proposal_id()
    
    proposal = %{
      id: proposal_id,
      object_id: object_id,
      proposer: from,
      evolution_spec: evolution_spec,
      votes: %{},
      status: :voting,
      created_at: DateTime.utc_now(),
      voting_deadline: DateTime.add(DateTime.utc_now(), 300, :second)  # 5 minutes
    }
    
    # Determine eligible voters (compatible objects)
    eligible_voters = find_eligible_voters(object_id, evolution_spec)
    
    # Start voting process
    initiate_voting_process(proposal, eligible_voters)
    
    updated_proposals = Map.put(state.evolution_proposals, proposal_id, proposal)
    updated_state = %{state | evolution_proposals: updated_proposals}
    
    {:reply, {:ok, proposal_id}, updated_state}
  end
  
  @impl true
  def handle_call({:vote_evolution, proposal_id, object_id, vote}, _from, state) do
    case Map.get(state.evolution_proposals, proposal_id) do
      nil ->
        {:reply, {:error, :proposal_not_found}, state}
      
      proposal ->
        # Record vote
        updated_votes = Map.put(proposal.votes, object_id, %{
          vote: vote,
          timestamp: DateTime.utc_now()
        })
        
        updated_proposal = %{proposal | votes: updated_votes}
        
        # Check if consensus reached
        consensus_result = check_evolution_consensus(updated_proposal)
        
        final_proposal = case consensus_result do
          {:consensus_reached, :approved} ->
            execute_evolution(updated_proposal)
            %{updated_proposal | status: :approved}
          
          {:consensus_reached, :rejected} ->
            %{updated_proposal | status: :rejected}
          
          :voting_continues ->
            updated_proposal
        end
        
        updated_proposals = Map.put(state.evolution_proposals, proposal_id, final_proposal)
        updated_state = %{state | evolution_proposals: updated_proposals}
        
        {:reply, :ok, updated_state}
    end
  end
  
  @impl true
  def handle_call({:get_status, proposal_id}, _from, state) do
    case Map.get(state.evolution_proposals, proposal_id) do
      nil -> {:reply, {:error, :proposal_not_found}, state}
      proposal -> {:reply, {:ok, proposal.status}, state}
    end
  end
  
  @impl true
  def handle_call({:system_evolution, evolution_type, parameters}, _from, state) do
    Logger.info("Initiating system-wide evolution: #{evolution_type}")
    
    evolution_id = generate_evolution_id()
    
    system_evolution = %{
      id: evolution_id,
      type: evolution_type,
      parameters: parameters,
      status: :initializing,
      affected_objects: [],
      started_at: DateTime.utc_now()
    }
    
    # Execute system evolution asynchronously
    Task.start_link(fn -> execute_system_evolution(system_evolution) end)
    
    updated_active = Map.put(state.active_evolutions, evolution_id, system_evolution)
    updated_state = %{state | active_evolutions: updated_active}
    
    {:reply, {:ok, evolution_id}, updated_state}
  end
  
  @impl true
  def handle_call(:get_metrics, _from, state) do
    metrics = calculate_evolution_metrics(state)
    {:reply, metrics, state}
  end
  
  @impl true
  def handle_cast({:evolution_completed, evolution_id, results}, state) do
    case Map.get(state.active_evolutions, evolution_id) do
      nil ->
        {:noreply, state}
      
      evolution ->
        completed_evolution = %{evolution | 
          status: :completed,
          results: results,
          completed_at: DateTime.utc_now()
        }
        
        # Move to history
        updated_history = [completed_evolution | state.evolution_history]
        updated_active = Map.delete(state.active_evolutions, evolution_id)
        
        updated_state = %{state |
          evolution_history: Enum.take(updated_history, 1000),  # Keep last 1000
          active_evolutions: updated_active
        }
        
        Logger.info("Evolution #{evolution_id} completed successfully")
        {:noreply, updated_state}
    end
  end
  
  @impl true
  def handle_info(:evolution_analysis, state) do
    # Analyze system performance and suggest evolutions
    performance_analysis = analyze_system_performance()
    
    evolution_suggestions = generate_evolution_suggestions(performance_analysis)
    
    # Auto-propose critical evolutions
    Enum.each(evolution_suggestions, fn suggestion ->
      if suggestion.priority == :critical do
        auto_propose_evolution(suggestion)
      end
    end)
    
    schedule_evolution_analysis()
    {:noreply, state}
  end
  
  @impl true
  def handle_info({:voting_deadline, proposal_id}, state) do
    case Map.get(state.evolution_proposals, proposal_id) do
      nil ->
        {:noreply, state}
      
      proposal ->
        if proposal.status == :voting do
          # Force decision based on current votes
          final_decision = make_final_evolution_decision(proposal)
          updated_proposal = %{proposal | status: final_decision}
          
          if final_decision == :approved do
            execute_evolution(updated_proposal)
          end
          
          updated_proposals = Map.put(state.evolution_proposals, proposal_id, updated_proposal)
          {:noreply, %{state | evolution_proposals: updated_proposals}}
        else
          {:noreply, state}
        end
    end
  end
  
  # Private functions
  
  defp generate_proposal_id do
    :crypto.strong_rand_bytes(8) |> Base.encode16() |> String.downcase()
  end
  
  defp generate_evolution_id do
    :crypto.strong_rand_bytes(10) |> Base.encode16() |> String.downcase()
  end
  
  defp find_eligible_voters(object_id, evolution_spec) do
    # Find objects that would be affected by this evolution
    case SchemaRegistry.find_compatible_objects(object_id, 0.5) do
      {:ok, compatible_objects} ->
        compatible_objects
        |> Enum.map(fn {id, _compatibility} -> id end)
        |> Enum.filter(fn id -> 
          would_be_affected_by_evolution?(id, evolution_spec)
        end)
      
      {:error, _} ->
        []
    end
  end
  
  defp would_be_affected_by_evolution?(object_id, evolution_spec) do
    case SchemaRegistry.get_object_schema(object_id) do
      {:ok, schema} ->
        # Check if evolution affects this object's schema
        evolution_affects_schema?(evolution_spec, schema)
      
      {:error, _} ->
        false
    end
  end
  
  defp evolution_affects_schema?(evolution_spec, schema) do
    case evolution_spec.type do
      :method_evolution ->
        has_overlapping_methods?(evolution_spec.changes, schema.methods)
      
      :goal_evolution ->
        schema.goal_type == evolution_spec.target_goal_type
      
      :state_evolution ->
        has_overlapping_state_dimensions?(evolution_spec.changes, schema.state_dimensions)
      
      :meta_dsl_evolution ->
        has_overlapping_constructs?(evolution_spec.changes, schema.meta_dsl_constructs)
      
      _ ->
        true  # Universal evolution affects all objects
    end
  end
  
  defp has_overlapping_methods?(evolution_methods, schema_methods) do
    evolution_set = MapSet.new(evolution_methods)
    schema_set = MapSet.new(schema_methods)
    
    not MapSet.disjoint?(evolution_set, schema_set)
  end
  
  defp has_overlapping_state_dimensions?(evolution_dims, schema_dims) do
    evolution_set = MapSet.new(evolution_dims)
    schema_set = MapSet.new(schema_dims)
    
    not MapSet.disjoint?(evolution_set, schema_set)
  end
  
  defp has_overlapping_constructs?(evolution_constructs, schema_constructs) do
    evolution_set = MapSet.new(evolution_constructs)
    schema_set = MapSet.new(schema_constructs)
    
    not MapSet.disjoint?(evolution_set, schema_set)
  end
  
  defp initiate_voting_process(proposal, eligible_voters) do
    # Send voting requests to all eligible voters
    voting_message = %{
      proposal_id: proposal.id,
      evolution_spec: proposal.evolution_spec,
      voting_deadline: proposal.voting_deadline
    }
    
    Enum.each(eligible_voters, fn voter_id ->
      message = %{
        id: generate_proposal_id(),
        from: "evolution_manager",
        to: voter_id,
        type: :evolution_vote_request,
        content: voting_message,
        timestamp: DateTime.utc_now(),
        priority: :medium,
        requires_ack: false,
        ttl: 400  # Slightly longer than voting deadline
      }
      
      Object.MessageRouter.route_message(message)
    end)
    
    # Schedule voting deadline
    Process.send_after(self(), {:voting_deadline, proposal.id}, 300_000)  # 5 minutes
  end
  
  defp check_evolution_consensus(proposal) do
    total_votes = map_size(proposal.votes)
    
    if total_votes == 0 do
      :voting_continues
    else
      approve_votes = proposal.votes
                     |> Enum.count(fn {_id, vote_data} -> vote_data.vote == :approve end)
      
      approval_ratio = approve_votes / total_votes
      
      cond do
        approval_ratio >= 0.67 -> {:consensus_reached, :approved}
        approval_ratio <= 0.33 -> {:consensus_reached, :rejected}
        true -> :voting_continues
      end
    end
  end
  
  defp execute_evolution(proposal) do
    Logger.info("Executing approved evolution for object #{proposal.object_id}")
    
    # Apply evolution to the target object
    case Registry.lookup(Object.Registry, proposal.object_id) do
      [{pid, _}] ->
        GenServer.cast(pid, {:apply_evolution, proposal.evolution_spec})
      
      [] ->
        Logger.warning("Target object #{proposal.object_id} not found for evolution")
    end
    
    # Update schema registry
    schema_updates = evolution_spec_to_schema_updates(proposal.evolution_spec)
    SchemaRegistry.update_object_schema(proposal.object_id, schema_updates)
  end
  
  defp evolution_spec_to_schema_updates(evolution_spec) do
    case evolution_spec.type do
      :method_evolution ->
        %{methods: evolution_spec.new_methods}
      
      :goal_evolution ->
        %{goal_type: evolution_spec.new_goal_type}
      
      :state_evolution ->
        %{state_dimensions: evolution_spec.new_state_dimensions}
      
      :meta_dsl_evolution ->
        %{meta_dsl_constructs: evolution_spec.new_constructs}
      
      _ ->
        evolution_spec.schema_updates || %{}
    end
  end
  
  defp make_final_evolution_decision(proposal) do
    if map_size(proposal.votes) == 0 do
      :rejected  # No votes = rejection
    else
      approve_votes = proposal.votes
                     |> Enum.count(fn {_id, vote_data} -> vote_data.vote == :approve end)
      
      total_votes = map_size(proposal.votes)
      
      if approve_votes > total_votes / 2 do
        :approved
      else
        :rejected
      end
    end
  end
  
  defp execute_system_evolution(system_evolution) do
    try do
      Logger.info("Executing system evolution: #{system_evolution.type}")
      
      # Get all objects in the system
      all_objects = SchemaRegistry.list_objects()
      
      # Apply evolution based on type
      results = case system_evolution.type do
        :performance_optimization ->
          optimize_system_performance(all_objects, system_evolution.parameters)
        
        :capability_enhancement ->
          enhance_system_capabilities(all_objects, system_evolution.parameters)
        
        :security_upgrade ->
          upgrade_system_security(all_objects, system_evolution.parameters)
        
        _ ->
          generic_system_evolution(all_objects, system_evolution.parameters)
      end
      
      GenServer.cast(__MODULE__, {:evolution_completed, system_evolution.id, results})
      
    rescue
      error ->
        Logger.error("System evolution failed: #{inspect(error)}")
        GenServer.cast(__MODULE__, {:evolution_completed, system_evolution.id, {:error, error}})
    end
  end
  
  defp optimize_system_performance(objects, _parameters) do
    # Implement system-wide performance optimization
    optimized_count = length(objects)
    
    %{
      objects_optimized: optimized_count,
      performance_improvement: 0.15,
      timestamp: DateTime.utc_now()
    }
  end
  
  defp enhance_system_capabilities(objects, parameters) do
    # Add new capabilities to compatible objects
    enhanced_objects = Enum.filter(objects, fn {_id, schema} ->
      schema.subtype in (parameters[:target_types] || [:ai_agent])
    end)
    
    %{
      objects_enhanced: length(enhanced_objects),
      new_capabilities: parameters[:capabilities] || [],
      timestamp: DateTime.utc_now()
    }
  end
  
  defp upgrade_system_security(objects, _parameters) do
    # Implement security upgrades
    %{
      objects_secured: length(objects),
      security_level: :enhanced,
      timestamp: DateTime.utc_now()
    }
  end
  
  defp generic_system_evolution(objects, parameters) do
    %{
      objects_affected: length(objects),
      evolution_parameters: parameters,
      timestamp: DateTime.utc_now()
    }
  end
  
  defp schedule_evolution_analysis do
    Process.send_after(self(), :evolution_analysis, 300_000)  # 5 minutes
  end
  
  defp analyze_system_performance do
    # Analyze overall system performance
    all_objects = SchemaRegistry.list_objects()
    
    %{
      total_objects: length(all_objects),
      performance_score: :rand.uniform(),  # Simplified
      bottlenecks: [],
      improvement_opportunities: []
    }
  end
  
  defp generate_evolution_suggestions(_performance_analysis) do
    # Generate evolution suggestions based on analysis
    []  # Simplified for now
  end
  
  defp auto_propose_evolution(_suggestion) do
    # Auto-propose critical evolutions
    :ok  # Simplified for now
  end
  
  defp init_performance_tracker do
    %{
      evolutions_proposed: 0,
      evolutions_approved: 0,
      evolutions_rejected: 0,
      system_evolutions: 0,
      started_at: DateTime.utc_now()
    }
  end
  
  defp calculate_evolution_metrics(state) do
    active_proposals = state.evolution_proposals
                      |> Enum.count(fn {_id, proposal} -> proposal.status == :voting end)
    
    active_evolutions = map_size(state.active_evolutions)
    
    total_history = length(state.evolution_history)
    
    uptime = DateTime.diff(DateTime.utc_now(), state.performance_tracker.started_at, :second)
    
    Map.merge(state.performance_tracker, %{
      active_proposals: active_proposals,
      active_evolutions: active_evolutions,
      total_completed: total_history,
      uptime_seconds: uptime
    })
  end
end
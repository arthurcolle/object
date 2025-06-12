defmodule Object.QuantumMeasurement do
  @moduledoc """
  Quantum measurement simulation with probabilistic outcomes and real-time correlation.
  
  Implements the fundamental postulates of quantum mechanics:
  1. State collapse upon measurement
  2. Born rule for measurement probabilities  
  3. Instantaneous correlation for entangled systems
  4. Measurement basis transformations
  """

  alias Object.QuantumEntanglement.{Complex, QubitState, EntangledPair}
  
  # Helper function to generate unique IDs
  defp generate_id do
    :crypto.strong_rand_bytes(16) |> Base.encode16(case: :lower)
  end
  
  defmodule MeasurementResult do
    defstruct [
      :qubit_id, :measurement_basis, :outcome, :probability, 
      :timestamp, :correlation_id, :entangled_partner_outcome
    ]
    
    @type t :: %__MODULE__{
      qubit_id: String.t(),
      measurement_basis: :z | :x | :y | {:custom, float()},
      outcome: 0 | 1,
      probability: float(),
      timestamp: DateTime.t(),
      correlation_id: String.t() | nil,
      entangled_partner_outcome: 0 | 1 | nil
    }
  end

  @doc """
  Performs Z-basis measurement on a single qubit.
  Returns {measurement_result, collapsed_state}
  """
  def measure_z_basis(%QubitState{measured: true} = state) do
    # Already measured - return previous result
    result = %MeasurementResult{
      qubit_id: generate_id(),
      measurement_basis: :z,
      outcome: state.measurement_result,
      probability: 1.0,
      timestamp: DateTime.utc_now(),
      correlation_id: nil,
      entangled_partner_outcome: nil
    }
    {result, state}
  end
  
  def measure_z_basis(%QubitState{} = state) do
    prob_0 = QubitState.probability_0(state)
    
    # Generate quantum random outcome based on Born rule
    outcome = if :rand.uniform() < prob_0, do: 0, else: 1
    
    # Collapse state vector
    collapsed_state = case outcome do
      0 -> %QubitState{
        amplitude_0: Complex.new(1.0),
        amplitude_1: Complex.new(0.0),
        measured: true,
        measurement_result: 0
      }
      1 -> %QubitState{
        amplitude_0: Complex.new(0.0),
        amplitude_1: Complex.new(1.0),
        measured: true,
        measurement_result: 1
      }
    end
    
    result = %MeasurementResult{
      qubit_id: generate_id(),
      measurement_basis: :z,
      outcome: outcome,
      probability: if(outcome == 0, do: prob_0, else: 1.0 - prob_0),
      timestamp: DateTime.utc_now(),
      correlation_id: nil,
      entangled_partner_outcome: nil
    }
    
    {result, collapsed_state}
  end

  @doc """
  Performs X-basis (Hadamard rotated) measurement on a single qubit.
  Transforms |0⟩ → (|0⟩ + |1⟩)/√2, |1⟩ → (|0⟩ - |1⟩)/√2
  """
  def measure_x_basis(%QubitState{} = state) do
    # Apply Hadamard transformation before Z-measurement
    hadamard_state = apply_hadamard(state)
    {result, collapsed} = measure_z_basis(hadamard_state)
    
    # Transform result back to X-basis interpretation
    x_result = %MeasurementResult{result | measurement_basis: :x}
    {x_result, collapsed}
  end

  @doc """
  Performs Y-basis measurement on a single qubit.
  """
  def measure_y_basis(%QubitState{} = state) do
    # Apply Y-basis rotation: |0⟩ → (|0⟩ + i|1⟩)/√2, |1⟩ → (|0⟩ - i|1⟩)/√2
    y_rotated_state = apply_y_rotation(state)
    {result, collapsed} = measure_z_basis(y_rotated_state)
    
    y_result = %MeasurementResult{result | measurement_basis: :y}
    {y_result, collapsed}
  end

  @doc """
  Measures one qubit of an entangled pair, causing instantaneous correlation.
  Returns {local_result, partner_result, collapsed_pair_state}
  """
  def measure_entangled_pair(%EntangledPair{measured_qubits: measured} = pair, qubit_index, basis \\ :z) 
      when qubit_index in [0, 1] do
    
    if MapSet.member?(measured, qubit_index) do
      # Qubit already measured
      {:error, :already_measured}
    else
      case basis do
        :z -> measure_entangled_z_basis(pair, qubit_index)
        :x -> measure_entangled_x_basis(pair, qubit_index) 
        :y -> measure_entangled_y_basis(pair, qubit_index)
      end
    end
  end

  defp measure_entangled_z_basis(%EntangledPair{} = pair, qubit_index) do
    # Calculate joint measurement probabilities
    probs = %{
      "00" => EntangledPair.probability_00(pair),
      "01" => EntangledPair.probability_01(pair),
      "10" => EntangledPair.probability_10(pair),
      "11" => EntangledPair.probability_11(pair)
    }
    
    # Generate correlated measurement outcomes
    rand_val = :rand.uniform()
    
    {outcome_local, outcome_partner, joint_prob} = cond do
      rand_val < probs["00"] -> 
        {0, 0, probs["00"]}
      rand_val < probs["00"] + probs["01"] -> 
        {0, 1, probs["01"]}
      rand_val < probs["00"] + probs["01"] + probs["10"] -> 
        {1, 0, probs["10"]}
      true -> 
        {1, 1, probs["11"]}
    end
    
    # Adjust outcomes based on which qubit is measured first
    {local_outcome, partner_outcome} = if qubit_index == 0 do
      {outcome_local, outcome_partner}
    else
      {outcome_partner, outcome_local}
    end
    
    timestamp = DateTime.utc_now()
    correlation_id = generate_id()
    
    local_result = %MeasurementResult{
      qubit_id: "entangled_#{pair.entanglement_id}_qubit_#{qubit_index}",
      measurement_basis: :z,
      outcome: local_outcome,
      probability: joint_prob,
      timestamp: timestamp,
      correlation_id: correlation_id,
      entangled_partner_outcome: partner_outcome
    }
    
    partner_result = %MeasurementResult{
      qubit_id: "entangled_#{pair.entanglement_id}_qubit_#{1 - qubit_index}",
      measurement_basis: :z,
      outcome: partner_outcome, 
      probability: joint_prob,
      timestamp: timestamp,
      correlation_id: correlation_id,
      entangled_partner_outcome: local_outcome
    }
    
    # Collapse the entangled state
    collapsed_pair = collapse_entangled_state(pair, local_outcome, partner_outcome, qubit_index)
    
    {local_result, partner_result, collapsed_pair}
  end

  defp measure_entangled_x_basis(%EntangledPair{} = pair, qubit_index) do
    # Transform to X-basis before measurement
    x_transformed_pair = apply_hadamard_to_pair(pair, qubit_index)
    {local, partner, collapsed} = measure_entangled_z_basis(x_transformed_pair, qubit_index)
    
    # Update measurement basis in results
    local_x = %MeasurementResult{local | measurement_basis: :x}
    partner_x = %MeasurementResult{partner | measurement_basis: :x}
    
    {local_x, partner_x, collapsed}
  end

  defp measure_entangled_y_basis(%EntangledPair{} = pair, qubit_index) do
    # Transform to Y-basis before measurement
    y_transformed_pair = apply_y_rotation_to_pair(pair, qubit_index)
    {local, partner, collapsed} = measure_entangled_z_basis(y_transformed_pair, qubit_index)
    
    # Update measurement basis in results
    local_y = %MeasurementResult{local | measurement_basis: :y}
    partner_y = %MeasurementResult{partner | measurement_basis: :y}
    
    {local_y, partner_y, collapsed}
  end

  @doc """
  Calculate Bell inequality CHSH correlation for entangled pair measurements.
  
  CHSH inequality: |E(a,b) - E(a,b') + E(a',b) + E(a',b')| ≤ 2 (classical)
  Quantum mechanics can violate this up to 2√2 ≈ 2.828 (Tsirelson bound)
  """
  def calculate_bell_correlation(measurements) when is_list(measurements) do
    # Group measurements by correlation_id
    correlated_pairs = measurements
    |> Enum.filter(& &1.correlation_id)
    |> Enum.group_by(& &1.correlation_id)
    |> Enum.filter(fn {_id, results} -> length(results) == 2 end)
    
    if length(correlated_pairs) < 4 do
      {:error, :insufficient_measurements}
    else
      # Calculate E(a,b) = ⟨AB⟩ correlation function
      correlation_sum = correlated_pairs
      |> Enum.map(fn {_id, [result1, result2]} ->
        # Convert 0,1 outcomes to -1,+1 for correlation calculation
        outcome1 = if result1.outcome == 0, do: -1, else: 1
        outcome2 = if result2.outcome == 0, do: -1, else: 1
        outcome1 * outcome2
      end)
      |> Enum.sum()
      
      correlation = correlation_sum / length(correlated_pairs)
      
      %{
        correlation: correlation,
        measurement_pairs: length(correlated_pairs),
        bell_parameter: abs(correlation),
        violates_local_realism: abs(correlation) > 2.0,
        quantum_correlation_strength: abs(correlation) / 2.828  # Normalized to Tsirelson bound
      }
    end
  end

  # Private helper functions

  defp apply_hadamard(%QubitState{amplitude_0: a0, amplitude_1: a1}) do
    # H|0⟩ = (|0⟩ + |1⟩)/√2, H|1⟩ = (|0⟩ - |1⟩)/√2
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    
    new_amp_0 = Complex.add(
      Complex.scale(a0, inv_sqrt2),
      Complex.scale(a1, inv_sqrt2)
    )
    
    new_amp_1 = Complex.add(
      Complex.scale(a0, inv_sqrt2),
      Complex.scale(a1, -inv_sqrt2)
    )
    
    %QubitState{amplitude_0: new_amp_0, amplitude_1: new_amp_1, measured: false}
  end

  defp apply_y_rotation(%QubitState{amplitude_0: a0, amplitude_1: a1}) do
    # Y-basis rotation: complex transformation with imaginary components
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    
    new_amp_0 = Complex.add(
      Complex.scale(a0, inv_sqrt2),
      Complex.multiply(Complex.scale(a1, inv_sqrt2), Complex.new(0.0, 1.0))
    )
    
    new_amp_1 = Complex.add(
      Complex.scale(a0, inv_sqrt2),
      Complex.multiply(Complex.scale(a1, -inv_sqrt2), Complex.new(0.0, 1.0))
    )
    
    %QubitState{amplitude_0: new_amp_0, amplitude_1: new_amp_1, measured: false}
  end

  defp apply_hadamard_to_pair(%EntangledPair{} = pair, qubit_index) do
    # Apply Hadamard gate to specified qubit in entangled pair
    # This is a simplified implementation - full tensor product transformation needed
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    
    if qubit_index == 0 do
      # Apply H ⊗ I (Hadamard on first qubit, identity on second)
      %EntangledPair{pair |
        amplitude_00: Complex.scale(Complex.add(pair.amplitude_00, pair.amplitude_10), inv_sqrt2),
        amplitude_01: Complex.scale(Complex.add(pair.amplitude_01, pair.amplitude_11), inv_sqrt2),
        amplitude_10: Complex.scale(Complex.add(pair.amplitude_00, Complex.scale(pair.amplitude_10, -1)), inv_sqrt2),
        amplitude_11: Complex.scale(Complex.add(pair.amplitude_01, Complex.scale(pair.amplitude_11, -1)), inv_sqrt2)
      }
    else
      # Apply I ⊗ H (identity on first qubit, Hadamard on second)
      %EntangledPair{pair |
        amplitude_00: Complex.scale(Complex.add(pair.amplitude_00, pair.amplitude_01), inv_sqrt2),
        amplitude_01: Complex.scale(Complex.add(pair.amplitude_00, Complex.scale(pair.amplitude_01, -1)), inv_sqrt2),
        amplitude_10: Complex.scale(Complex.add(pair.amplitude_10, pair.amplitude_11), inv_sqrt2),
        amplitude_11: Complex.scale(Complex.add(pair.amplitude_10, Complex.scale(pair.amplitude_11, -1)), inv_sqrt2)
      }
    end
  end

  defp apply_y_rotation_to_pair(%EntangledPair{} = pair, qubit_index) do
    # Similar to Hadamard but with Y-gate transformation
    # Simplified implementation - would need full tensor algebra in production
    inv_sqrt2 = 1.0 / :math.sqrt(2)
    i = Complex.new(0.0, 1.0)
    
    if qubit_index == 0 do
      %EntangledPair{pair |
        amplitude_00: Complex.scale(Complex.add(pair.amplitude_00, Complex.multiply(pair.amplitude_10, i)), inv_sqrt2),
        amplitude_01: Complex.scale(Complex.add(pair.amplitude_01, Complex.multiply(pair.amplitude_11, i)), inv_sqrt2),
        amplitude_10: Complex.scale(Complex.add(pair.amplitude_00, Complex.multiply(pair.amplitude_10, Complex.scale(i, -1))), inv_sqrt2),
        amplitude_11: Complex.scale(Complex.add(pair.amplitude_01, Complex.multiply(pair.amplitude_11, Complex.scale(i, -1))), inv_sqrt2)
      }
    else
      %EntangledPair{pair |
        amplitude_00: Complex.scale(Complex.add(pair.amplitude_00, Complex.multiply(pair.amplitude_01, i)), inv_sqrt2),
        amplitude_01: Complex.scale(Complex.add(pair.amplitude_00, Complex.multiply(pair.amplitude_01, Complex.scale(i, -1))), inv_sqrt2),
        amplitude_10: Complex.scale(Complex.add(pair.amplitude_10, Complex.multiply(pair.amplitude_11, i)), inv_sqrt2),
        amplitude_11: Complex.scale(Complex.add(pair.amplitude_10, Complex.multiply(pair.amplitude_11, Complex.scale(i, -1))), inv_sqrt2)
      }
    end
  end

  defp collapse_entangled_state(%EntangledPair{} = pair, outcome_0, outcome_1, measured_qubit) do
    # Collapse to definite computational basis state
    {new_00, new_01, new_10, new_11} = case {outcome_0, outcome_1} do
      {0, 0} -> {Complex.new(1.0), Complex.new(0.0), Complex.new(0.0), Complex.new(0.0)}
      {0, 1} -> {Complex.new(0.0), Complex.new(1.0), Complex.new(0.0), Complex.new(0.0)}
      {1, 0} -> {Complex.new(0.0), Complex.new(0.0), Complex.new(1.0), Complex.new(0.0)}
      {1, 1} -> {Complex.new(0.0), Complex.new(0.0), Complex.new(0.0), Complex.new(1.0)}
    end
    
    updated_stats = %{
      measurements: pair.correlation_stats.measurements + 1,
      correlations: [
        %{outcome_0: outcome_0, outcome_1: outcome_1, timestamp: DateTime.utc_now()} | 
        pair.correlation_stats.correlations
      ]
    }
    
    %EntangledPair{pair |
      amplitude_00: new_00,
      amplitude_01: new_01, 
      amplitude_10: new_10,
      amplitude_11: new_11,
      measured_qubits: MapSet.put(pair.measured_qubits, measured_qubit),
      correlation_stats: updated_stats
    }
  end
end
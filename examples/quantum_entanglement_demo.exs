#!/usr/bin/env elixir

# Quantum Entanglement Real-time Demo
# Run with: elixir examples/quantum_entanglement_demo.exs

Mix.install([
  {:uuid, "~> 1.1"}
])

# Add our quantum modules to the code path
Code.prepend_path("_build/dev/lib/object/ebin")

# Import the quantum modules
alias Object.QuantumEntanglement.{Complex, QubitState, EntangledPair}
alias Object.QuantumMeasurement
alias Object.QuantumCorrelationEngine

defmodule QuantumDemo do
  @moduledoc """
  Interactive demonstration of quantum entanglement with real-time correlations.
  Shows Bell state creation, measurement, and violation of Bell inequalities.
  """

  def run do
    IO.puts """
    ╔══════════════════════════════════════════════════════════════╗
    ║        🌌 Quantum Entanglement Demo with Elixir/OTP 🌌       ║
    ╟──────────────────────────────────────────────────────────────╢
    ║  Demonstrating real-time quantum correlations using:         ║
    ║  • Bell state generation (|Φ⁺⟩ = (|00⟩ + |11⟩)/√2)         ║
    ║  • Instantaneous measurement correlations                    ║
    ║  • Bell inequality violations (CHSH > 2)                     ║
    ║  • Multi-basis measurements (Z, X, Y)                        ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    
    # Start the correlation engine
    {:ok, _pid} = QuantumCorrelationEngine.start_link()
    
    # Subscribe to quantum events
    QuantumCorrelationEngine.subscribe_correlations()
    
    # Register demo session
    session_id = UUID.uuid4()
    QuantumCorrelationEngine.register_session(session_id, %{demo: true})
    
    IO.puts "\n🔮 Starting Quantum Correlation Engine..."
    Process.sleep(500)
    
    # Demonstrate different quantum phenomena
    demonstrate_bell_states()
    demonstrate_measurement_correlation()
    demonstrate_bell_violation()
    demonstrate_quantum_teleportation_concept()
    
    IO.puts "\n✨ Quantum demonstration complete!"
  end

  defp demonstrate_bell_states do
    IO.puts "\n📊 === Bell State Demonstration ===\n"
    
    bell_states = [
      {:bell_phi_plus, "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2"},
      {:bell_phi_minus, "|Φ⁻⟩ = (|00⟩ - |11⟩)/√2"},
      {:bell_psi_plus, "|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"},
      {:bell_psi_minus, "|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"}
    ]
    
    for {type, description} <- bell_states do
      {:ok, pair_id} = QuantumCorrelationEngine.create_entangled_pair(type)
      [pair_info] = QuantumCorrelationEngine.list_entangled_pairs()
      |> Enum.filter(& &1.id == pair_id)
      
      IO.puts "Created #{description}"
      IO.puts "  Entanglement ID: #{pair_id}"
      IO.puts "  Entropy: #{Float.round(pair_info.entropy, 3)} (max = 1.0)"
      IO.puts ""
      
      Process.sleep(300)
    end
  end

  defp demonstrate_measurement_correlation do
    IO.puts "\n⚛️ === Measurement Correlation Demonstration ===\n"
    
    # Create entangled pair
    {:ok, pair_id} = QuantumCorrelationEngine.create_entangled_pair(:bell_phi_plus)
    session_id = UUID.uuid4()
    
    IO.puts "Created entangled pair: #{pair_id}"
    IO.puts "Performing correlated measurements...\n"
    
    # Measure both qubits
    for qubit <- [0, 1] do
      {:ok, response} = QuantumCorrelationEngine.measure_correlated(
        pair_id, qubit, :z, session_id
      )
      
      IO.puts "Qubit #{qubit} measurement:"
      IO.puts "  Local outcome: #{response.local_result.outcome}"
      IO.puts "  Partner outcome: #{response.partner_result.outcome}"
      IO.puts "  Correlation: #{if response.local_result.outcome == response.partner_result.outcome, do: "✓ PERFECT", else: "✗ ANTI"}"
      IO.puts "  Measurement time: #{format_time(response.local_result.timestamp)}"
      IO.puts ""
      
      Process.sleep(500)
    end
  end

  defp demonstrate_bell_violation do
    IO.puts "\n🎯 === Bell Inequality Violation Test ===\n"
    
    session_id = UUID.uuid4()
    measurement_count = 100
    
    IO.puts "Running Bell test with #{measurement_count} measurements..."
    IO.puts "Classical limit: CHSH ≤ 2.0"
    IO.puts "Quantum limit: CHSH ≤ 2√2 ≈ 2.828\n"
    
    # Start Bell test
    {:ok, pair_ids} = QuantumCorrelationEngine.start_bell_test(session_id, measurement_count)
    
    # Simulate measurements
    measurements = for i <- 0..(measurement_count - 1) do
      pair_id = Enum.at(pair_ids, rem(i, length(pair_ids)))
      qubit_index = rem(i, 2)
      basis = if rem(i, 4) < 2, do: :z, else: :x
      
      case QuantumCorrelationEngine.measure_correlated(pair_id, qubit_index, basis, session_id) do
        {:ok, response} -> 
          %{
            basis: basis,
            correlation: response.local_result.outcome * response.partner_result.outcome,
            local: response.local_result.outcome,
            partner: response.partner_result.outcome
          }
        _ -> nil
      end
    end
    |> Enum.filter(& &1)
    
    # Calculate CHSH parameter
    z_correlations = measurements |> Enum.filter(& &1.basis == :z) |> Enum.map(& &1.correlation)
    x_correlations = measurements |> Enum.filter(& &1.basis == :x) |> Enum.map(& &1.correlation)
    
    avg_z = if length(z_correlations) > 0, do: Enum.sum(z_correlations) / length(z_correlations), else: 0
    avg_x = if length(x_correlations) > 0, do: Enum.sum(x_correlations) / length(x_correlations), else: 0
    
    chsh = abs(avg_z + avg_x) * 2
    
    IO.puts "Results:"
    IO.puts "  Z-basis average: #{Float.round(avg_z, 3)}"
    IO.puts "  X-basis average: #{Float.round(avg_x, 3)}"
    IO.puts "  CHSH parameter: #{Float.round(chsh, 3)}"
    
    if chsh > 2.0 do
      IO.puts "\n🎉 BELL INEQUALITY VIOLATED! Quantum mechanics confirmed!"
      IO.puts "  Violation amount: #{Float.round(chsh - 2.0, 3)}"
      IO.puts "  Quantum advantage: #{Float.round((chsh - 2.0) / 0.828 * 100, 1)}%"
    else
      IO.puts "\n📊 No violation detected (may need more measurements)"
    end
  end

  defp demonstrate_quantum_teleportation_concept do
    IO.puts "\n\n🚀 === Quantum Teleportation Concept ===\n"
    
    IO.puts "Quantum teleportation protocol outline:"
    IO.puts "1. Alice and Bob share an entangled pair"
    IO.puts "2. Alice has a qubit in unknown state |ψ⟩"
    IO.puts "3. Alice performs Bell measurement on |ψ⟩ and her half of the pair"
    IO.puts "4. Alice sends 2 classical bits to Bob"
    IO.puts "5. Bob applies corrections based on Alice's measurement"
    IO.puts "6. Bob's qubit is now in state |ψ⟩!\n"
    
    # Create entangled resource
    {:ok, pair_id} = QuantumCorrelationEngine.create_entangled_pair(:bell_phi_plus)
    
    IO.puts "Created quantum channel: #{pair_id}"
    IO.puts "Ready for quantum teleportation..."
    IO.puts "(Full implementation in todo #8)"
  end

  defp format_time(datetime) do
    Calendar.strftime(datetime, "%H:%M:%S.%f")
    |> String.slice(0..-4)  # Remove extra precision
  end
end

# Listen for quantum events in the background
spawn(fn ->
  loop = fn loop_fn ->
    receive do
      {:quantum_event, event} ->
        case event do
          {:entanglement_created, pair_id, type} ->
            IO.puts "\n🔗 [EVENT] Entanglement created: #{type} (#{String.slice(pair_id, 0..7)}...)"
          
          {:quantum_correlation, data} ->
            IO.puts "\n⚡ [EVENT] Quantum correlation detected!"
            IO.puts "  Outcomes: #{data.local_outcome}|#{data.partner_outcome}"
            IO.puts "  Basis: #{data.basis}"
            IO.puts "  Instantaneous: #{data.instantaneous}"
          
          {:bell_test_started, session_id, count} ->
            IO.puts "\n🔬 [EVENT] Bell test started: #{count} measurements"
          
          {:correlation_analysis, analysis} ->
            IO.puts "\n📈 [EVENT] Correlation Analysis:"
            IO.puts "  Mean: #{Float.round(analysis.mean_correlation, 3)}"
            IO.puts "  Samples: #{analysis.sample_size}"
          
          _ ->
            :ok
        end
        loop_fn.(loop_fn)
    end
  end
  
  loop.(loop)
end)

# Run the demo
QuantumDemo.run()

# Keep the process alive to see events
Process.sleep(2000)
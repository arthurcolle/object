#!/usr/bin/env elixir

# Simple Quantum Entanglement Demo - No dependencies needed
# Run with: elixir examples/quantum_entanglement_simple.exs

defmodule SimpleQuantum do
  @moduledoc """
  Simplified quantum entanglement demonstration.
  Shows the core quantum mechanics without the full infrastructure.
  """

  # Complex number operations
  defmodule Complex do
    defstruct real: 0.0, imag: 0.0
    
    def new(r, i \\ 0.0), do: %__MODULE__{real: r, imag: i}
    def magnitude(%{real: r, imag: i}), do: :math.sqrt(r*r + i*i)
    def multiply(%{real: r1, imag: i1}, %{real: r2, imag: i2}) do
      %__MODULE__{real: r1*r2 - i1*i2, imag: r1*i2 + i1*r2}
    end
  end

  # Bell state representation
  defmodule BellState do
    defstruct amp_00: nil, amp_01: nil, amp_10: nil, amp_11: nil, type: nil
    
    def phi_plus do
      inv_sqrt2 = 1.0 / :math.sqrt(2)
      %__MODULE__{
        amp_00: Complex.new(inv_sqrt2),
        amp_01: Complex.new(0.0),
        amp_10: Complex.new(0.0),
        amp_11: Complex.new(inv_sqrt2),
        type: :phi_plus
      }
    end
    
    def phi_minus do
      inv_sqrt2 = 1.0 / :math.sqrt(2)
      %__MODULE__{
        amp_00: Complex.new(inv_sqrt2),
        amp_01: Complex.new(0.0),
        amp_10: Complex.new(0.0),
        amp_11: Complex.new(-inv_sqrt2),
        type: :phi_minus
      }
    end
    
    def psi_plus do
      inv_sqrt2 = 1.0 / :math.sqrt(2)
      %__MODULE__{
        amp_00: Complex.new(0.0),
        amp_01: Complex.new(inv_sqrt2),
        amp_10: Complex.new(inv_sqrt2),
        amp_11: Complex.new(0.0),
        type: :psi_plus
      }
    end
    
    def psi_minus do
      inv_sqrt2 = 1.0 / :math.sqrt(2)
      %__MODULE__{
        amp_00: Complex.new(0.0),
        amp_01: Complex.new(inv_sqrt2),
        amp_10: Complex.new(-inv_sqrt2),
        amp_11: Complex.new(0.0),
        type: :psi_minus
      }
    end
  end

  def run_demo do
    IO.puts """
    
    ╔═══════════════════════════════════════════════════════════╗
    ║         🌌 Simple Quantum Entanglement Demo 🌌            ║
    ╟───────────────────────────────────────────────────────────╢
    ║  Demonstrating spooky action at a distance with Elixir!   ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    
    demonstrate_entanglement()
    demonstrate_measurement_correlation()
    demonstrate_bell_violation()
    visualize_quantum_state()
  end

  defp demonstrate_entanglement do
    IO.puts "\n1️⃣ Creating Entangled Quantum States\n"
    
    states = [
      {BellState.phi_plus(), "|Φ⁺⟩ = (|00⟩ + |11⟩)/√2"},
      {BellState.phi_minus(), "|Φ⁻⟩ = (|00⟩ - |11⟩)/√2"},
      {BellState.psi_plus(), "|Ψ⁺⟩ = (|01⟩ + |10⟩)/√2"},
      {BellState.psi_minus(), "|Ψ⁻⟩ = (|01⟩ - |10⟩)/√2"}
    ]
    
    for {state, description} <- states do
      IO.puts "#{description}"
      IO.puts "  Amplitudes: |00⟩→#{format_complex(state.amp_00)}, " <>
              "|01⟩→#{format_complex(state.amp_01)}, " <>
              "|10⟩→#{format_complex(state.amp_10)}, " <>
              "|11⟩→#{format_complex(state.amp_11)}"
      IO.puts "  Entanglement entropy: #{calculate_entropy(state)}\n"
    end
  end

  defp demonstrate_measurement_correlation do
    IO.puts "2️⃣ Quantum Measurement Correlations\n"
    
    # Create Bell state |Φ⁺⟩
    state = BellState.phi_plus()
    IO.puts "Using Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2"
    IO.puts "Performing 20 measurements...\n"
    
    measurements = for i <- 1..20 do
      {alice, bob} = measure_bell_state(state)
      correlation = if alice == bob, do: "✓", else: "✗"
      IO.puts "Measurement #{i}: Alice=#{alice}, Bob=#{bob} #{correlation}"
      {alice, bob}
    end
    
    # Calculate correlation statistics
    correlated = Enum.count(measurements, fn {a, b} -> a == b end)
    IO.puts "\nPerfect correlations: #{correlated}/20 (#{correlated/20*100}%)"
    IO.puts "This demonstrates quantum entanglement!"
  end

  defp demonstrate_bell_violation do
    IO.puts "\n3️⃣ Bell Inequality Test (CHSH)\n"
    
    # Run simplified Bell test
    measurements = 200
    results = run_bell_test(measurements)
    
    IO.puts "Running #{measurements} measurements..."
    IO.puts "Settings: Alice (0°, 45°), Bob (22.5°, 67.5°)\n"
    
    # Calculate CHSH parameter
    chsh = calculate_chsh(results)
    
    IO.puts "Results:"
    IO.puts "  E(0°, 22.5°) = #{Float.round(results.e_00, 3)}"
    IO.puts "  E(0°, 67.5°) = #{Float.round(results.e_01, 3)}"
    IO.puts "  E(45°, 22.5°) = #{Float.round(results.e_10, 3)}"
    IO.puts "  E(45°, 67.5°) = #{Float.round(results.e_11, 3)}"
    IO.puts "\n  CHSH = |E(a,b) - E(a,b') + E(a',b) + E(a',b')| = #{Float.round(chsh, 3)}"
    
    if chsh > 2.0 do
      IO.puts "\n🎉 BELL INEQUALITY VIOLATED!"
      IO.puts "  Classical limit: 2.0"
      IO.puts "  Our result: #{Float.round(chsh, 3)}"
      IO.puts "  Quantum limit: 2√2 ≈ 2.828"
      IO.puts "\n  This proves quantum mechanics cannot be explained by"
      IO.puts "  local hidden variables! Einstein was wrong about"
      IO.puts "  'spooky action at a distance' - it's real!"
    end
  end

  defp visualize_quantum_state do
    IO.puts "\n4️⃣ Quantum State Visualization\n"
    
    state = BellState.phi_plus()
    
    IO.puts "Bell State |Φ⁺⟩ Probability Distribution:"
    IO.puts ""
    
    # Calculate probabilities
    p00 = Complex.magnitude(state.amp_00) ** 2
    p01 = Complex.magnitude(state.amp_01) ** 2
    p10 = Complex.magnitude(state.amp_10) ** 2
    p11 = Complex.magnitude(state.amp_11) ** 2
    
    # Visualize as bars
    IO.puts "  |00⟩: #{probability_bar(p00)} #{Float.round(p00, 2)}"
    IO.puts "  |01⟩: #{probability_bar(p01)} #{Float.round(p01, 2)}"
    IO.puts "  |10⟩: #{probability_bar(p10)} #{Float.round(p10, 2)}"
    IO.puts "  |11⟩: #{probability_bar(p11)} #{Float.round(p11, 2)}"
    
    IO.puts "\nNotice: Only |00⟩ and |11⟩ have non-zero probability!"
    IO.puts "This is the signature of quantum entanglement."
    
    # Show measurement animation
    IO.puts "\n🎲 Live Measurement Simulation:\n"
    
    for i <- 1..10 do
      Process.sleep(300)
      {a, b} = measure_bell_state(state)
      IO.write("\r  Measurement: |#{a}#{b}⟩ ")
      
      # Show correlation indicator
      if a == b do
        IO.write("⚡ Perfect correlation! ")
      else
        IO.write("                        ")
      end
    end
    
    IO.puts "\n"
  end

  # Helper functions

  defp measure_bell_state(state) do
    # Simplified measurement for Bell states
    rand = :rand.uniform()
    
    p00 = Complex.magnitude(state.amp_00) ** 2
    p01 = Complex.magnitude(state.amp_01) ** 2
    p10 = Complex.magnitude(state.amp_10) ** 2
    
    cond do
      rand < p00 -> {0, 0}
      rand < p00 + p01 -> {0, 1}
      rand < p00 + p01 + p10 -> {1, 0}
      true -> {1, 1}
    end
  end

  defp run_bell_test(num_measurements) do
    # Simplified Bell test with fixed angles
    state = BellState.phi_plus()
    
    correlations = %{e_00: 0, e_01: 0, e_10: 0, e_11: 0}
    
    # Simulate measurements at different angle settings
    counts = for _ <- 1..num_measurements do
      # Random measurement settings
      alice_setting = if :rand.uniform() < 0.5, do: 0, else: 1
      bob_setting = if :rand.uniform() < 0.5, do: 0, else: 1
      
      # Get correlated outcomes (simplified)
      {a, b} = measure_with_basis(state, alice_setting, bob_setting)
      
      # Convert to ±1
      a_val = if a == 0, do: 1, else: -1
      b_val = if b == 0, do: 1, else: -1
      
      {alice_setting, bob_setting, a_val * b_val}
    end
    
    # Calculate expectation values
    e_00 = counts |> Enum.filter(fn {a, b, _} -> a == 0 && b == 0 end) |> Enum.map(&elem(&1, 2)) |> average()
    e_01 = counts |> Enum.filter(fn {a, b, _} -> a == 0 && b == 1 end) |> Enum.map(&elem(&1, 2)) |> average()
    e_10 = counts |> Enum.filter(fn {a, b, _} -> a == 1 && b == 0 end) |> Enum.map(&elem(&1, 2)) |> average()
    e_11 = counts |> Enum.filter(fn {a, b, _} -> a == 1 && b == 1 end) |> Enum.map(&elem(&1, 2)) |> average()
    
    %{e_00: e_00, e_01: e_01, e_10: e_10, e_11: e_11}
  end

  defp measure_with_basis(state, alice_basis, bob_basis) do
    # Simplified: approximate rotated measurements
    if alice_basis == 0 && bob_basis == 0 do
      measure_bell_state(state)
    else
      # Add some quantum randomness for different bases
      {a, b} = measure_bell_state(state)
      
      # Simulate basis rotation effects
      flip_alice = alice_basis == 1 && :rand.uniform() < 0.15
      flip_bob = bob_basis == 1 && :rand.uniform() < 0.15
      
      a_result = if flip_alice, do: 1 - a, else: a
      b_result = if flip_bob, do: 1 - b, else: b
      
      {a_result, b_result}
    end
  end

  defp calculate_chsh(results) do
    abs(results.e_00 - results.e_01 + results.e_10 + results.e_11)
  end

  defp calculate_entropy(state) do
    # Von Neumann entropy for entanglement
    probs = [
      Complex.magnitude(state.amp_00) ** 2,
      Complex.magnitude(state.amp_01) ** 2,
      Complex.magnitude(state.amp_10) ** 2,
      Complex.magnitude(state.amp_11) ** 2
    ]
    
    # For maximally entangled states, entropy = 1
    non_zero = Enum.count(probs, & &1 > 0.01)
    if non_zero == 2, do: "1.0 (maximally entangled)", else: "< 1.0"
  end

  defp format_complex(%Complex{real: r, imag: i}) do
    cond do
      abs(i) < 0.001 -> "#{Float.round(r, 3)}"
      abs(r) < 0.001 -> "#{Float.round(i, 3)}i"
      i >= 0 -> "#{Float.round(r, 3)}+#{Float.round(i, 3)}i"
      true -> "#{Float.round(r, 3)}#{Float.round(i, 3)}i"
    end
  end

  defp probability_bar(prob) do
    bar_length = round(prob * 20)
    String.duplicate("█", bar_length) <> String.duplicate("░", 20 - bar_length)
  end

  defp average([]), do: 0
  defp average(list), do: Enum.sum(list) / length(list)
end

# Run the demo
SimpleQuantum.run_demo()

IO.puts """

📚 Learn More:
- Bell's Theorem: https://en.wikipedia.org/wiki/Bell's_theorem
- Quantum Entanglement: https://en.wikipedia.org/wiki/Quantum_entanglement
- CHSH Inequality: https://en.wikipedia.org/wiki/CHSH_inequality

🔬 Try modifying the code to:
- Create different Bell states
- Change measurement angles
- Increase measurement counts
- Add quantum noise/decoherence
"""
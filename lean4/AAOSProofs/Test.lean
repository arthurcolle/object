/-
  Simple test file to verify LEAN4 setup
-/

import Mathlib.Data.Real.Basic

namespace AAOSProofs

theorem simple_test : 2 + 2 = 4 := by
  norm_num

theorem convergence_example : ∃ (L : ℝ), ∀ ε > 0, ∃ N, ∀ n ≥ N, |1 / n - L| < ε := by
  use 0
  intro ε hε
  sorry

end AAOSProofs
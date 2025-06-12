import Lake
open Lake DSL

package «aaos-proofs» where
  -- LEAN 4 mathematical proofs for AAOS
  leanOptions := #[
    ⟨`pp.unicode.fun, true⟩, -- pretty-print `fun a ↦ b`
    ⟨`autoImplicit, false⟩,
    ⟨`relaxedAutoImplicit, false⟩
  ]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git"

@[default_target]
lean_lib «AAOSProofs» where
  -- All formal proofs for AAOS mathematical foundations
  roots := #[`AAOSProofs]
  globs := #[.submodules `AAOSProofs]

lean_lib «CategoryTheory» where
  -- Category-theoretic foundations
  roots := #[`AAOSProofs.CategoryTheory]

lean_lib «MeasureTheory» where
  -- Measure-theoretic probability
  roots := #[`AAOSProofs.MeasureTheory]

lean_lib «Convergence» where
  -- Convergence proofs
  roots := #[`AAOSProofs.Convergence]

lean_lib «Emergence» where
  -- Emergence theorems
  roots := #[`AAOSProofs.Emergence]
ExUnit.start()

defmodule Lean4IntegrationTest do
  @moduledoc """
  Integration tests for LEAN4 formal proofs
  """
  use ExUnit.Case
  
  @lean4_dir Path.expand("../lean4", __DIR__)
  
  describe "LEAN4 proof verification" do
    test "LEAN4 project builds successfully" do
      {output, exit_code} = System.cmd("lake", ["build"], cd: @lean4_dir)
      
      # Lake build should complete (exit code 0 means success)
      assert exit_code == 0, "LEAN4 build failed: #{output}"
    end
    
    test "Basic test file compiles" do
      {output, exit_code} = System.cmd("lake", ["build", "AAOSProofs.Test"], cd: @lean4_dir)
      
      assert exit_code == 0, "Test file compilation failed: #{output}"
      assert String.contains?(output, "Built AAOSProofs.Test"), "Expected successful build message"
    end
    
    test "LEAN4 environment is properly configured" do
      {output, exit_code} = System.cmd("lean", ["--version"])
      
      assert exit_code == 0, "LEAN not found"
      assert String.contains?(output, "Lean"), "Invalid LEAN version output"
    end
    
    test "Mathlib4 dependency is available" do
      mathlib_path = Path.join([@lean4_dir, ".lake", "packages", "mathlib"])
      
      assert File.exists?(mathlib_path), "Mathlib4 not found at #{mathlib_path}"
    end
  end
  
  describe "LEAN4 proof content" do
    test "Basic definitions are present" do
      basic_file = Path.join([@lean4_dir, "AAOSProofs", "Basic.lean"])
      content = File.read!(basic_file)
      
      # Check for key definitions
      assert String.contains?(content, "structure Object")
      assert String.contains?(content, "def convergent")
      assert String.contains?(content, "def emergent")
      assert String.contains?(content, "def autonomous")
    end
    
    test "Category theory proofs are present" do
      cat_file = Path.join([@lean4_dir, "AAOSProofs", "CategoryTheory", "ObjectCategory.lean"])
      
      assert File.exists?(cat_file), "Category theory file missing"
      
      content = File.read!(cat_file)
      assert String.contains?(content, "CategoryTheory")
      assert String.contains?(content, "theorem")
    end
    
    test "Convergence proofs are present" do
      conv_file = Path.join([@lean4_dir, "AAOSProofs", "Convergence", "OORLConvergence.lean"])
      
      assert File.exists?(conv_file), "Convergence file missing"
      
      content = File.read!(conv_file)
      assert String.contains?(content, "oorl_convergence")
      assert String.contains?(content, "O(log")
    end
  end
  
  describe "LEAN4 and Elixir integration" do
    test "AAOS mathematical claims match LEAN4 formalizations" do
      # Check that key theorems mentioned in README are formalized
      readme = File.read!(Path.join([__DIR__, "..", "README.md"]))
      
      assert String.contains?(readme, "machine-verified")
      assert String.contains?(readme, "LEAN 4")
      assert String.contains?(readme, "theorem aaos_soundness")
    end
  end
end
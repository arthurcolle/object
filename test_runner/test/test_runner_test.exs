defmodule TestRunnerTest do
  use ExUnit.Case
  doctest TestRunner

  test "greets the world" do
    assert TestRunner.hello() == :world
  end
end

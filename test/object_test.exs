defmodule ObjectTest do
  use ExUnit.Case
  doctest Object

  test "creates basic object" do
    object = Object.new(id: "test_obj", state: %{value: 1})
    assert object.id == "test_obj"
    assert object.state.value == 1
  end
end

defmodule Object.MixProject do
  use Mix.Project

  def project do
    [
      app: :object,
      version: "0.1.0",
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger, :crypto],
      mod: {Object.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:gen_stage, "~> 1.2"},
      {:telemetry, "~> 1.0"},
      {:dspy, "~> 0.1.0", optional: true},
      {:lmstudio, "~> 0.1.0", optional: true},
      {:httpoison, "~> 2.0"},
      {:jason, "~> 1.4"},
      {:yaml_elixir, "~> 2.9"},
      {:typed_struct, "~> 0.3"},
      {:msgpax, "~> 2.3"},
      {:ex_doc, ">= 0.0.0", only: :dev, runtime: false}
    ]
  end

  defp description do
    "A comprehensive Elixir object system with AI integration, hierarchical coordination, and meta-schema evolution capabilities"
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{
        "GitHub" => "https://github.com/arthurcolle/object",
        "OORL Framework" => "https://x.com/arthurcolle/status/1881166459499622496"
      }
    ]
  end
end

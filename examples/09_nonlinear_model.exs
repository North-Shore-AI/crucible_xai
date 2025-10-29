#!/usr/bin/env elixir

# Nonlinear Model Example
# This example demonstrates explaining complex nonlinear models with LIME

IO.puts("=" |> String.duplicate(80))
IO.puts("Nonlinear Model Explanation Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a complex nonlinear model
# This simulates a neural network or other complex model
# Model combines polynomial, trigonometric, and interaction terms
nonlinear_model = fn [x, y] ->
  # Polynomial terms
  poly = 0.5 * x * x + 0.3 * y * y

  # Trigonometric terms
  trig = 2.0 * :math.sin(x) + 1.5 * :math.cos(y)

  # Interaction term
  interaction = 0.4 * x * y

  # Linear terms
  linear = 1.2 * x + 0.8 * y

  # Combine all
  poly + trig + interaction + linear + 5.0
end

# Multiple instances to explain
instances = [
  [0.0, 0.0],
  [1.0, 1.0],
  [2.0, 2.0],
  [3.0, 1.0],
  [-1.0, 2.0]
]

IO.puts("Model: Complex nonlinear function")
IO.puts("Components: polynomial + trigonometric + interaction + linear terms")
IO.puts("Formula: 0.5x² + 0.3y² + 2sin(x) + 1.5cos(y) + 0.4xy + 1.2x + 0.8y + 5")
IO.puts("")

IO.puts("Explaining #{length(instances)} different points:")
IO.puts("")

# Explain each instance
explanations =
  Enum.map(instances, fn instance ->
    prediction = nonlinear_model.(instance)
    explanation = CrucibleXai.explain(instance, nonlinear_model, num_samples: 5000)
    {instance, prediction, explanation}
  end)

# Display results
Enum.with_index(explanations, 1)
|> Enum.each(fn {{instance, prediction, explanation}, idx} ->
  [x, y] = instance

  IO.puts("Point #{idx}: [#{Float.round(x, 2)}, #{Float.round(y, 2)}]")
  IO.puts("-" |> String.duplicate(60))
  IO.puts("  Prediction: #{Float.round(prediction, 4)}")
  IO.puts("  R² score: #{Float.round(explanation.score, 4)}")
  IO.puts("  Local linear approximation:")

  IO.puts(
    "    Feature 0 (x) weight: #{Float.round(Map.get(explanation.feature_weights, 0, 0.0), 4)}"
  )

  IO.puts(
    "    Feature 1 (y) weight: #{Float.round(Map.get(explanation.feature_weights, 1, 0.0), 4)}"
  )

  IO.puts("    Intercept: #{Float.round(explanation.intercept, 4)}")

  # Calculate what the true local gradient would be
  # ∂f/∂x = x + 2cos(x) + 0.4y + 1.2
  # ∂f/∂y = 0.6y - 1.5sin(y) + 0.4x + 0.8
  true_grad_x = x + 2.0 * :math.cos(x) + 0.4 * y + 1.2
  true_grad_y = 0.6 * y - 1.5 * :math.sin(y) + 0.4 * x + 0.8

  IO.puts("  True local gradient:")
  IO.puts("    ∂f/∂x ≈ #{Float.round(true_grad_x, 4)}")
  IO.puts("    ∂f/∂y ≈ #{Float.round(true_grad_y, 4)}")
  IO.puts("")
end)

# Analyze quality of explanations
avg_score =
  explanations
  |> Enum.map(fn {_, _, exp} -> exp.score end)
  |> Enum.sum()
  |> Kernel./(length(explanations))

IO.puts("Overall Quality Metrics:")
IO.puts("=" |> String.duplicate(80))
IO.puts("  Average R² score: #{Float.round(avg_score, 4)}")

cond do
  avg_score > 0.9 ->
    IO.puts("  ✓ Excellent: R² > 0.9 indicates very good local approximation")

  avg_score > 0.8 ->
    IO.puts("  ✓ Good: R² > 0.8 indicates good local approximation")

  avg_score > 0.7 ->
    IO.puts("  ⚠ Fair: R² > 0.7 indicates acceptable approximation")

  true ->
    IO.puts("  ⚠ Poor: R² < 0.7 suggests model is highly nonlinear locally")
end

IO.puts("")

IO.puts("KEY INSIGHTS:")
IO.puts("=" |> String.duplicate(80))
IO.puts("  1. LIME approximates nonlinear models with local linear models")
IO.puts("  2. R² score indicates how well the linear model fits locally")
IO.puts("  3. Feature weights represent local sensitivity (like gradients)")
IO.puts("  4. Weights can vary significantly across different points")
IO.puts("  5. For highly nonlinear models, use more samples for better fit")
IO.puts("")

IO.puts("Understanding LIME for Nonlinear Models:")
IO.puts("  • LIME creates a linear approximation around each point")
IO.puts("  • This is like computing a tangent plane to the function")
IO.puts("  • Feature weights are similar to partial derivatives")
IO.puts("  • Different points can have very different local explanations")
IO.puts("  • This is correct behavior - it shows how model behavior changes")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

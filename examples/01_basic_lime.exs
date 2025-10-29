#!/usr/bin/env elixir

# Basic LIME Explanation Example
# This example demonstrates how to use LIME to explain predictions from a simple linear model

IO.puts("=" |> String.duplicate(80))
IO.puts("Basic LIME Explanation Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a simple linear prediction function
# This represents a model that takes two features and makes a prediction
# Model: prediction = 2.0 * x + 3.0 * y + 1.0
predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 1.0 end

# Instance to explain
instance = [1.0, 2.0]

IO.puts("Explaining prediction for instance: #{inspect(instance)}")
IO.puts("Model: f(x, y) = 2.0*x + 3.0*y + 1.0")
IO.puts("")

# Get the actual prediction
prediction = predict_fn.(instance)
IO.puts("Actual prediction: #{prediction}")
IO.puts("")

# Generate LIME explanation
IO.puts("Generating LIME explanation...")
explanation = CrucibleXai.explain(instance, predict_fn)

IO.puts("")
IO.puts("Explanation Results:")
IO.puts("-------------------")
IO.puts("Feature weights: #{inspect(explanation.feature_weights)}")
IO.puts("Intercept: #{explanation.intercept}")
IO.puts("R² score (local fidelity): #{explanation.score}")
IO.puts("Method: #{explanation.method}")
IO.puts("")

# Get top features
IO.puts("Top Features by Importance:")
top_features = CrucibleXAI.Explanation.top_features(explanation, 5)

Enum.each(top_features, fn {idx, weight} ->
  IO.puts("  Feature #{idx}: #{Float.round(weight, 4)}")
end)

IO.puts("")

# Display as text
IO.puts("Text Representation:")
IO.puts(CrucibleXAI.Explanation.to_text(explanation))
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

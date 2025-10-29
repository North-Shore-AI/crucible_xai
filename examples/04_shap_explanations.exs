#!/usr/bin/env elixir

# SHAP Explanations Example
# This example demonstrates how to use SHAP for theoretically grounded feature attribution

IO.puts("=" |> String.duplicate(80))
IO.puts("SHAP Explanations Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a prediction function
predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

# Instance to explain
instance = [1.0, 1.0]

# Background data (representative baseline samples)
background = [
  [0.0, 0.0],
  [1.0, 1.0],
  [2.0, 2.0]
]

IO.puts("Explaining prediction for instance: #{inspect(instance)}")
IO.puts("Model: f(x, y) = 2.0*x + 3.0*y")
IO.puts("Background data: #{inspect(background)}")
IO.puts("")

# Get predictions for context
prediction = predict_fn.(instance)
baseline_predictions = Enum.map(background, &predict_fn.(&1))
avg_baseline = Enum.sum(baseline_predictions) / length(baseline_predictions)

IO.puts("Predictions:")
IO.puts("  Instance: #{prediction}")
IO.puts("  Average baseline: #{Float.round(avg_baseline, 4)}")
IO.puts("  Difference: #{Float.round(prediction - avg_baseline, 4)}")
IO.puts("")

# Generate SHAP values
IO.puts("Generating SHAP values...")
shap_values = CrucibleXai.explain_shap(instance, background, predict_fn, num_samples: 2000)

IO.puts("")
IO.puts("SHAP Values:")
IO.puts("  Feature 0: #{Float.round(Map.get(shap_values, 0, 0.0), 4)}")
IO.puts("  Feature 1: #{Float.round(Map.get(shap_values, 1, 0.0), 4)}")
IO.puts("")

# Verify additivity property
shap_sum = shap_values |> Map.values() |> Enum.sum()
IO.puts("SHAP Properties Verification:")
IO.puts("  Sum of SHAP values: #{Float.round(shap_sum, 4)}")
IO.puts("  Expected (prediction - baseline): #{Float.round(prediction - avg_baseline, 4)}")
IO.puts("  Additivity satisfied: #{abs(shap_sum - (prediction - avg_baseline)) < 0.5}")
IO.puts("")

# Verify using built-in validator
is_valid = CrucibleXAI.SHAP.verify_additivity(shap_values, instance, background, predict_fn)
IO.puts("  Built-in validator result: #{is_valid}")
IO.puts("")

# Compare with LIME
IO.puts("Comparison with LIME:")
IO.puts("-" |> String.duplicate(40))
lime_explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

IO.puts("LIME weights:")

Enum.each(lime_explanation.feature_weights, fn {idx, weight} ->
  IO.puts("  Feature #{idx}: #{Float.round(weight, 4)}")
end)

IO.puts("")

IO.puts("SHAP values:")

Enum.each(shap_values, fn {idx, value} ->
  IO.puts("  Feature #{idx}: #{Float.round(value, 4)}")
end)

IO.puts("")

IO.puts("Key Differences:")
IO.puts("  - SHAP provides theoretically grounded attributions (Shapley values)")
IO.puts("  - LIME provides local linear approximations")
IO.puts("  - SHAP guarantees additivity: sum of values = prediction - baseline")
IO.puts("  - For linear models, both methods converge to similar results")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

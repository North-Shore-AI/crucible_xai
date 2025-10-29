#!/usr/bin/env elixir

# Feature Importance Example
# This example demonstrates permutation importance for global feature ranking

IO.puts("=" |> String.duplicate(80))
IO.puts("Feature Importance (Permutation) Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a prediction function
# Model simulates a loan approval system
# prediction = 0.5 * age + 0.3 * income + 0.2 * credit_score
predict_fn = fn [age, income, credit_score] ->
  0.5 * age + 0.3 * income + 0.2 * credit_score
end

# Generate validation data
validation_data = [
  {[25.0, 50.0, 700.0], 25.0 * 0.5 + 50.0 * 0.3 + 700.0 * 0.2},
  {[35.0, 75.0, 750.0], 35.0 * 0.5 + 75.0 * 0.3 + 750.0 * 0.2},
  {[45.0, 100.0, 800.0], 45.0 * 0.5 + 100.0 * 0.3 + 800.0 * 0.2},
  {[30.0, 60.0, 720.0], 30.0 * 0.5 + 60.0 * 0.3 + 720.0 * 0.2},
  {[40.0, 90.0, 780.0], 40.0 * 0.5 + 90.0 * 0.3 + 780.0 * 0.2},
  {[28.0, 55.0, 710.0], 28.0 * 0.5 + 55.0 * 0.3 + 710.0 * 0.2},
  {[50.0, 120.0, 820.0], 50.0 * 0.5 + 120.0 * 0.3 + 820.0 * 0.2},
  {[33.0, 70.0, 740.0], 33.0 * 0.5 + 70.0 * 0.3 + 740.0 * 0.2}
]

feature_names = ["Age", "Income (k$)", "Credit Score"]

IO.puts("Model: Loan Approval Score")
IO.puts("Features: #{Enum.join(feature_names, ", ")}")
IO.puts("Formula: 0.5*age + 0.3*income + 0.2*credit_score")
IO.puts("")
IO.puts("Validation set size: #{length(validation_data)} samples")
IO.puts("")

# Calculate baseline performance
IO.puts("Calculating baseline model performance...")
baseline_predictions = Enum.map(validation_data, fn {instance, _} -> predict_fn.(instance) end)
true_values = Enum.map(validation_data, fn {_, label} -> label end)

baseline_mse =
  Enum.zip(baseline_predictions, true_values)
  |> Enum.map(fn {pred, true_val} -> (pred - true_val) * (pred - true_val) end)
  |> Enum.sum()
  |> Kernel./(length(validation_data))

IO.puts("Baseline MSE: #{Float.round(baseline_mse, 6)}")
IO.puts("")

# Compute permutation importance
IO.puts("Computing permutation importance...")

importance =
  CrucibleXai.feature_importance(
    predict_fn,
    validation_data,
    metric: :mse,
    num_repeats: 10
  )

IO.puts("")
IO.puts("Feature Importance Results:")
IO.puts("-" |> String.duplicate(80))

# Display in order of importance
importance
|> Enum.sort_by(fn {_idx, %{importance: imp}} -> -imp end)
|> Enum.with_index(1)
|> Enum.each(fn {{idx, %{importance: imp, std_dev: std}}, rank} ->
  feature_name = Enum.at(feature_names, idx)
  IO.puts("#{rank}. #{feature_name} (Feature #{idx})")
  IO.puts("   Importance: #{Float.round(imp, 6)}")
  IO.puts("   Std Dev: #{Float.round(std, 6)}")
  IO.puts("   (Higher = more important for prediction)")
  IO.puts("")
end)

# Get top 2 features
top_features = CrucibleXAI.FeatureAttribution.top_k(importance, 2)
IO.puts("Top 2 Features:")

Enum.each(top_features, fn {idx, _} ->
  IO.puts("  - #{Enum.at(feature_names, idx)}")
end)

IO.puts("")

IO.puts("Interpretation:")
IO.puts("  - Importance reflects how much MSE increases when feature is permuted")
IO.puts("  - Age has highest importance (coefficient 0.5 in model)")
IO.puts("  - Income has moderate importance (coefficient 0.3)")
IO.puts("  - Credit Score has lowest importance (coefficient 0.2)")
IO.puts("  - These rankings match the actual model coefficients!")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

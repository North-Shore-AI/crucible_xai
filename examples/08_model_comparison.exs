#!/usr/bin/env elixir

# Model Comparison Example
# This example shows how to compare explanations from different models

IO.puts("=" |> String.duplicate(80))
IO.puts("Model Comparison Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define two different models
# Model A: Linear model emphasizing income
model_a = fn [age, income, credit_score] ->
  0.2 * age + 0.6 * income + 0.2 * credit_score
end

# Model B: Linear model emphasizing credit score
model_b = fn [age, income, credit_score] ->
  0.2 * age + 0.2 * income + 0.6 * credit_score
end

# Test instance
instance = [35.0, 75.0, 750.0]
feature_names = ["Age", "Income (k$)", "Credit Score"]

IO.puts("Comparing two loan approval models:")
IO.puts("Instance: #{inspect(instance)}")
IO.puts("Features: #{Enum.join(feature_names, ", ")}")
IO.puts("")

# Get predictions from both models
pred_a = model_a.(instance)
pred_b = model_b.(instance)

IO.puts("Predictions:")
IO.puts("  Model A: #{Float.round(pred_a, 2)}")
IO.puts("  Model B: #{Float.round(pred_b, 2)}")
IO.puts("  Difference: #{Float.round(abs(pred_a - pred_b), 2)}")
IO.puts("")

# Explain both models
IO.puts("Generating explanations...")
exp_a = CrucibleXai.explain(instance, model_a, num_samples: 3000)
exp_b = CrucibleXai.explain(instance, model_b, num_samples: 3000)
IO.puts("✓ Explanations generated")
IO.puts("")

# Display Model A explanation
IO.puts("Model A Explanation:")
IO.puts("-" |> String.duplicate(80))
IO.puts("Strategy: Income-focused model")
IO.puts("R² score: #{Float.round(exp_a.score, 4)}")
IO.puts("")
IO.puts("Feature Importance:")
top_a = CrucibleXAI.Explanation.top_features(exp_a, 3)

Enum.each(top_a, fn {idx, weight} ->
  feature = Enum.at(feature_names, idx)
  IO.puts("  #{feature}: #{Float.round(weight, 4)}")
end)

IO.puts("")

# Display Model B explanation
IO.puts("Model B Explanation:")
IO.puts("-" |> String.duplicate(80))
IO.puts("Strategy: Credit-score-focused model")
IO.puts("R² score: #{Float.round(exp_b.score, 4)}")
IO.puts("")
IO.puts("Feature Importance:")
top_b = CrucibleXAI.Explanation.top_features(exp_b, 3)

Enum.each(top_b, fn {idx, weight} ->
  feature = Enum.at(feature_names, idx)
  IO.puts("  #{feature}: #{Float.round(weight, 4)}")
end)

IO.puts("")

# Compare feature importance
IO.puts("Feature Importance Comparison:")
IO.puts("-" |> String.duplicate(80))

IO.puts(
  String.pad_trailing("Feature", 20) <>
    " | " <>
    String.pad_trailing("Model A", 12) <>
    " | " <>
    String.pad_trailing("Model B", 12) <>
    " | " <>
    "Difference"
)

IO.puts(String.duplicate("-", 80))

Enum.with_index(feature_names)
|> Enum.each(fn {feature, idx} ->
  weight_a = Map.get(exp_a.feature_weights, idx, 0.0)
  weight_b = Map.get(exp_b.feature_weights, idx, 0.0)
  diff = abs(weight_a - weight_b)

  IO.puts(
    (String.pad_trailing(feature, 20) <>
       " | " <>
       String.pad_trailing(Float.round(weight_a, 4) |> to_string(), 12) <>
       " | " <>
       String.pad_trailing(Float.round(weight_b, 4) |> to_string(), 12) <>
       " | " <>
       Float.round(diff, 4))
    |> to_string()
  )
end)

IO.puts("")

# Find the most different features
feature_diffs =
  Enum.with_index(feature_names)
  |> Enum.map(fn {feature, idx} ->
    weight_a = Map.get(exp_a.feature_weights, idx, 0.0)
    weight_b = Map.get(exp_b.feature_weights, idx, 0.0)
    {feature, abs(weight_a - weight_b)}
  end)
  |> Enum.sort_by(fn {_, diff} -> -diff end)

IO.puts("Key Differences (sorted by magnitude):")

Enum.each(feature_diffs, fn {feature, diff} ->
  IO.puts("  #{feature}: #{Float.round(diff, 4)}")
end)

IO.puts("")

IO.puts("INSIGHTS:")
IO.puts("=" |> String.duplicate(80))
{most_diff_feature, max_diff} = hd(feature_diffs)
IO.puts("  • Most different feature: #{most_diff_feature} (diff: #{Float.round(max_diff, 4)})")
IO.puts("  • Model A prioritizes Income over Credit Score")
IO.puts("  • Model B prioritizes Credit Score over Income")
IO.puts("  • Both models assign similar weight to Age")
IO.puts("")

IO.puts("Use Cases for Model Comparison:")
IO.puts("  1. Model selection: Choose model with better feature importance")
IO.puts("  2. Ensemble verification: Ensure models have diverse strategies")
IO.puts("  3. Debugging: Identify unexpected differences")
IO.puts("  4. Stakeholder communication: Explain model differences")
IO.puts("  5. Bias detection: Compare feature usage across demographics")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

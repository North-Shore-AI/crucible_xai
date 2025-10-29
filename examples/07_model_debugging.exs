#!/usr/bin/env elixir

# Model Debugging Example
# This example shows how to use XAI to debug and understand model behavior

IO.puts("=" |> String.duplicate(80))
IO.puts("Model Debugging Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Simulate a potentially problematic model
# This model might be using unexpected features or have biases
# Model: prediction = 0.1*age + 0.05*income + 0.8*zipcode + 0.05*credit_score
# Note: zipcode has unexpectedly high weight (potential data leakage)
buggy_model = fn [age, income, zipcode, credit_score] ->
  0.1 * age + 0.05 * income + 0.8 * zipcode + 0.05 * credit_score
end

# Create test instances
test_instances = [
  # San Francisco
  [30.0, 75.0, 94102.0, 750.0],
  # New York
  [35.0, 80.0, 10001.0, 760.0],
  # Chicago
  [40.0, 70.0, 60601.0, 740.0]
]

feature_names = ["Age", "Income", "Zipcode", "Credit Score"]

IO.puts("Debugging Model: Loan Approval Predictor")
IO.puts("Features: #{Enum.join(feature_names, ", ")}")
IO.puts("")

IO.puts("Test Instances:")

Enum.with_index(test_instances, 1)
|> Enum.each(fn {instance, idx} ->
  prediction = buggy_model.(instance)
  IO.puts("  #{idx}. #{inspect(instance)} -> Prediction: #{Float.round(prediction, 2)}")
end)

IO.puts("")

# Explain each instance
IO.puts("Generating explanations for each instance...")
IO.puts("=" |> String.duplicate(80))

explanations =
  Enum.map(test_instances, fn instance ->
    CrucibleXai.explain(instance, buggy_model, num_samples: 3000)
  end)

# Analyze explanations
Enum.zip(test_instances, explanations)
|> Enum.with_index(1)
|> Enum.each(fn {{instance, explanation}, idx} ->
  IO.puts("")
  IO.puts("Instance #{idx}: #{inspect(instance)}")
  IO.puts("Prediction: #{Float.round(buggy_model.(instance), 2)}")
  IO.puts("R² score: #{Float.round(explanation.score, 4)}")
  IO.puts("")

  IO.puts("Feature Contributions:")
  top_features = CrucibleXAI.Explanation.top_features(explanation, 4)

  Enum.each(top_features, fn {feature_idx, weight} ->
    feature_name = Enum.at(feature_names, feature_idx)
    IO.puts("  #{feature_name}: #{Float.round(weight, 4)}")
  end)

  IO.puts("-" |> String.duplicate(80))
end)

IO.puts("")
IO.puts("DEBUGGING INSIGHTS:")
IO.puts("=" |> String.duplicate(80))

# Check which feature has highest average importance
avg_weights =
  0..3
  |> Enum.map(fn feature_idx ->
    weights =
      Enum.map(explanations, fn exp ->
        Map.get(exp.feature_weights, feature_idx, 0.0) |> abs()
      end)

    avg = Enum.sum(weights) / length(weights)
    {feature_idx, avg}
  end)
  |> Enum.sort_by(fn {_, avg} -> -avg end)

IO.puts("Average Feature Importance (across all instances):")

Enum.each(avg_weights, fn {feature_idx, avg} ->
  feature_name = Enum.at(feature_names, feature_idx)
  IO.puts("  #{feature_name}: #{Float.round(avg, 4)}")
end)

IO.puts("")

# Identify the problem
{most_important_idx, _} = hd(avg_weights)
most_important_feature = Enum.at(feature_names, most_important_idx)

IO.puts("🚨 POTENTIAL ISSUE DETECTED:")
IO.puts("  Feature '#{most_important_feature}' has the highest importance!")
IO.puts("")

if most_important_feature == "Zipcode" do
  IO.puts("  ⚠️  WARNING: Zipcode should not be the most important feature!")
  IO.puts("  ⚠️  This suggests potential data leakage or bias")
  IO.puts("  ⚠️  Zipcode might be a proxy for protected attributes")
  IO.puts("  ⚠️  Consider removing or reducing weight of this feature")
end

IO.puts("")
IO.puts("RECOMMENDATIONS:")
IO.puts("  1. Investigate why Zipcode has such high importance")
IO.puts("  2. Check if Zipcode is correlated with protected attributes")
IO.puts("  3. Consider feature engineering to reduce reliance on Zipcode")
IO.puts("  4. Re-train model with different feature weights")
IO.puts("  5. Use feature importance to guide feature selection")
IO.puts("")

IO.puts("This example demonstrates how XAI can help identify:")
IO.puts("  ✓ Data leakage issues")
IO.puts("  ✓ Unexpected feature dependencies")
IO.puts("  ✓ Potential biases in models")
IO.puts("  ✓ Features that should be re-evaluated")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

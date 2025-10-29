#!/usr/bin/env elixir

# Batch Explanations Example
# This example demonstrates how to efficiently explain multiple instances

IO.puts("=" |> String.duplicate(80))
IO.puts("Batch Explanations Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a prediction function
predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 0.5 end

# Multiple instances to explain
instances = [
  [1.0, 2.0],
  [2.0, 3.0],
  [3.0, 4.0],
  [4.0, 5.0],
  [5.0, 6.0]
]

IO.puts("Explaining #{length(instances)} instances:")

Enum.each(instances, fn instance ->
  IO.puts("  #{inspect(instance)} -> #{predict_fn.(instance)}")
end)

IO.puts("")

# Explain all instances in batch
IO.puts("Generating batch explanations...")
start_time = System.monotonic_time(:millisecond)
explanations = CrucibleXai.explain_batch(instances, predict_fn, num_samples: 1000)
end_time = System.monotonic_time(:millisecond)
total_duration = end_time - start_time

IO.puts("Batch processing completed in #{total_duration}ms")
IO.puts("")

# Analyze results
IO.puts("Results:")
IO.puts("-" |> String.duplicate(80))

Enum.zip(instances, explanations)
|> Enum.with_index(1)
|> Enum.each(fn {{instance, explanation}, idx} ->
  IO.puts("Instance #{idx}: #{inspect(instance)}")
  IO.puts("  Prediction: #{Float.round(predict_fn.(instance), 4)}")
  IO.puts("  R² score: #{Float.round(explanation.score, 4)}")
  IO.puts("  Feature weights: #{inspect(explanation.feature_weights)}")
  IO.puts("  Duration: #{explanation.metadata.duration_ms}ms")
  IO.puts("")
end)

# Calculate statistics
avg_score =
  explanations
  |> Enum.map(& &1.score)
  |> Enum.sum()
  |> Kernel./(length(explanations))

avg_duration =
  explanations
  |> Enum.map(& &1.metadata.duration_ms)
  |> Enum.sum()
  |> Kernel./(length(explanations))

IO.puts("Summary Statistics:")
IO.puts("  Average R² score: #{Float.round(avg_score, 4)}")
IO.puts("  Average duration per explanation: #{Float.round(avg_duration, 2)}ms")
IO.puts("  Total duration: #{total_duration}ms")
IO.puts("")

# Check consistency of feature importance
IO.puts("Feature Importance Consistency:")
feature_0_weights = Enum.map(explanations, &Map.get(&1.feature_weights, 0, 0.0))
feature_1_weights = Enum.map(explanations, &Map.get(&1.feature_weights, 1, 0.0))

avg_f0 = Enum.sum(feature_0_weights) / length(feature_0_weights)
avg_f1 = Enum.sum(feature_1_weights) / length(feature_1_weights)

IO.puts("  Feature 0 average weight: #{Float.round(avg_f0, 4)} (expected ~2.0)")
IO.puts("  Feature 1 average weight: #{Float.round(avg_f1, 4)} (expected ~3.0)")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

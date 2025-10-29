#!/usr/bin/env elixir

# Customized LIME Example
# This example shows how to fine-tune LIME parameters for better explanations

IO.puts("=" |> String.duplicate(80))
IO.puts("Customized LIME Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a more complex nonlinear model
# Model: prediction = x^2 + 2*y + sin(x) + 1
predict_fn = fn [x, y] ->
  x * x + 2.0 * y + :math.sin(x) + 1.0
end

instance = [2.0, 3.0]

IO.puts("Explaining prediction for instance: #{inspect(instance)}")
IO.puts("Model: f(x, y) = x² + 2*y + sin(x) + 1")
IO.puts("")

prediction = predict_fn.(instance)
IO.puts("Actual prediction: #{Float.round(prediction, 4)}")
IO.puts("")

# Generate explanation with default settings
IO.puts("1. LIME with default settings:")
IO.puts("-" |> String.duplicate(40))
explanation_default = CrucibleXai.explain(instance, predict_fn)
IO.puts("R² score: #{Float.round(explanation_default.score, 4)}")
IO.puts("Feature weights: #{inspect(explanation_default.feature_weights)}")
IO.puts("")

# Generate explanation with custom settings
IO.puts("2. LIME with custom settings (more samples, different kernel):")
IO.puts("-" |> String.duplicate(40))

explanation_custom =
  CrucibleXai.explain(
    instance,
    predict_fn,
    # More samples for better approximation
    num_samples: 10000,
    # Tighter locality
    kernel_width: 0.5,
    # Cosine kernel instead of exponential
    kernel: :cosine,
    # Select top 5 features
    num_features: 5,
    # Use Lasso for feature selection
    feature_selection: :lasso,
    # Gaussian sampling
    sampling_method: :gaussian
  )

IO.puts("R² score: #{Float.round(explanation_custom.score, 4)}")
IO.puts("Feature weights: #{inspect(explanation_custom.feature_weights)}")
IO.puts("Duration: #{explanation_custom.metadata.duration_ms}ms")
IO.puts("")

# Try forward selection
IO.puts("3. LIME with forward selection:")
IO.puts("-" |> String.duplicate(40))

explanation_forward =
  CrucibleXai.explain(
    instance,
    predict_fn,
    num_samples: 5000,
    feature_selection: :forward_selection,
    num_features: 2
  )

IO.puts("R² score: #{Float.round(explanation_forward.score, 4)}")
IO.puts("Feature weights: #{inspect(explanation_forward.feature_weights)}")
IO.puts("")

# Try highest weights selection
IO.puts("4. LIME with highest weights selection:")
IO.puts("-" |> String.duplicate(40))

explanation_highest =
  CrucibleXai.explain(
    instance,
    predict_fn,
    num_samples: 5000,
    feature_selection: :highest_weights,
    num_features: 2
  )

IO.puts("R² score: #{Float.round(explanation_highest.score, 4)}")
IO.puts("Feature weights: #{inspect(explanation_highest.feature_weights)}")
IO.puts("")

IO.puts("Key Insights:")
IO.puts("- More samples generally improve R² score")
IO.puts("- Different feature selection methods may give different insights")
IO.puts("- R² > 0.8 indicates good local fidelity")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

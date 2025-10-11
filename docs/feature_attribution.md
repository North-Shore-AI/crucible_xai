# Feature Attribution Methods

## Overview

Feature attribution methods quantify the contribution of each input feature to a model's prediction. Different methods have different assumptions, computational costs, and interpretations.

## Attribution Methods

### 1. Permutation Importance

**Concept**: Measure how much performance degrades when a feature's values are randomly shuffled.

**Advantages**:
- Model-agnostic
- Easy to understand
- Captures feature interactions

**Disadvantages**:
- Computationally expensive (requires multiple re-evaluations)
- Assumes features are independent

#### Implementation

```elixir
defmodule CrucibleXAI.FeatureAttribution.Permutation do
  @doc """
  Calculate permutation importance for each feature.

  ## Algorithm
  1. Measure baseline performance on validation set
  2. For each feature:
     a. Shuffle that feature's values
     b. Measure performance on shuffled data
     c. Importance = baseline - shuffled performance
  3. Repeat multiple times and average

  ## Options
    * `:metric` - Performance metric (e.g., :accuracy, :mse)
    * `:num_repeats` - Number of permutations per feature (default: 10)
  """
  def calculate(model, validation_data, opts \\ []) do
    metric = Keyword.get(opts, :metric, :accuracy)
    num_repeats = Keyword.get(opts, :num_repeats, 10)

    baseline_score = evaluate_model(model, validation_data, metric)
    n_features = num_features(validation_data)

    # Calculate importance for each feature
    importances = for feature_idx <- 0..(n_features - 1) do
      # Repeat permutation multiple times
      scores = for _ <- 1..num_repeats do
        shuffled_data = permute_feature(validation_data, feature_idx)
        evaluate_model(model, shuffled_data, metric)
      end

      avg_score = Enum.sum(scores) / num_repeats
      importance = baseline_score - avg_score

      {feature_idx, importance, std_dev(scores)}
    end

    %{
      method: :permutation,
      importances: Enum.into(importances, %{}, fn {idx, imp, std} ->
        {idx, %{importance: imp, std: std}}
      end),
      baseline_score: baseline_score
    }
  end

  defp permute_feature(data, feature_idx) do
    # Extract feature column
    feature_values = Enum.map(data, fn {x, y} -> Enum.at(x, feature_idx) end)

    # Shuffle feature values
    shuffled_values = Enum.shuffle(feature_values)

    # Replace feature column with shuffled values
    data
    |> Enum.zip(shuffled_values)
    |> Enum.map(fn {{x, y}, shuffled_val} ->
      new_x = List.replace_at(x, feature_idx, shuffled_val)
      {new_x, y}
    end)
  end

  defp evaluate_model(model, data, metric) do
    predictions = Enum.map(data, fn {x, _} -> model.predict(x) end)
    labels = Enum.map(data, fn {_, y} -> y end)

    calculate_metric(predictions, labels, metric)
  end

  defp calculate_metric(predictions, labels, :accuracy) do
    correct = Enum.zip(predictions, labels)
              |> Enum.count(fn {pred, label} -> pred == label end)
    correct / length(labels)
  end

  defp calculate_metric(predictions, labels, :mse) do
    Enum.zip(predictions, labels)
    |> Enum.map(fn {pred, label} -> (pred - label) ** 2 end)
    |> Enum.sum()
    |> Kernel./(length(labels))
  end
end
```

### 2. Gradient-based Attribution

**Concept**: Use gradients to measure feature sensitivity.

**Advantages**:
- Fast computation (single backward pass)
- Exact for linear models

**Disadvantages**:
- Only works for differentiable models
- Gradients can saturate
- Doesn't account for feature interactions well

#### Gradient × Input

```elixir
defmodule CrucibleXAI.FeatureAttribution.Gradient do
  @doc """
  Gradient × Input attribution.

  Attribution_i = (∂f/∂x_i) × x_i

  This measures the local sensitivity of the output to each input feature,
  scaled by the feature's magnitude.
  """
  def gradient_input(model, instance) do
    # Compute gradients using automatic differentiation
    gradients = compute_gradients(model, instance)

    # Multiply gradients by input values
    attributions = Nx.multiply(gradients, Nx.tensor(instance))
                   |> Nx.to_flat_list()

    %{
      method: :gradient_input,
      attributions: attributions |> Enum.with_index() |> Enum.into(%{}, fn {v, i} -> {i, v} end)
    }
  end

  defp compute_gradients(model, instance) do
    # Use Nx automatic differentiation
    grad_fn = Nx.Defn.grad(fn x -> model.forward(x) end)
    grad_fn.(Nx.tensor(instance))
  end
end
```

#### Integrated Gradients

**Paper**: Sundararajan et al. (2017). Axiomatic Attribution for Deep Networks.

**Concept**: Integrate gradients along the path from a baseline to the input.

**Advantages**:
- Satisfies desirable axioms (sensitivity, implementation invariance)
- More stable than vanilla gradients
- Accounts for non-linearities

```elixir
defmodule CrucibleXAI.FeatureAttribution.IntegratedGradients do
  @doc """
  Integrated Gradients attribution.

  IG_i(x) = (x_i - x'_i) × ∫_{α=0}^1 (∂f/∂x_i)(x' + α(x - x')) dα

  where x' is a baseline (e.g., all zeros or mean of training data).

  ## Algorithm
  1. Define a baseline (reference point)
  2. Generate path from baseline to instance
  3. Compute gradients at points along path
  4. Integrate gradients (approximate with Riemann sum)
  5. Scale by (input - baseline)
  """
  def calculate(model, instance, baseline, opts \\ []) do
    steps = Keyword.get(opts, :steps, 50)

    # Generate interpolated inputs
    alphas = Nx.linspace(0, 1, n: steps)
    instance_tensor = Nx.tensor(instance)
    baseline_tensor = Nx.tensor(baseline)

    # Path from baseline to instance
    path = Nx.multiply(alphas, Nx.subtract(instance_tensor, baseline_tensor))
           |> Nx.add(baseline_tensor)

    # Compute gradients at each point on path
    gradients = compute_path_gradients(model, path)

    # Integrate using trapezoidal rule
    integrated = integrate_gradients(gradients, steps)

    # Scale by (input - baseline)
    attributions = Nx.multiply(integrated, Nx.subtract(instance_tensor, baseline_tensor))
                   |> Nx.to_flat_list()

    %{
      method: :integrated_gradients,
      attributions: attributions |> Enum.with_index() |> Enum.into(%{}, fn {v, i} -> {i, v} end),
      baseline: baseline,
      steps: steps
    }
  end

  defp compute_path_gradients(model, path) do
    # Compute gradients for each point on path
    grad_fn = Nx.Defn.grad(fn x -> model.forward(x) end)

    # Vectorize gradient computation
    Nx.map(path, grad_fn)
  end

  defp integrate_gradients(gradients, steps) do
    # Trapezoidal rule integration
    Nx.mean(gradients, axes: [0])
  end
end
```

### 3. Occlusion-based Attribution

**Concept**: Measure the change in prediction when a feature is occluded (set to baseline).

**Advantages**:
- Model-agnostic
- Intuitive interpretation
- Can handle discrete features

```elixir
defmodule CrucibleXAI.FeatureAttribution.Occlusion do
  @doc """
  Occlusion-based feature attribution.

  For each feature:
  1. Set feature to baseline value (e.g., 0 or mean)
  2. Measure change in prediction
  3. Attribution = original_prediction - occluded_prediction
  """
  def calculate(model, instance, opts \\ []) do
    baseline_value = Keyword.get(opts, :baseline, :zero)

    original_prediction = model.predict(instance)

    attributions = instance
    |> Enum.with_index()
    |> Enum.map(fn {_value, idx} ->
      # Create occluded instance
      occluded = occlude_feature(instance, idx, baseline_value)

      # Measure prediction change
      occluded_prediction = model.predict(occluded)
      attribution = original_prediction - occluded_prediction

      {idx, attribution}
    end)
    |> Enum.into(%{})

    %{
      method: :occlusion,
      attributions: attributions,
      baseline: baseline_value
    }
  end

  defp occlude_feature(instance, feature_idx, baseline) do
    baseline_val = case baseline do
      :zero -> 0
      :mean -> calculate_mean_for_feature(feature_idx)
      value -> value
    end

    List.replace_at(instance, feature_idx, baseline_val)
  end

  @doc """
  Sliding window occlusion for sequential data.

  Useful for time series or text where we want to occlude
  contiguous segments rather than individual features.
  """
  def sliding_window(model, instance, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, 3)
    stride = Keyword.get(opts, :stride, 1)

    original_prediction = model.predict(instance)

    # Generate all window positions
    window_positions = 0..(length(instance) - window_size)//stride

    attributions = Enum.map(window_positions, fn start_idx ->
      # Occlude window
      occluded = occlude_window(instance, start_idx, window_size)

      # Measure impact
      occluded_prediction = model.predict(occluded)
      impact = original_prediction - occluded_prediction

      {start_idx, impact}
    end)

    %{
      method: :sliding_window_occlusion,
      attributions: Enum.into(attributions, %{}),
      window_size: window_size,
      stride: stride
    }
  end

  defp occlude_window(instance, start_idx, window_size) do
    instance
    |> Enum.with_index()
    |> Enum.map(fn {value, idx} ->
      if idx >= start_idx and idx < start_idx + window_size do
        0  # Occlude with zero
      else
        value
      end
    end)
  end
end
```

### 4. DeepLIFT

**Paper**: Shrikumar et al. (2017). Learning Important Features Through Propagating Activation Differences.

**Concept**: Compare neuron activations to reference activations and back-propagate the differences.

```elixir
defmodule CrucibleXAI.FeatureAttribution.DeepLIFT do
  @doc """
  DeepLIFT attribution.

  Decomposes the output prediction into contributions from each input feature
  by comparing activations to reference activations.
  """
  def calculate(model, instance, baseline, opts \\ []) do
    # Forward pass on instance
    instance_activations = model.forward_with_activations(instance)

    # Forward pass on baseline
    baseline_activations = model.forward_with_activations(baseline)

    # Backward pass: compute contributions
    contributions = backpropagate_contributions(
      model,
      instance_activations,
      baseline_activations
    )

    %{
      method: :deep_lift,
      attributions: contributions,
      baseline: baseline
    }
  end

  defp backpropagate_contributions(model, instance_acts, baseline_acts) do
    # Start from output layer
    # For each layer going backward:
    #   1. Compute Δactivation = instance_act - baseline_act
    #   2. Distribute Δactivation to inputs using chain rule
    #   3. Accumulate contributions
    # Implementation depends on model architecture
  end
end
```

### 5. Layer-wise Relevance Propagation (LRP)

**Concept**: Decompose the prediction into relevance scores by backward propagation.

```elixir
defmodule CrucibleXAI.FeatureAttribution.LRP do
  @doc """
  Layer-wise Relevance Propagation.

  Redistributes the output prediction back through the network
  following conservation of relevance principle.
  """
  def calculate(model, instance, opts \\ []) do
    rule = Keyword.get(opts, :rule, :epsilon)

    # Forward pass
    activations = model.forward_with_activations(instance)

    # Initialize relevance at output
    output_relevance = List.last(activations)

    # Backward pass: propagate relevance
    relevances = propagate_relevance_backward(
      model,
      activations,
      output_relevance,
      rule
    )

    # Input layer relevances are the attributions
    input_relevances = List.first(relevances)

    %{
      method: :lrp,
      attributions: input_relevances |> Enum.with_index() |> Enum.into(%{}, fn {v, i} -> {i, v} end),
      rule: rule
    }
  end

  defp propagate_relevance_backward(model, activations, output_relevance, rule) do
    # For each layer from output to input:
    #   Apply relevance propagation rule
    #   Common rules:
    #   - ε-rule: R_j = Σ_k (a_j * w_jk / (Σ_j a_j * w_jk + ε)) * R_k
    #   - γ-rule: favor positive contributions
    #   - α-β rule: separately handle positive and negative contributions
  end
end
```

## Comparison of Methods

```elixir
defmodule CrucibleXAI.FeatureAttribution.Comparison do
  @doc """
  Compare different attribution methods on the same instance.
  """
  def compare_methods(model, instance, opts \\ []) do
    methods = Keyword.get(opts, :methods, [:permutation, :gradient_input, :integrated_gradients, :occlusion])

    results = Enum.map(methods, fn method ->
      attribution = case method do
        :permutation ->
          Permutation.calculate(model, instance, opts)
        :gradient_input ->
          Gradient.gradient_input(model, instance)
        :integrated_gradients ->
          baseline = Keyword.get(opts, :baseline, zeros_like(instance))
          IntegratedGradients.calculate(model, instance, baseline, opts)
        :occlusion ->
          Occlusion.calculate(model, instance, opts)
      end

      {method, attribution}
    end)

    %{
      instance: instance,
      methods: Enum.into(results, %{}),
      correlation: calculate_correlation_matrix(results)
    }
  end

  defp calculate_correlation_matrix(results) do
    # Calculate pairwise correlation between attribution methods
    # Helps identify agreement/disagreement between methods
  end
end
```

## Validation and Metrics

### Faithfulness

**Concept**: How well do attributions reflect the model's actual behavior?

```elixir
defmodule CrucibleXAI.FeatureAttribution.Validation do
  @doc """
  Measure faithfulness by incrementally removing features.

  Remove features in order of attribution (highest to lowest).
  A faithful attribution should cause rapid performance degradation.
  """
  def faithfulness_test(model, instance, attributions, opts \\ []) do
    n_features = length(instance)

    # Sort features by attribution (descending)
    sorted_features = attributions
                      |> Enum.sort_by(fn {_, v} -> abs(v) end, :desc)
                      |> Enum.map(fn {idx, _} -> idx end)

    # Incrementally remove features
    predictions = Enum.map(0..n_features, fn k ->
      # Remove top k features
      features_to_remove = Enum.take(sorted_features, k)
      modified_instance = remove_features(instance, features_to_remove)

      model.predict(modified_instance)
    end)

    # Calculate AUC of prediction curve
    auc = calculate_auc(predictions)

    %{
      metric: :faithfulness,
      auc: auc,
      predictions: predictions
    }
  end

  @doc """
  Monotonicity test: removing important features should monotonically
  decrease prediction (for positive class).
  """
  def monotonicity_test(model, instance, attributions) do
    # Similar to faithfulness but checks for monotonic decrease
  end

  @doc """
  Infidelity metric from Yeh et al. (2019).

  Measures correlation between attribution and actual prediction changes.
  """
  def infidelity(model, instance, attributions, opts \\ []) do
    n_samples = Keyword.get(opts, :n_samples, 100)

    # Generate perturbations
    perturbations = generate_perturbations(instance, n_samples)

    # For each perturbation:
    #   1. Compute actual prediction difference
    #   2. Compute expected difference using attributions
    #   3. Calculate squared error

    errors = Enum.map(perturbations, fn perturbation ->
      actual_diff = model.predict(perturbation) - model.predict(instance)

      expected_diff = dot_product(attributions, subtract(perturbation, instance))

      (actual_diff - expected_diff) ** 2
    end)

    %{
      metric: :infidelity,
      score: Enum.sum(errors) / n_samples
    }
  end
end
```

### Sensitivity

**Concept**: How sensitive are attributions to small input changes?

```elixir
def sensitivity_test(model, instance, attribution_method, opts \\ []) do
  n_trials = Keyword.get(opts, :n_trials, 10)
  noise_scale = Keyword.get(opts, :noise_scale, 0.01)

  # Compute attributions for original instance
  original_attr = attribution_method.(model, instance)

  # Add small noise and recompute attributions
  noisy_attributions = Enum.map(1..n_trials, fn _ ->
    noisy_instance = add_noise(instance, noise_scale)
    attribution_method.(model, noisy_instance)
  end)

  # Measure variance in attributions
  variance = calculate_attribution_variance(original_attr, noisy_attributions)

  %{
    metric: :sensitivity,
    variance: variance,
    trials: n_trials
  }
end
```

## Best Practices

### 1. Use Multiple Methods

Different methods have different strengths. Compare multiple methods:

```elixir
comparison = CrucibleXAI.FeatureAttribution.compare_methods(
  model,
  instance,
  methods: [:permutation, :integrated_gradients, :lime]
)

# Features that are important across all methods are more reliable
consensus = find_consensus_features(comparison)
```

### 2. Validate Attributions

Always validate attributions using faithfulness tests:

```elixir
validation = CrucibleXAI.FeatureAttribution.Validation.faithfulness_test(
  model,
  instance,
  attributions
)

if validation.auc < 0.7 do
  IO.puts("Warning: Low faithfulness score. Attributions may be unreliable.")
end
```

### 3. Consider Computational Cost

| Method | Complexity | Model Requirements |
|--------|-----------|-------------------|
| Permutation | O(n × m) | None (model-agnostic) |
| Gradient × Input | O(1) | Differentiable |
| Integrated Gradients | O(k) | Differentiable |
| Occlusion | O(n) | None (model-agnostic) |
| LIME | O(s) | None (model-agnostic) |

Where:
- n = number of features
- m = size of validation set (for permutation)
- k = number of integration steps
- s = number of LIME samples

### 4. Choose Appropriate Baseline

For gradient-based methods, baseline choice matters:

```elixir
# Common baselines
baseline_zero = List.duplicate(0, n_features)
baseline_mean = calculate_training_mean()
baseline_blur = apply_blur(instance)  # For images

# Try multiple baselines
baselines = [baseline_zero, baseline_mean, baseline_blur]
attributions = Enum.map(baselines, fn baseline ->
  IntegratedGradients.calculate(model, instance, baseline)
end)

# Average attributions across baselines
avg_attribution = average_attributions(attributions)
```

## References

1. Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
2. Shrikumar, A., Greenside, P., & Kundaje, A. (2017). Learning Important Features Through Propagating Activation Differences. *ICML*.
3. Yeh, C. K., et al. (2019). On the (In)fidelity and Sensitivity of Explanations. *NeurIPS*.
4. Breiman, L. (2001). Random Forests. *Machine Learning*.

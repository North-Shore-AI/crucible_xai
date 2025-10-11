<p align="center">
  <img src="assets/crucible_xai.svg" alt="CrucibleXAI" width="150"/>
</p>

# CrucibleXAI

**Explainable AI (XAI) Tools for the Crucible Framework**

A comprehensive Explainable AI (XAI) library for Elixir, providing model interpretability and explanation tools. CrucibleXAI helps you understand and explain AI model predictions through LIME implementations, SHAP-like explanations, feature attribution methods, and comprehensive model interpretability techniques for both local and global explanations.

## Features

- **LIME (Local Interpretable Model-agnostic Explanations)**: Explain individual predictions with locally faithful interpretable models
- **SHAP-like Explanations**: Shapley value-based feature attribution for understanding feature contributions
- **Feature Attribution**: Multiple methods for quantifying feature importance
- **Model Interpretability**: Tools for understanding model behavior and decision boundaries
- **Local Explanations**: Understand individual predictions in detail
- **Global Explanations**: Gain insights into overall model behavior
- **Integration with Crucible**: Seamless integration with the Crucible AI framework
- **Nx-Powered**: Built on Nx for high-performance numerical computations

## Design Principles

1. **Model-Agnostic**: Works with any black-box model that can produce predictions
2. **Faithful Explanations**: Locally faithful interpretable models that accurately reflect the original model's behavior
3. **Human-Interpretable**: Explanations designed for human understanding
4. **Flexible**: Supports various explanation types and customization options
5. **Efficient**: Optimized implementations using Nx tensors

## Installation

Add `crucible_xai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_xai, "~> 0.1.0"}
  ]
end
```

Or install from GitHub:

```elixir
def deps do
  [
    {:crucible_xai, github: "North-Shore-AI/crucible_xai"}
  ]
end
```

## Quick Start

### LIME Explanations

```elixir
# Define your model prediction function
predict_fn = fn input ->
  YourModel.predict(input)
end

# Generate LIME explanation for a single instance
explanation = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: predict_fn,
  num_samples: 5000,
  num_features: 10
)

# View feature importance
IO.inspect(explanation.feature_weights)
# => %{feature_1: 0.85, feature_2: -0.42, feature_3: 0.31, ...}
```

### SHAP-like Explanations

```elixir
# Calculate Shapley values for feature attribution
shap_values = CrucibleXAI.SHAP.explain(
  instance: instance,
  background_data: background_samples,
  predict_fn: predict_fn
)

# Visualize contributions
CrucibleXAI.SHAP.plot_contributions(shap_values)
```

### Feature Attribution

```elixir
# Calculate feature importance scores
importance = CrucibleXAI.FeatureAttribution.calculate(
  model: model,
  data: validation_data,
  method: :permutation
)

# Get top features
top_features = CrucibleXAI.FeatureAttribution.top_k(importance, k: 10)
```

### Global Model Interpretability

```elixir
# Analyze global feature importance
global_importance = CrucibleXAI.Global.feature_importance(
  model: model,
  data: training_data,
  method: :aggregate_lime
)

# Discover decision rules
rules = CrucibleXAI.Global.extract_rules(
  model: model,
  data: training_data,
  max_depth: 3
)
```

## LIME Implementation

CrucibleXAI provides a robust implementation of LIME (Ribeiro et al., 2016):

### How LIME Works

1. **Perturbation**: Generate perturbed samples around the instance to explain
2. **Prediction**: Get predictions for perturbed samples from the black-box model
3. **Weighting**: Weight samples by proximity to the original instance
4. **Interpretation**: Fit an interpretable model (e.g., linear regression) on weighted samples
5. **Explanation**: Extract feature weights from the interpretable model

### Configuration Options

```elixir
CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: predict_fn,

  # Sampling options
  num_samples: 5000,              # Number of perturbed samples
  sampling_method: :gaussian,     # :gaussian, :uniform, or :categorical

  # Feature selection
  num_features: 10,               # Number of features to include in explanation
  feature_selection: :lasso,      # :lasso, :forward, or :highest_weights

  # Weighting
  kernel_width: 0.75,             # Kernel width for sample weighting
  kernel: :exponential,           # :exponential or :cosine

  # Interpretable model
  model_type: :linear_regression, # :linear_regression or :decision_tree

  # Other options
  discretize_continuous: true,    # Discretize continuous features
  categorical_features: []        # List of categorical feature indices
)
```

## SHAP-like Explanations

### Shapley Value Computation

```elixir
# Calculate exact Shapley values (computationally expensive)
shap_values = CrucibleXAI.SHAP.explain(
  instance: instance,
  background_data: background,
  predict_fn: predict_fn,
  method: :exact
)

# Use sampling-based approximation (faster)
shap_values = CrucibleXAI.SHAP.explain(
  instance: instance,
  background_data: background,
  predict_fn: predict_fn,
  method: :kernel_shap,
  num_samples: 1000
)

# TreeSHAP for tree-based models
shap_values = CrucibleXAI.SHAP.explain(
  instance: instance,
  model: tree_model,
  method: :tree_shap
)
```

### Visualization

```elixir
# Force plot showing feature contributions
CrucibleXAI.SHAP.force_plot(shap_values, instance)

# Summary plot for multiple instances
CrucibleXAI.SHAP.summary_plot(shap_values_list, feature_names)

# Dependence plot for feature interactions
CrucibleXAI.SHAP.dependence_plot(shap_values, feature_index: 0)
```

## Feature Attribution Methods

### Permutation Importance

```elixir
# Calculate permutation importance
importance = CrucibleXAI.FeatureAttribution.permutation_importance(
  model: model,
  data: validation_data,
  metric: :accuracy,
  num_repeats: 10
)
```

### Gradient-based Attribution

```elixir
# Gradient × Input attribution
attributions = CrucibleXAI.FeatureAttribution.gradient_input(
  model: neural_network,
  instance: instance
)

# Integrated Gradients
attributions = CrucibleXAI.FeatureAttribution.integrated_gradients(
  model: neural_network,
  instance: instance,
  baseline: baseline,
  steps: 50
)
```

### Occlusion-based Methods

```elixir
# Occlusion sensitivity
sensitivity = CrucibleXAI.FeatureAttribution.occlusion(
  model: model,
  instance: instance,
  window_size: 3,
  stride: 1
)
```

## Global Interpretability

### Partial Dependence Plots

```elixir
# Calculate partial dependence
pd = CrucibleXAI.Global.partial_dependence(
  model: model,
  data: training_data,
  feature: :age,
  num_grid_points: 20
)

# Plot partial dependence
CrucibleXAI.Global.plot_partial_dependence(pd)
```

### Individual Conditional Expectation (ICE)

```elixir
# ICE plots for instance-level effects
ice = CrucibleXAI.Global.ice_plot(
  model: model,
  data: training_data,
  feature: :age,
  num_instances: 50
)
```

### Feature Interaction Detection

```elixir
# H-statistic for feature interactions
interactions = CrucibleXAI.Global.h_statistic(
  model: model,
  data: training_data,
  feature_pairs: [{:age, :income}, {:education, :experience}]
)
```

## Module Structure

```
lib/crucible_xai/
├── xai.ex                        # Main API
├── lime.ex                       # LIME implementation
├── shap.ex                       # SHAP-like explanations
├── feature_attribution.ex        # Feature attribution methods
├── global.ex                     # Global interpretability
└── utils/
    ├── sampling.ex               # Sampling strategies
    ├── kernels.ex                # Kernel functions
    ├── interpretable_models.ex   # Linear models, decision trees
    └── visualization.ex          # Plotting utilities
```

## Integration with Crucible Framework

CrucibleXAI integrates seamlessly with other Crucible components:

```elixir
# Use with Crucible models
model = Crucible.Model.load("my_model")
explanation = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: &Crucible.Model.predict(model, &1)
)

# Combine with benchmarking
benchmark_result = CrucibleBench.compare(
  model_a_predictions,
  model_b_predictions
)

# Add explanations to understand performance differences
explanations = Enum.map(test_instances, fn instance ->
  CrucibleXAI.LIME.explain(instance: instance, predict_fn: predict_fn)
end)
```

## Use Cases

### Model Debugging

```elixir
# Find instances where model is uncertain
uncertain_instances = Enum.filter(test_data, fn instance ->
  prediction = model.predict(instance)
  prediction.confidence < 0.6
end)

# Explain uncertain predictions
explanations = Enum.map(uncertain_instances, fn instance ->
  CrucibleXAI.LIME.explain(instance: instance, predict_fn: &model.predict/1)
end)
```

### Fairness Analysis

```elixir
# Analyze feature importance across demographic groups
groups = [:group_a, :group_b, :group_c]

importance_by_group = Enum.map(groups, fn group ->
  data = filter_by_group(training_data, group)
  {group, CrucibleXAI.Global.feature_importance(model: model, data: data)}
end)
```

### Model Comparison

```elixir
# Compare explanations from different models
explanation_a = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: &model_a.predict/1
)

explanation_b = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: &model_b.predict/1
)

# Analyze differences
CrucibleXAI.compare_explanations(explanation_a, explanation_b)
```

### Trust and Validation

```elixir
# Validate that model uses expected features
important_features = CrucibleXAI.FeatureAttribution.top_k(
  importance,
  k: 10
)

expected_features = [:feature_a, :feature_b, :feature_c]
unexpected = important_features -- expected_features

if length(unexpected) > 0 do
  IO.puts("Warning: Model relies on unexpected features: #{inspect(unexpected)}")
end
```

## Advanced Topics

### Custom Sampling Strategies

```elixir
# Define custom sampling function
custom_sampler = fn instance, num_samples ->
  # Your custom sampling logic
  generate_samples(instance, num_samples)
end

explanation = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: predict_fn,
  sampler: custom_sampler
)
```

### Custom Interpretable Models

```elixir
# Use custom interpretable model
custom_model = %{
  fit: fn samples, weights -> train_custom_model(samples, weights) end,
  explain: fn model -> extract_explanation(model) end
}

explanation = CrucibleXAI.LIME.explain(
  instance: instance,
  predict_fn: predict_fn,
  interpretable_model: custom_model
)
```

### Batch Explanations

```elixir
# Efficiently explain multiple instances
explanations = CrucibleXAI.LIME.explain_batch(
  instances: instances,
  predict_fn: predict_fn,
  parallel: true,
  max_concurrency: 8
)
```

## Performance Considerations

- **Sampling**: More samples → better approximation but slower computation
- **Parallelization**: Use `parallel: true` for batch explanations
- **Caching**: Cache perturbed samples and predictions when explaining similar instances
- **Feature Selection**: Limit `num_features` for faster computation and simpler explanations

## Testing

Run the test suite:

```bash
mix test
```

Run specific tests:

```bash
mix test test/lime_test.exs
mix test test/shap_test.exs
mix test test/feature_attribution_test.exs
```

## Examples

See the `examples/` directory for comprehensive examples:

1. `examples/lime_basic.exs` - Basic LIME usage
2. `examples/shap_explanations.exs` - SHAP value calculations
3. `examples/feature_importance.exs` - Feature attribution methods
4. `examples/global_analysis.exs` - Global interpretability
5. `examples/model_debugging.exs` - Debugging with XAI

Run examples:

```bash
mix run examples/lime_basic.exs
```

## References

### Research Papers

- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS*.
- Sundararajan, M., Taly, A., & Yan, Q. (2017). Axiomatic Attribution for Deep Networks. *ICML*.
- Goldstein, A., et al. (2015). Peeking Inside the Black Box: Visualizing Statistical Learning with Plots of Individual Conditional Expectation. *JCGS*.

### Books

- Molnar, C. (2022). *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable*.
- Samek, W., et al. (2019). *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*.

## Contributing

This is part of the Crucible AI Research Infrastructure. See the main project documentation for contribution guidelines.

## Roadmap

See [docs/roadmap.md](docs/roadmap.md) for planned features and development timeline.

## License

MIT License - see [LICENSE](https://github.com/North-Shore-AI/crucible_xai/blob/main/LICENSE) file for details

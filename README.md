<p align="center">
  <img src="assets/crucible_xai.svg" alt="CrucibleXAI" width="150"/>
</p>

# CrucibleXAI

**Explainable AI (XAI) Library for Elixir**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-red.svg)](https://www.erlang.org)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/North-Shore-AI/crucible_xai/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-98_passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-84.8%25-green.svg)]()

---

A production-ready Explainable AI (XAI) library for Elixir, providing model interpretability through **LIME (Local Interpretable Model-agnostic Explanations)**. Built on Nx for high-performance numerical computing with comprehensive test coverage and strict quality standards.

## ✨ Features

### Currently Implemented

- ✅ **LIME Implementation**: Full LIME algorithm with local linear approximations
- ✅ **Multiple Sampling Strategies**: Gaussian, Uniform, Categorical, and Combined
- ✅ **Flexible Kernels**: Exponential and Cosine proximity weighting
- ✅ **Feature Selection**: Highest weights, Forward selection, Lasso-approximation
- ✅ **Interpretable Models**: Weighted Linear Regression and Ridge Regression
- ✅ **Batch Processing**: Efficient explanation of multiple instances
- ✅ **Model-Agnostic**: Works with any prediction function
- ✅ **High Performance**: Nx tensor operations, typical <50ms per explanation
- ✅ **Well-Tested**: 98 tests (77 unit + 14 property-based + 7 doctests), 84.8% coverage
- ✅ **Zero Warnings**: Strict compilation with comprehensive type specifications

### Roadmap

- 🚧 **SHAP-like Explanations**: Shapley value-based feature attribution (Phase 2)
- 🚧 **Global Interpretability**: Partial dependence plots, feature interactions (Phase 3)
- 🚧 **Visualization**: Interactive HTML plots and charts (Phase 3)
- 🚧 **CrucibleTrace Integration**: Combined explanations with reasoning traces (Phase 4)

## 📦 Installation

Add `crucible_xai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_xai, github: "North-Shore-AI/crucible_xai"}
  ]
end
```

## 🚀 Quick Start

### Basic LIME Explanation

```elixir
# Define your prediction function (any model that returns a number)
predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 1.0 end

# Instance to explain
instance = [1.0, 2.0]

# Generate explanation
explanation = CrucibleXai.explain(instance, predict_fn)

# View feature weights
IO.inspect(explanation.feature_weights)
# => %{0 => 2.0, 1 => 3.0}

# Get top features
top_features = CrucibleXAI.Explanation.top_features(explanation, 5)

# View as text
IO.puts(CrucibleXAI.Explanation.to_text(explanation))
```

### Customized LIME

```elixir
# Fine-tune LIME parameters
explanation = CrucibleXai.explain(
  instance,
  predict_fn,
  num_samples: 5000,              # More samples = better approximation
  kernel_width: 0.75,             # Locality width
  kernel: :exponential,           # or :cosine
  num_features: 10,               # Top K features to select
  feature_selection: :lasso,      # :highest_weights, :forward_selection, or :lasso
  sampling_method: :gaussian      # :gaussian, :uniform, or :combined
)

# Check explanation quality
IO.puts("R² score: #{explanation.score}")  # Should be > 0.8 for good local fidelity
```

### Batch Explanations

```elixir
# Explain multiple instances efficiently
instances = [
  [1.0, 2.0],
  [2.0, 3.0],
  [3.0, 4.0]
]

explanations = CrucibleXai.explain_batch(instances, predict_fn, num_samples: 1000)

# Analyze consistency
Enum.each(explanations, fn exp ->
  IO.puts("R² = #{exp.score}, Duration = #{exp.metadata.duration_ms}ms")
end)
```

## 📊 Understanding LIME

### How It Works

1. **Perturbation**: Generate samples around the instance (e.g., Gaussian noise)
2. **Prediction**: Get predictions from your black-box model
3. **Weighting**: Weight samples by proximity to the instance (closer = higher weight)
4. **Feature Selection**: Optionally select top K most important features
5. **Fit**: Train a simple linear model on weighted samples
6. **Extract**: Feature weights = explanation

### Visual Example

```
Original Instance: [5.0, 10.0]
↓
Generate 5000 perturbed samples around it
↓
Get predictions from your complex model
↓
Weight samples (closer to [5.0, 10.0] = higher weight)
↓
Fit: prediction ≈ 2.1*feature₀ + 3.2*feature₁ + 0.5
↓
Explanation: Feature 1 has impact 3.2, Feature 0 has impact 2.1
```

## 🎯 Configuration Options

### Sampling Methods

```elixir
# Gaussian (default): Add normal noise scaled by feature std dev
sampling_method: :gaussian

# Uniform: Add uniform noise within a range
sampling_method: :uniform

# Categorical: Sample from possible categorical values
sampling_method: :categorical

# Combined: Mix continuous and categorical features
sampling_method: :combined
```

### Kernel Functions

```elixir
# Exponential (default): exp(-distance²/width²)
kernel: :exponential, kernel_width: 0.75

# Cosine: (1 + cos(π*distance))/2
kernel: :cosine
```

### Feature Selection

```elixir
# Highest Weights: Fastest, selects by absolute coefficient
feature_selection: :highest_weights

# Forward Selection: Greedy, adds features improving R²
feature_selection: :forward_selection

# Lasso: L1 regularization approximation via Ridge
feature_selection: :lasso
```

## 📖 API Documentation

### Main Functions

```elixir
# Single explanation
@spec CrucibleXai.explain(instance, predict_fn, opts) :: Explanation.t()

# Batch explanations
@spec CrucibleXai.explain_batch([instance], predict_fn, opts) :: [Explanation.t()]
```

### Explanation Struct

```elixir
%CrucibleXAI.Explanation{
  instance: [1.0, 2.0],                    # Original instance
  feature_weights: %{0 => 2.0, 1 => 3.0},  # Feature importance
  intercept: 1.0,                           # Baseline value
  score: 0.95,                              # R² goodness of fit
  method: :lime,                            # XAI method used
  metadata: %{                              # Additional info
    num_samples: 5000,
    kernel: :exponential,
    duration_ms: 45
  }
}
```

### Utility Functions

```elixir
# Get top K features by importance
Explanation.top_features(explanation, k)

# Get features that increase prediction
Explanation.positive_features(explanation)

# Get features that decrease prediction
Explanation.negative_features(explanation)

# Feature importance (absolute values)
Explanation.feature_importance(explanation)

# Text visualization
Explanation.to_text(explanation, num_features: 10)

# JSON export
Explanation.to_map(explanation) |> Jason.encode!()
```

## 🏗️ Module Structure

```
lib/crucible_xai/
├── crucible_xai.ex                  # Public API
├── explanation.ex                    # Explanation struct & utilities
├── lime.ex                          # Main LIME algorithm
└── lime/
    ├── sampling.ex                  # Perturbation strategies
    ├── kernels.ex                   # Proximity weighting
    ├── interpretable_models.ex      # Linear/Ridge regression
    └── feature_selection.ex         # Feature selection methods
```

## 🧪 Testing

```bash
# Run all tests
mix test

# Run with coverage
mix coveralls

# Run specific module tests
mix test test/crucible_xai/lime_test.exs

# Run property-based tests only
mix test --only property

# Quality checks
mix compile --warnings-as-errors  # Zero warnings
mix dialyzer                       # Type checking
mix credo --strict                 # Code quality
```

## 📈 Performance

Typical performance on M1 Mac:

- Single explanation (5000 samples): **40-60ms**
- Batch of 100 instances: **~5 seconds**
- Linear model R² scores: **>0.95** (excellent local fidelity)
- Nonlinear model R² scores: **0.85-0.95** (good approximation)

## 🔬 Example Use Cases

### Model Debugging

```elixir
# Find where model relies on unexpected features
explanation = CrucibleXai.explain(problematic_instance, predict_fn)

top_features = Explanation.top_features(explanation, 5)
# => [{3, 0.85}, {7, 0.62}, {1, -0.45}, ...]

# Feature 3 shouldn't be important!
if {3, _} in top_features do
  Logger.warn("Model unexpectedly uses feature 3")
end
```

### Model Comparison

```elixir
# Compare two models on same instance
exp_a = CrucibleXai.explain(instance, &model_a.predict/1)
exp_b = CrucibleXai.explain(instance, &model_b.predict/1)

# Different feature importance?
IO.puts("Model A top features: #{inspect(Explanation.top_features(exp_a, 3))}")
IO.puts("Model B top features: #{inspect(Explanation.top_features(exp_b, 3))}")
```

### Trust Validation

```elixir
# Validate model uses domain knowledge
explanations = CrucibleXai.explain_batch(validation_set, predict_fn)

# Check if important features make sense
Enum.each(explanations, fn exp ->
  top = Explanation.top_features(exp, 1) |> hd() |> elem(0)

  unless top in expected_important_features do
    Logger.warn("Unexpected important feature: #{top}")
  end
end)
```

## 📚 References

### Research Papers

- **Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).** "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD*. [Paper](https://arxiv.org/abs/1602.04938)

### Books

- **Molnar, C. (2022).** Interpretable Machine Learning. [Online Book](https://christophm.github.io/interpretable-ml-book/)

## 🤝 Contributing

This is part of the Crucible AI Research Infrastructure. Contributions welcome!

## 📋 License

MIT License - see [LICENSE](LICENSE) file for details

---

**Built with ❤️ by North Shore AI** | [Documentation](https://hexdocs.pm/crucible_xai) | [GitHub](https://github.com/North-Shore-AI/crucible_xai)

<p align="center">
  <img src="assets/crucible_xai.svg" alt="CrucibleXAI" width="200"/>
</p>

# CrucibleXAI

**Explainable AI (XAI) Library for Elixir**

[![Elixir](https://img.shields.io/badge/elixir-1.14+-purple.svg)](https://elixir-lang.org)
[![OTP](https://img.shields.io/badge/otp-25+-blue.svg)](https://www.erlang.org)
[![Hex.pm](https://img.shields.io/hexpm/v/crucible_xai.svg)](https://hex.pm/packages/crucible_xai)
[![Documentation](https://img.shields.io/badge/docs-hexdocs-purple.svg)](https://hexdocs.pm/crucible_xai)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/North-Shore-AI/crucible_xai/blob/main/LICENSE)
[![Tests](https://img.shields.io/badge/tests-362_passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-96%25-green.svg)]()

---

A production-ready Explainable AI (XAI) library for Elixir, providing model interpretability through **LIME, SHAP, and Feature Attribution methods**. Built on Nx for high-performance numerical computing with comprehensive test coverage and strict quality standards.

**Version**: 0.4.0 | **Tests**: 362+ passing | **Coverage**: 96%+

## ✨ Features

### Currently Implemented

#### Local Attribution Methods (10 methods)
- ✅ **LIME**: Full algorithm with Gaussian/Uniform/Categorical sampling, multiple kernels, feature selection
- ✅ **KernelSHAP**: Model-agnostic SHAP via weighted regression (~1s, approximate)
- ✅ **LinearSHAP**: Ultra-fast exact SHAP for linear models (<2ms, 1000x faster than Kernel)
- ✅ **SamplingShap**: Monte Carlo SHAP approximation (~100ms, faster than Kernel)
- ✅ **Gradient × Input**: Fast gradient-based attribution (<1ms, requires Nx)
- ✅ **Integrated Gradients**: Axiomatic gradient method with completeness guarantee (5-50ms)
- ✅ **SmoothGrad**: Noise-reduced gradient attribution via averaging (10-100ms)
- ✅ **Feature Occlusion**: Model-agnostic attribution via feature removal (1-5ms per feature)
- ✅ **Sliding Window Occlusion**: Sequential data attribution (1-10ms per window)
- ✅ **Occlusion Sensitivity**: Normalized occlusion scores with optional absolute values

#### Global Interpretability Methods (7 methods)
- ✅ **Permutation Importance**: Global feature ranking across validation set
- ✅ **PDP 1D**: Partial dependence plots for single features (10-50ms)
- ✅ **PDP 2D**: Feature interaction visualization (50-200ms)
- ✅ **ICE**: Individual conditional expectation curves (10-100ms)
- ✅ **Centered ICE**: Relative change visualization from baseline
- ✅ **ALE**: Accumulated local effects, robust for correlated features (10-100ms)
- ✅ **H-Statistic**: Friedman's interaction strength detection (50-300ms per pair)

#### Infrastructure & Quality
- ✅ **Parallel Batch Processing**: LIME, SHAP, and Occlusion with configurable concurrency (40-60% faster)
- ✅ **Model-Agnostic**: Works with any prediction function (black-box or white-box)
- ✅ **High Performance**: Nx tensor operations throughout
- ✅ **HTML Visualizations**: Interactive Chart.js visualizations for LIME and SHAP
- ✅ **Well-Tested**: 337+ tests (unit, property-based, doctests), >96% coverage
- ✅ **Zero Warnings**: Strict compilation with comprehensive type specifications
- ✅ **Shapley Properties**: SHAP additivity, symmetry, and dummy properties validated

#### Validation & Quality Metrics (new in v0.3.0)
- ✅ **Faithfulness**: Feature removal correlation + monotonicity checks
- ✅ **Infidelity**: Perturbation-based error between predicted and actual changes
- ✅ **Sensitivity**: Input and hyperparameter stability scoring
- ✅ **Axioms**: Completeness, symmetry, dummy, linearity validation
- ✅ **Quality Gates**: `quick_validate/4` fast pass/fail for production
- ✅ **Benchmarking**: Compare methods by quality score and runtime

### Roadmap

- 🚧 **TreeSHAP**: Efficient exact SHAP for tree-based models
- 🚧 **Advanced Visualizations**: Enhanced interactive plots for all methods
- 🚧 **Visualization**: Interactive HTML plots and charts (Phase 5)
- 🚧 **CrucibleTrace Integration**: Combined explanations with reasoning traces (Phase 6)

## 📦 Installation

Add `crucible_xai` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:crucible_xai, "~> 0.4.0"}
  ]
end
```

Then run:
```bash
mix deps.get
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

### Pipeline Stage Integration (new in v0.4.0)

Use CrucibleXAI as a stage in Crucible framework pipelines:

```elixir
# In a Crucible pipeline
context = %{
  model_fn: &MyModel.predict/1,
  instances: [[1.0, 2.0], [3.0, 4.0]],
  background_data: [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
  experiment: %{
    reliability: %{
      xai: %{
        methods: [:lime, :shap, :feature_importance],
        lime_opts: %{num_samples: 1000},
        shap_opts: %{num_samples: 500},
        parallel: true
      }
    }
  }
}

{:ok, updated_context} = CrucibleXAI.Stage.run(context)

# Access results
lime_explanations = updated_context.xai.explanations.lime
shap_values = updated_context.xai.explanations.shap
feature_importance = updated_context.xai.explanations.feature_importance

# Introspect the stage
stage_info = CrucibleXAI.Stage.describe(%{verbose: true})
IO.inspect(stage_info.available_methods)
```

The Stage module integrates seamlessly with:
- **crucible_bench** - Statistical testing
- **crucible_telemetry** - Metrics tracking
- **crucible_harness** - Experiment orchestration
- **crucible_trace** - Causal transparency

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

# Sequential processing (default)
explanations = CrucibleXai.explain_batch(instances, predict_fn, num_samples: 1000)

# Parallel processing for faster batch explanations (40-60% speed improvement)
explanations = CrucibleXai.explain_batch(instances, predict_fn,
  num_samples: 1000,
  parallel: true,
  max_concurrency: 4  # Optional: control concurrent tasks
)

# Analyze consistency
Enum.each(explanations, fn exp ->
  IO.puts("R² = #{exp.score}, Duration = #{exp.metadata.duration_ms}ms")
end)
```

### SHAP Explanations

CrucibleXAI supports multiple SHAP variants for different use cases:

```elixir
# KernelSHAP: Model-agnostic, most accurate but slower
predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
instance = [1.0, 1.0]
background = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

shap_values = CrucibleXai.explain_shap(instance, background, predict_fn, num_samples: 2000)
# => %{0 => 2.0, 1 => 3.0}

# LinearSHAP: Exact and ultra-fast for linear models (1000x faster!)
coefficients = %{0 => 2.0, 1 => 3.0}
intercept = 0.0

linear_shap = CrucibleXai.explain_shap(
  instance,
  background,
  predict_fn,
  method: :linear_shap,
  coefficients: coefficients,
  intercept: intercept
)
# => %{0 => 2.0, 1 => 3.0} (exact, <2ms)

# SamplingShap: Monte Carlo approximation, faster than KernelSHAP
sampling_shap = CrucibleXai.explain_shap(
  instance,
  background,
  predict_fn,
  method: :sampling_shap,
  num_samples: 1000
)
# => %{0 => ~2.0, 1 => ~3.0} (approximate, ~100ms)

# Verify additivity: SHAP values sum to (prediction - baseline)
prediction = predict_fn.(instance)
baseline = predict_fn.([0.0, 0.0])
shap_sum = Enum.sum(Map.values(shap_values))
# shap_sum ≈ prediction - baseline

# Verify with built-in validator
is_valid = CrucibleXAI.SHAP.verify_additivity(shap_values, instance, background, predict_fn)
# => true
```

### Validate Explanations (new in v0.3.0)

Run end-to-end quality checks on any explanation:

```elixir
explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

validation = CrucibleXai.validate_explanation(
  explanation,
  instance,
  predict_fn,
  include_sensitivity: true,         # slower, enables stability scoring
  baseline: background_data,         # used for axiom checks (SHAP/IG)
  num_perturbations: 100             # used for infidelity
)

IO.puts(validation.summary)
validation.quality_score            # 0.0 - 1.0 (weighted faithfulness/infidelity/axioms)
validation.faithfulness.faithfulness_score
validation.infidelity.infidelity_score
validation.sensitivity.stability_score
validation.axioms.all_satisfied
```

Fast production gate:

```elixir
quick = CrucibleXai.quick_validate(explanation, instance, predict_fn)

quick.passes_quality_gate           # true/false (faithfulness >= 0.7 and infidelity <= 0.1)
quick.interpretation                # "Excellent" | "Good" | "Acceptable" | "Poor"
```

Direct metric access:

```elixir
# Faithfulness: correlation between attribution rank and prediction drop
faithfulness = CrucibleXai.measure_faithfulness(instance, explanation, predict_fn,
  baseline_value: 0.0,
  correlation_method: :spearman
)

# Infidelity: mean squared error between predicted and actual changes
infidelity = CrucibleXai.compute_infidelity(instance, explanation.feature_weights, predict_fn,
  num_perturbations: 200,
  perturbation_std: 0.1,
  normalize: true
)

# Axioms only (completeness, symmetry, dummy, linearity)
axioms = CrucibleXAI.Validation.Axioms.validate_all_axioms(
  explanation.feature_weights,
  instance,
  predict_fn,
  method: explanation.method,
  baseline: background_data
)
```

Compare methods by quality and speed:

```elixir
result = CrucibleXAI.Validation.benchmark_methods(
  instance,
  predict_fn,
  [
    {:lime, num_samples: 2000},
    {:shap, background_data, [num_samples: 1000]}
  ]
)

IO.puts(result.comparison_summary)
# Method  | Faithfulness | Infidelity | Quality | Time
```

### Gradient-based Attribution

For neural networks and differentiable models built with Nx:

```elixir
# Define a differentiable model
model_fn = fn params ->
  # Example: f(x, y) = x^2 + 2*y^2
  Nx.sum(Nx.add(Nx.pow(params[0], 2), Nx.multiply(2.0, Nx.pow(params[1], 2))))
end

instance = Nx.tensor([3.0, 4.0])

# Method 1: Gradient × Input (fastest, simplest)
grad_input = CrucibleXAI.GradientAttribution.gradient_x_input(model_fn, instance)
# => Tensor showing feature attributions

# Method 2: Integrated Gradients (most principled, satisfies completeness axiom)
baseline = Nx.tensor([0.0, 0.0])
integrated = CrucibleXAI.GradientAttribution.integrated_gradients(
  model_fn,
  instance,
  baseline,
  steps: 50
)
# Verify: sum(integrated) ≈ f(instance) - f(baseline)

# Method 3: SmoothGrad (reduces noise via averaging)
smooth = CrucibleXAI.GradientAttribution.smooth_grad(
  model_fn,
  instance,
  noise_level: 0.15,
  n_samples: 50
)
# => Smoother, less noisy attributions
```

### Occlusion-based Attribution

Model-agnostic attribution without requiring gradients:

```elixir
# Works with ANY model (black-box, non-differentiable, etc.)
predict_fn = fn [age, income, credit] ->
  # Any complex logic here
  if income > 50000, do: age * 0.5 + credit * 0.3, else: age * 0.2
end

instance = [35.0, 60000.0, 720.0]

# Feature occlusion: Remove each feature and measure impact
occlusion_attrs = CrucibleXAI.OcclusionAttribution.feature_occlusion(
  instance,
  predict_fn,
  baseline_value: 0.0  # Value to use for occluded features
)
# => %{0 => 17.5, 1 => 21.6, 2 => 10.8}

# Sliding window occlusion: For sequential/time-series data
time_series = [1.0, 2.0, 3.0, 4.0, 5.0]
window_attrs = CrucibleXAI.OcclusionAttribution.sliding_window_occlusion(
  time_series,
  predict_fn,
  window_size: 2,  # Occlude 2 consecutive features
  stride: 1        # Slide by 1 each time
)

# Normalized sensitivity scores
sensitivity = CrucibleXAI.OcclusionAttribution.occlusion_sensitivity(
  instance,
  predict_fn,
  normalize: true,  # Sum to 1.0
  absolute: true    # Use absolute values
)
```

### Global Interpretability (PDP & ICE)

Understand model behavior across the entire feature space:

```elixir
# Partial Dependence Plot (PDP) - Shows average feature effect
predict_fn = fn [age, income, experience] ->
  0.5 * age + 0.3 * income + 0.2 * experience
end

data = [
  [25.0, 50000.0, 2.0],
  [35.0, 75000.0, 5.0],
  [45.0, 100000.0, 10.0]
  # ... more instances
]

# 1D PDP: How does income affect predictions on average?
pdp_1d = CrucibleXAI.Global.PDP.partial_dependence(
  predict_fn,
  data,
  1,  # Feature index for income
  num_grid_points: 20
)
# => %{grid_values: [50k, 55k, ..., 100k], predictions: [avg_pred_at_50k, ...]}

# 2D PDP: How do age AND income interact?
pdp_2d = CrucibleXAI.Global.PDP.partial_dependence_2d(
  predict_fn,
  data,
  {0, 1},  # Age and income
  num_grid_points: 10
)
# => %{grid_values_x: [...], grid_values_y: [...], predictions: [[...]]}

# ICE: Show how predictions change for EACH individual instance
ice = CrucibleXAI.Global.ICE.ice_curves(
  predict_fn,
  data,
  1,  # Analyze income
  num_grid_points: 20
)
# => %{grid_values: [...], curves: [[curve1], [curve2], ...]}

# Centered ICE: Show relative changes
centered = CrucibleXAI.Global.ICE.centered_ice(ice)

# Average ICE equals PDP
pdp_from_ice = CrucibleXAI.Global.ICE.average_ice_curves(ice)

# ALE: Better than PDP when features are correlated
ale = CrucibleXAI.Global.ALE.accumulated_local_effects(
  predict_fn,
  data,
  1,  # Analyze income
  num_bins: 10  # Quantile-based bins
)
# => %{bin_centers: [...], effects: [...], feature_index: 1}
# Effects are centered around zero, showing local changes

# H-Statistic: Detect feature interactions
h_stat = CrucibleXAI.Global.Interaction.h_statistic(
  predict_fn,
  data,
  {0, 1},  # Check if age and income interact
  num_grid_points: 10
)
# => 0.15 (weak interaction)

# Find all pairwise interactions
all_interactions = CrucibleXAI.Global.Interaction.find_all_interactions(
  predict_fn,
  data,
  num_grid_points: 10,
  sort: true,      # Sort by strength
  threshold: 0.2   # Only show H >= 0.2
)
# => [%{feature_pair: {1, 2}, h_statistic: 0.45, interpretation: "Moderate interaction"}, ...]
```

### Feature Attribution (Permutation Importance)

```elixir
# Calculate global feature importance across validation set
predict_fn = fn [age, income, credit_score] ->
  0.5 * age + 0.3 * income + 0.2 * credit_score
end

validation_data = [
  {[25.0, 50000.0, 700.0], 25.5},
  {[35.0, 75000.0, 750.0], 35.3},
  {[45.0, 100000.0, 800.0], 45.2}
  # ... more validation samples
]

# Compute permutation importance
importance = CrucibleXai.feature_importance(
  predict_fn,
  validation_data,
  metric: :mse,
  num_repeats: 10
)
# => %{
#   0 => %{importance: 1.2, std_dev: 0.3},  # Age
#   1 => %{importance: 0.8, std_dev: 0.2},  # Income
#   2 => %{importance: 0.4, std_dev: 0.1}   # Credit score
# }

# Get top 2 features
top_features = CrucibleXAI.FeatureAttribution.top_k(importance, 2)
# => [{0, %{importance: 1.2, ...}}, {1, %{importance: 0.8, ...}}]
```

### Interactive Visualizations

```elixir
# Generate HTML visualization
explanation = CrucibleXai.explain(instance, predict_fn)
html = CrucibleXAI.Visualization.to_html(
  explanation,
  feature_names: %{0 => "Age", 1 => "Income", 2 => "Credit Score"}
)

# Save to file
CrucibleXAI.Visualization.save_html(explanation, "explanation.html")

# Compare LIME vs SHAP
lime_exp = CrucibleXai.explain(instance, predict_fn)
shap_vals = CrucibleXai.explain_shap(instance, background, predict_fn)

comparison_html = CrucibleXAI.Visualization.comparison_html(
  lime_exp,
  shap_vals,
  instance,
  feature_names: %{0 => "Feature A", 1 => "Feature B"}
)

File.write!("comparison.html", comparison_html)
```

## 📊 Understanding the Algorithms

### LIME: Local Interpretable Model-agnostic Explanations

#### How It Works

1. **Perturbation**: Generate samples around the instance (e.g., Gaussian noise)
2. **Prediction**: Get predictions from your black-box model
3. **Weighting**: Weight samples by proximity to the instance (closer = higher weight)
4. **Feature Selection**: Optionally select top K most important features
5. **Fit**: Train a simple linear model on weighted samples
6. **Extract**: Feature weights = explanation

#### Visual Example

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

### SHAP: SHapley Additive exPlanations

#### How It Works

1. **Coalition Generation**: Generate random feature subsets (coalitions)
2. **Coalition Instances**: For each coalition, create instance with only selected features
3. **Predictions**: Get predictions for all coalition instances
4. **SHAP Weighting**: Weight coalitions using SHAP kernel based on size
5. **Regression**: Solve weighted regression to get Shapley values
6. **Properties**: Guarantees additivity, symmetry, and dummy properties

#### Visual Example

```
Instance: [5.0, 10.0], Background: [0.0, 0.0]
↓
Generate coalitions: [0,0], [1,0], [0,1], [1,1], ...
↓
Create instances: [0,0], [5,0], [0,10], [5,10], ...
↓
Get predictions from model for each coalition
↓
Calculate SHAP kernel weights (empty/full coalitions get high weight)
↓
Solve: predictions = coalition_matrix @ shapley_values
↓
Result: φ₀ = 2.0, φ₁ = 3.0
Verify: φ₀ + φ₁ = prediction(5,10) - prediction(0,0) ✓
```

#### LIME vs SHAP

| Aspect | LIME | SHAP (Kernel) | SHAP (Linear) | SHAP (Sampling) |
|--------|------|---------------|---------------|-----------------|
| **Speed** | Fast (~50ms) | Slow (~1s) | Ultra-fast (~1ms) | Fast (~100ms) |
| **Theory** | Heuristic | Game theory | Game theory | Game theory |
| **Accuracy** | Approximate | Approximate | Exact | Approximate |
| **Model Type** | Any | Any | Linear only | Any |
| **Guarantee** | Local fidelity | Additivity | Additivity | Additivity |
| **Use When** | Quick insights | Precise attribution | Linear models | Faster SHAP |

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
├── crucible_xai.ex                      # Public API (explain, explain_batch, explain_shap, feature_importance)
├── stage.ex                              # Pipeline stage for Crucible framework integration (new in v0.4.0)
├── explanation.ex                        # Explanation struct & utilities
│
├── lime.ex                              # LIME with parallel batch processing
├── lime/
│   ├── sampling.ex                      # Gaussian, Uniform, Categorical, Combined
│   ├── kernels.ex                       # Exponential, Cosine kernels
│   ├── interpretable_models.ex          # Linear Regression, Ridge
│   └── feature_selection.ex             # Highest weights, Forward selection, Lasso
│
├── shap.ex                              # SHAP API (3 variants)
├── shap/
│   ├── kernel_shap.ex                   # Model-agnostic approximation (~1s)
│   ├── linear_shap.ex                   # Exact for linear models (<2ms)
│   └── sampling_shap.ex                 # Monte Carlo approximation (~100ms)
│
├── gradient_attribution.ex              # Gradient × Input, Integrated Gradients, SmoothGrad
├── occlusion_attribution.ex             # Feature occlusion, Sliding Window, Sensitivity
│
├── global/
│   ├── pdp.ex                           # Partial Dependence Plots (1D & 2D)
│   ├── ice.ex                           # Individual Conditional Expectation
│   ├── ale.ex                           # Accumulated Local Effects
│   └── interaction.ex                   # H-Statistic for feature interactions
│
├── feature_attribution.ex               # Main attribution API
├── feature_attribution/
│   └── permutation.ex                   # Permutation importance
│
└── visualization.ex                     # HTML visualizations (Chart.js)
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

## 📚 Examples

The `examples/` directory contains 10 comprehensive, runnable examples demonstrating all features:

1. **01_basic_lime.exs** - Basic LIME explanation workflow
2. **02_customized_lime.exs** - Parameter tuning and configuration
3. **03_batch_explanations.exs** - Efficient batch processing
4. **04_shap_explanations.exs** - SHAP values and comparison with LIME
5. **05_feature_importance.exs** - Global feature ranking
6. **06_visualization.exs** - HTML visualization generation
7. **07_model_debugging.exs** - Using XAI for debugging
8. **08_model_comparison.exs** - Comparing different models
9. **09_nonlinear_model.exs** - Explaining complex nonlinear models
10. **10_complete_workflow.exs** - End-to-end XAI workflow

Run any example with:
```bash
mix run examples/01_basic_lime.exs
```

See [examples/README.md](examples/README.md) for detailed documentation.

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

MIT License - see LICENSE file for details

---

**Built with ❤️ by North Shore AI** | [Documentation](https://hexdocs.pm/crucible_xai) | [GitHub](https://github.com/North-Shore-AI/crucible_xai)

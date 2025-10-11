# CrucibleXAI Buildout Plan

## Overview

This document provides a comprehensive implementation plan for CrucibleXAI, an Explainable AI (XAI) library for Elixir. This plan is designed to guide developers through the complete implementation process, from foundational LIME modules to advanced neural network explanations and production features.

CrucibleXAI delivers model-agnostic explainability tools including LIME, SHAP, feature attribution methods, and global interpretability techniques, all built on Nx for high-performance numerical computing.

## Required Reading

Before beginning implementation, developers **must** read the following documents in order:

1. **[docs/architecture.md](docs/architecture.md)** - System architecture and design patterns
   - Understand the model-agnostic interface pattern
   - Learn the module organization: LIME, SHAP, Feature Attribution, Global
   - Review integration with Nx for numerical computations
   - Study the data flow through explanation pipelines
   - Review extension points and behavior protocols

2. **[docs/lime.md](docs/lime.md)** - LIME implementation design
   - Master the LIME algorithm and mathematical formulation
   - Understand sampling strategies (Gaussian, uniform, categorical)
   - Learn kernel functions for proximity weighting
   - Study interpretable models (linear regression, ridge, lasso)
   - Review feature selection methods (forward, lasso, highest weights)

3. **[docs/feature_attribution.md](docs/feature_attribution.md)** - Feature attribution methods
   - Learn permutation importance and gradient-based methods
   - Understand Integrated Gradients and occlusion-based attribution
   - Study DeepLIFT and Layer-wise Relevance Propagation (LRP)
   - Review validation metrics (faithfulness, infidelity, sensitivity)
   - Master best practices for attribution method selection

4. **[docs/roadmap.md](docs/roadmap.md)** - 8-phase implementation roadmap
   - Understand the overall vision and phased approach
   - Review deliverables for each phase (Foundation → Production)
   - Note technical milestones and success metrics
   - Review cross-cutting concerns (testing, documentation, performance)

## Implementation Phases

### Phase 1: Foundation (v0.1.0) - Q1 2025

**Objective**: Establish core infrastructure and basic LIME implementation for tabular data

#### Week 1-2: Core Infrastructure

**Tasks**:
1. Set up development environment
   ```bash
   cd crucible_xai
   mix deps.get
   mix test
   mix docs
   ```

2. Implement core module structure:
   ```elixir
   # lib/crucible_xai.ex
   defmodule CrucibleXAI do
     @moduledoc """
     Main API for explainable AI in Elixir.
     Model-agnostic explanations using LIME, SHAP, and attribution methods.
     """

     def explain(instance, predict_fn, opts \\ [])
     def lime_explain(instance, predict_fn, opts \\ [])
     def shap_explain(instance, predict_fn, opts \\ [])
   end
   ```

3. Create Explanation struct:
   ```elixir
   # lib/crucible_xai/explanation.ex
   defmodule CrucibleXAI.Explanation do
     defstruct [
       :instance,
       :feature_weights,
       :intercept,
       :score,
       :method,
       :metadata
     ]

     def top_features(explanation, k)
     def to_text(explanation)
     def to_json(explanation)
   end
   ```

4. Set up testing framework:
   ```elixir
   # test/support/test_helpers.ex
   defmodule CrucibleXAI.TestHelpers do
     def linear_model(x), do: # simple linear model
     def dummy_predict_fn(x), do: # test prediction function
     def generate_test_data(n_samples, n_features)
   end
   ```

**Deliverables**:
- [ ] Core module structure implemented
- [ ] Explanation struct with utility functions
- [ ] Test infrastructure with helpers
- [ ] Development documentation
- [ ] CI/CD pipeline configured

**Reading Focus**: docs/architecture.md (Module Organization, Design Patterns, Nx Integration)

#### Week 3-4: LIME Sampling & Kernels

**Tasks**:
1. Implement sampling strategies:
   ```elixir
   # lib/crucible_xai/lime/sampling.ex
   defmodule CrucibleXAI.LIME.Sampling do
     def gaussian(instance, n_samples, opts \\ [])
     def uniform(instance, n_samples, opts \\ [])
     def categorical(instance, n_samples, opts \\ [])
     def combined(instance, n_samples, opts \\ [])
   end
   ```

2. Implement kernel functions:
   ```elixir
   # lib/crucible_xai/lime/kernels.ex
   defmodule CrucibleXAI.LIME.Kernels do
     def exponential(distances, kernel_width \\ 0.75)
     def cosine(distances)
     def euclidean_distance(samples, instance)
   end
   ```

3. Comprehensive testing:
   - Test Gaussian perturbation with known statistics
   - Verify categorical sampling distributions
   - Test kernel weight properties (sum to 1, decreasing with distance)
   - Property-based tests for sampling invariants

4. Nx optimization:
   - Use vectorized operations for distance calculations
   - Batch kernel weight computation
   - Memory-efficient tensor operations

**Deliverables**:
- [ ] All sampling strategies implemented
- [ ] Kernel functions with Nx optimization
- [ ] Test coverage > 90%
- [ ] Benchmarks for sampling performance

**Reading Focus**: docs/lime.md (Sampling Module, Kernel Functions)

#### Week 5-6: Interpretable Models

**Tasks**:
1. Implement weighted linear regression:
   ```elixir
   # lib/crucible_xai/lime/interpretable_models/linear_regression.ex
   defmodule CrucibleXAI.LIME.InterpretableModels.LinearRegression do
     def fit(samples, labels, weights)
     def predict(model, samples)
     def coefficients(model)
   end
   ```

2. Implement Ridge regression (L2):
   ```elixir
   # lib/crucible_xai/lime/interpretable_models/ridge.ex
   defmodule CrucibleXAI.LIME.InterpretableModels.Ridge do
     def fit(samples, labels, weights, lambda \\ 1.0)
   end
   ```

3. Add numerical stability:
   - Condition number checks
   - Ridge regularization for ill-conditioned matrices
   - Pseudo-inverse fallback

4. Testing:
   - Verify coefficients match known linear models
   - Test numerical stability with collinear features
   - Validate R² score calculations

**Deliverables**:
- [ ] LinearRegression module complete
- [ ] Ridge regression with regularization
- [ ] Numerical stability measures
- [ ] Comprehensive test coverage

**Reading Focus**: docs/lime.md (Interpretable Models section)

#### Week 7-8: Feature Selection & Main LIME API

**Tasks**:
1. Implement feature selection:
   ```elixir
   # lib/crucible_xai/lime/feature_selection.ex
   defmodule CrucibleXAI.LIME.FeatureSelection do
     def lasso(samples, labels, weights, n_features)
     def forward_selection(samples, labels, weights, n_features)
     def highest_weights(samples, labels, weights, n_features)
   end
   ```

2. Complete main LIME interface:
   ```elixir
   # lib/crucible_xai/lime.ex
   defmodule CrucibleXAI.LIME do
     @default_opts [
       num_samples: 5000,
       kernel_width: 0.75,
       kernel: :exponential,
       num_features: 10,
       feature_selection: :lasso,
       model_type: :linear_regression,
       sampling_method: :gaussian
     ]

     def explain(instance, predict_fn, opts \\ [])
   end
   ```

3. Integration testing:
   - End-to-end LIME explanations
   - Test with various model types
   - Validate local fidelity
   - Test explanation consistency

4. Prepare for v0.1.0:
   - Update CHANGELOG.md
   - Polish README.md with examples
   - Generate documentation: `mix docs`
   - Package validation: `mix hex.build`

**Deliverables**:
- [ ] Feature selection methods complete
- [ ] Full LIME API implemented
- [ ] Local fidelity tests passing
- [ ] v0.1.0 ready for release

**Reading Focus**: docs/lime.md (Main LIME Interface, Testing and Validation)

---

### Phase 2: SHAP & Feature Attribution (v0.2.0) - Q2 2025

**Objective**: Implement SHAP variants and comprehensive feature attribution methods

#### Week 9-10: KernelSHAP

**Tasks**:
1. Implement coalition sampling:
   ```elixir
   # lib/crucible_xai/shap/kernel_shap.ex
   defmodule CrucibleXAI.SHAP.KernelSHAP do
     def explain(instance, background_data, predict_fn, opts \\ [])
     def generate_coalitions(n_features, n_samples)
     def shapley_kernel_weights(coalitions)
   end
   ```

2. Add weighted linear regression solver for SHAP:
   ```elixir
   def solve_shapley_values(coalitions, predictions, weights)
   ```

3. Verify SHAP properties:
   - Efficiency: SHAP values sum to prediction difference
   - Symmetry: equivalent features get equal values
   - Dummy: zero-impact features get zero value

4. Performance optimization:
   - Batch prediction for coalitions
   - Parallel coalition evaluation
   - EXLA backend support

**Deliverables**:
- [ ] KernelSHAP module complete
- [ ] SHAP property tests passing
- [ ] Performance benchmarks
- [ ] Examples and documentation

**Reading Focus**: docs/architecture.md (SHAP Module), docs/roadmap.md (Phase 2)

#### Week 11-12: Additional SHAP Methods

**Tasks**:
1. Implement SamplingShap (Monte Carlo):
   ```elixir
   # lib/crucible_xai/shap/sampling_shap.ex
   defmodule CrucibleXAI.SHAP.SamplingShap do
     def explain(instance, background_data, predict_fn, opts \\ [])
     def monte_carlo_approximation(instance, features, predict_fn, n_samples)
   end
   ```

2. Implement LinearSHAP (for linear models):
   ```elixir
   # lib/crucible_xai/shap/linear_shap.ex
   defmodule CrucibleXAI.SHAP.LinearSHAP do
     def explain(instance, model_coefficients, feature_means)
   end
   ```

3. Add visualization support:
   ```elixir
   # lib/crucible_xai/shap/visualization.ex
   def force_plot(shap_values, instance)
   def summary_plot(shap_values_list, feature_names)
   def dependence_plot(shap_values, feature_index)
   ```

**Deliverables**:
- [ ] SamplingShap and LinearSHAP modules
- [ ] Visualization utilities
- [ ] Comparative analysis tools
- [ ] Usage examples

**Reading Focus**: docs/roadmap.md (SHAP Implementation section)

#### Week 13-14: Permutation Importance

**Tasks**:
1. Implement permutation importance:
   ```elixir
   # lib/crucible_xai/feature_attribution/permutation.ex
   defmodule CrucibleXAI.FeatureAttribution.Permutation do
     def calculate(model, validation_data, opts \\ [])
     def permute_feature(data, feature_idx)
     def with_confidence_intervals(importances, num_repeats)
   end
   ```

2. Add metrics support:
   - Accuracy
   - Mean Squared Error (MSE)
   - R² score
   - Custom metrics

3. Parallel computation:
   - Parallel feature permutation
   - Batch evaluation
   - Progress tracking

**Deliverables**:
- [ ] Permutation importance module
- [ ] Multiple metrics supported
- [ ] Parallel computation
- [ ] Confidence intervals

**Reading Focus**: docs/feature_attribution.md (Permutation Importance)

#### Week 15-16: Gradient-based Attribution

**Tasks**:
1. Implement Gradient × Input:
   ```elixir
   # lib/crucible_xai/feature_attribution/gradient.ex
   defmodule CrucibleXAI.FeatureAttribution.Gradient do
     def gradient_input(model, instance)
     def compute_gradients(model, instance)
   end
   ```

2. Implement Integrated Gradients:
   ```elixir
   # lib/crucible_xai/feature_attribution/integrated_gradients.ex
   defmodule CrucibleXAI.FeatureAttribution.IntegratedGradients do
     def calculate(model, instance, baseline, opts \\ [])
     def compute_path_gradients(model, path)
     def integrate_gradients(gradients, steps)
   end
   ```

3. Add SmoothGrad:
   ```elixir
   def smooth_grad(model, instance, noise_level, n_samples)
   ```

4. Axon integration for neural networks

**Deliverables**:
- [ ] Gradient-based methods complete
- [ ] Axon integration
- [ ] Baseline selection strategies
- [ ] v0.2.0 release

**Reading Focus**: docs/feature_attribution.md (Gradient-based Attribution, Integrated Gradients)

---

### Phase 3: Global Interpretability (v0.3.0) - Q3 2025

**Objective**: Implement global model analysis tools

#### Week 17-18: Partial Dependence Plots

**Tasks**:
1. Implement PDP:
   ```elixir
   # lib/crucible_xai/global/pdp.ex
   defmodule CrucibleXAI.Global.PDP do
     def partial_dependence(model, data, feature, opts \\ [])
     def partial_dependence_2d(model, data, feature_pair, opts \\ [])
     def create_grid(data, feature, n_points)
   end
   ```

2. Optimize computation:
   - Efficient grid sampling
   - Batch predictions
   - Parallel instance evaluation

3. Visualization data:
   - 1D PDP curves
   - 2D PDP heatmaps
   - Export for plotting libraries

**Deliverables**:
- [ ] PDP module for 1D and 2D
- [ ] Efficient computation
- [ ] Visualization support
- [ ] Examples with plots

**Reading Focus**: docs/roadmap.md (Global Analysis Tools)

#### Week 19-20: ICE and ALE

**Tasks**:
1. Implement ICE plots:
   ```elixir
   # lib/crucible_xai/global/ice.ex
   defmodule CrucibleXAI.Global.ICE do
     def ice_plot(model, data, feature, opts \\ [])
     def centered_ice(ice_curves)
   end
   ```

2. Implement ALE:
   ```elixir
   # lib/crucible_xai/global/ale.ex
   defmodule CrucibleXAI.Global.ALE do
     def accumulated_local_effects(model, data, feature, opts \\ [])
   end
   ```

3. Feature interaction detection:
   ```elixir
   # lib/crucible_xai/global/interactions.ex
   def h_statistic(model, data, feature_pairs)
   def interaction_strength(model, data, feature_a, feature_b)
   ```

**Deliverables**:
- [ ] ICE module complete
- [ ] ALE implementation
- [ ] H-statistic for interactions
- [ ] Comparative analysis tools

**Reading Focus**: docs/roadmap.md (Global Analysis Tools - ICE, ALE)

#### Week 21-22: Visualization & Integration

**Tasks**:
1. Comprehensive visualization module:
   ```elixir
   # lib/crucible_xai/visualization.ex
   defmodule CrucibleXAI.Visualization do
     def feature_importance_plot(importances, opts \\ [])
     def explanation_plot(explanation, opts \\ [])
     def pdp_plot(pdp_data, opts \\ [])
     def ice_plot(ice_data, opts \\ [])
   end
   ```

2. VegaLite integration for interactive plots

3. Export formats:
   - JSON for web apps
   - SVG for publications
   - Interactive HTML

4. LiveBook examples:
   - Create comprehensive tutorials
   - Interactive demonstrations
   - Case studies

**Deliverables**:
- [ ] Visualization module complete
- [ ] VegaLite integration
- [ ] Export formats
- [ ] LiveBook tutorials
- [ ] v0.3.0 release

**Reading Focus**: docs/architecture.md (Visualization), docs/roadmap.md (Visualization)

---

### Phase 4: Advanced Explanations (v0.4.0) - Q4 2025

**Objective**: Counterfactual explanations, anchors, and example-based methods

#### Week 23-24: Counterfactual Generation

**Tasks**:
1. Implement DiCE (Diverse Counterfactual Explanations):
   ```elixir
   # lib/crucible_xai/counterfactual/dice.ex
   defmodule CrucibleXAI.Counterfactual.DiCE do
     def generate(instance, predict_fn, desired_outcome, opts \\ [])
     def optimize_with_diversity(candidates, diversity_weight)
   end
   ```

2. Add constraints:
   - Actionability (mutable features only)
   - Plausibility (within data distribution)
   - Minimal perturbation
   - Diversity among counterfactuals

3. Optimization methods:
   - Gradient-based optimization
   - Genetic algorithms
   - Random search with constraints

**Deliverables**:
- [ ] DiCE implementation
- [ ] Constraint handling
- [ ] Multiple optimization methods
- [ ] Examples and documentation

**Reading Focus**: docs/roadmap.md (Counterfactual Explanations)

#### Week 25-26: Anchors & Rules

**Tasks**:
1. Implement Anchors:
   ```elixir
   # lib/crucible_xai/anchors.ex
   defmodule CrucibleXAI.Anchors do
     def explain(instance, predict_fn, opts \\ [])
     def beam_search(instance, predict_fn, beam_width)
     def multi_armed_bandit(instance, predict_fn)
   end
   ```

2. Rule extraction:
   - High-precision rules
   - Coverage metrics
   - Precision metrics

3. Optimization:
   - Efficient rule search
   - Early stopping
   - Parallelization

**Deliverables**:
- [ ] Anchors module complete
- [ ] Rule extraction
- [ ] Coverage and precision metrics
- [ ] Performance optimization

**Reading Focus**: docs/roadmap.md (Anchors section)

#### Week 27-28: Example-based Explanations

**Tasks**:
1. Implement influential instances:
   ```elixir
   # lib/crucible_xai/example_based/influence.ex
   defmodule CrucibleXAI.ExampleBased.Influence do
     def influential_instances(model, instance, training_data, opts \\ [])
   end
   ```

2. Prototypes and criticisms:
   ```elixir
   def find_prototypes(data, k)
   def find_criticisms(data, prototypes, k)
   ```

3. k-NN explanations:
   ```elixir
   def knn_explanation(instance, training_data, k, distance_metric)
   ```

**Deliverables**:
- [ ] Influence functions
- [ ] Prototypes and criticisms
- [ ] k-NN explanations
- [ ] v0.4.0 release

**Reading Focus**: docs/roadmap.md (Example-based Explanations)

---

### Phase 5: Neural Network Support (v0.5.0) - Q1 2026

**Objective**: Deep learning XAI methods with Nx/Axon integration

#### Week 29-30: Layer-wise Relevance Propagation

**Tasks**:
1. Implement LRP:
   ```elixir
   # lib/crucible_xai/neural/lrp.ex
   defmodule CrucibleXAI.Neural.LRP do
     def calculate(model, instance, opts \\ [])
     def propagate_relevance_backward(model, activations, output_relevance, rule)
   end
   ```

2. LRP rules:
   - ε-rule
   - γ-rule
   - α-β rule
   - Layer-specific rule selection

3. Axon integration:
   - Layer activation extraction
   - Backward relevance propagation
   - Compatible with Axon models

**Deliverables**:
- [ ] LRP module with multiple rules
- [ ] Axon integration
- [ ] Layer-specific configuration
- [ ] Examples with neural networks

**Reading Focus**: docs/roadmap.md (Layer-wise Relevance Propagation)

#### Week 31-32: DeepLIFT & GradCAM

**Tasks**:
1. Implement DeepLIFT:
   ```elixir
   # lib/crucible_xai/neural/deep_lift.ex
   defmodule CrucibleXAI.Neural.DeepLIFT do
     def calculate(model, instance, baseline, opts \\ [])
     def backpropagate_contributions(model, instance_acts, baseline_acts)
   end
   ```

2. Implement GradCAM for CNNs:
   ```elixir
   # lib/crucible_xai/neural/grad_cam.ex
   defmodule CrucibleXAI.Neural.GradCAM do
     def calculate(model, instance, target_layer, target_class)
     def guided_backpropagation(model, instance)
   end
   ```

3. Attention visualization for transformers:
   ```elixir
   # lib/crucible_xai/neural/attention.ex
   def visualize_attention(model, instance, layer_idx)
   def multi_head_analysis(attention_weights)
   ```

**Deliverables**:
- [ ] DeepLIFT implementation
- [ ] GradCAM for CNNs
- [ ] Attention visualization
- [ ] Vision and NLP examples

**Reading Focus**: docs/roadmap.md (DeepLIFT, GradCAM, Attention)

#### Week 33-34: Saliency Maps & Integration

**Tasks**:
1. Implement saliency methods:
   ```elixir
   # lib/crucible_xai/neural/saliency.ex
   defmodule CrucibleXAI.Neural.Saliency do
     def vanilla_gradients(model, instance)
     def smooth_grad(model, instance, n_samples, noise_level)
     def integrated_gradients(model, instance, baseline, steps)
     def guided_backpropagation(model, instance)
   end
   ```

2. Unified neural XAI API:
   ```elixir
   # lib/crucible_xai/neural.ex
   defmodule CrucibleXAI.Neural do
     def explain(model, instance, method, opts \\ [])
   end
   ```

3. Comprehensive examples:
   - CNN saliency maps
   - Transformer attention analysis
   - RNN interpretability

**Deliverables**:
- [ ] All saliency methods
- [ ] Unified neural XAI API
- [ ] Comprehensive examples
- [ ] v0.5.0 release

**Reading Focus**: docs/roadmap.md (Saliency Maps)

---

### Phase 6: Production Features (v0.6.0) - Q2 2026

**Objective**: Performance optimization and production-ready features

#### Week 35-36: Performance Optimization

**Tasks**:
1. Batch explanation generation:
   ```elixir
   # lib/crucible_xai/batch.ex
   defmodule CrucibleXAI.Batch do
     def explain_batch(instances, predict_fn, method, opts \\ [])
     def parallel_explain(instances, predict_fn, max_concurrency)
   end
   ```

2. EXLA GPU acceleration:
   ```elixir
   # Enable GPU backend
   Nx.default_backend(EXLA.Backend)
   ```

3. Caching strategies:
   ```elixir
   # lib/crucible_xai/cache.ex
   defmodule CrucibleXAI.Cache do
     def cache_samples(instance, samples)
     def get_or_compute(key, compute_fn)
   end
   ```

4. Streaming for large datasets:
   ```elixir
   def stream_explanations(data_stream, predict_fn, opts \\ [])
   ```

**Deliverables**:
- [ ] Batch processing
- [ ] GPU acceleration via EXLA
- [ ] Caching system
- [ ] Streaming support

**Reading Focus**: docs/architecture.md (Performance Considerations), docs/roadmap.md (Performance Optimization)

#### Week 37-38: Explanation Management

**Tasks**:
1. Persistence:
   ```elixir
   # lib/crucible_xai/persistence.ex
   defmodule CrucibleXAI.Persistence do
     def save_explanation(explanation, path)
     def load_explanation(path)
     def version_tracking(explanation, version)
   end
   ```

2. Comparison tools:
   ```elixir
   # lib/crucible_xai/comparison.ex
   defmodule CrucibleXAI.Comparison do
     def compare_across_models(instance, models, method)
     def compare_across_instances(instances, model, method)
     def explanation_similarity(exp1, exp2)
   end
   ```

3. Aggregation:
   ```elixir
   def aggregate_explanations(explanations, opts \\ [])
   def summary_statistics(explanations)
   def distribution_analysis(explanations)
   ```

**Deliverables**:
- [ ] Persistence system
- [ ] Comparison tools
- [ ] Aggregation methods
- [ ] Version tracking

**Reading Focus**: docs/roadmap.md (Model Management)

#### Week 39-40: Quality Assurance

**Tasks**:
1. Faithfulness metrics:
   ```elixir
   # lib/crucible_xai/validation/faithfulness.ex
   defmodule CrucibleXAI.Validation.Faithfulness do
     def faithfulness_test(model, instance, attributions, opts \\ [])
     def monotonicity_test(model, instance, attributions)
     def infidelity(model, instance, attributions, opts \\ [])
   end
   ```

2. Sensitivity analysis:
   ```elixir
   # lib/crucible_xai/validation/sensitivity.ex
   def sensitivity_test(model, instance, attribution_method, opts \\ [])
   ```

3. Robustness testing:
   ```elixir
   def robustness_test(model, instance, method, noise_levels)
   ```

4. Validation suite:
   ```elixir
   def comprehensive_validation(model, instances, methods)
   ```

**Deliverables**:
- [ ] Faithfulness metrics
- [ ] Sensitivity analysis
- [ ] Robustness testing
- [ ] Validation suite
- [ ] v0.6.0 release

**Reading Focus**: docs/feature_attribution.md (Validation and Metrics)

---

### Phase 7: Ecosystem Integration (v0.7.0) - Q3 2026

**Objective**: Full integration with Crucible framework and external tools

#### Week 41-42: Crucible Integration

**Tasks**:
1. Seamless model integration:
   ```elixir
   # lib/crucible_xai/integrations/crucible.ex
   defmodule CrucibleXAI.Integrations.Crucible do
     def explain_crucible_model(model, instance, opts \\ [])
   end
   ```

2. CrucibleBench integration:
   ```elixir
   def explain_benchmark_results(benchmark, instances, opts \\ [])
   def statistical_significance_of_explanations(exp1, exp2)
   ```

3. Workflow automation:
   ```elixir
   def auto_explain_pipeline(model, test_data, opts \\ [])
   def explanation_based_model_selection(models, validation_data, criteria)
   ```

**Deliverables**:
- [ ] Crucible model integration
- [ ] CrucibleBench integration
- [ ] Workflow automation
- [ ] Pipeline tools

**Reading Focus**: docs/architecture.md (Integration Points), docs/roadmap.md (Crucible Framework Integration)

#### Week 43-44: Export & External Tools

**Tasks**:
1. Export formats:
   ```elixir
   # lib/crucible_xai/export.ex
   defmodule CrucibleXAI.Export do
     def to_json(explanation)
     def to_html_report(explanations, opts \\ [])
     def to_latex(explanation, opts \\ [])
     def to_interactive_dashboard(explanations)
   end
   ```

2. Model format support:
   ```elixir
   # lib/crucible_xai/model_wrappers.ex
   def wrap_onnx_model(onnx_path)
   def wrap_axon_model(axon_model)
   def wrap_custom_model(predict_fn, metadata)
   ```

3. Integration examples:
   - Web application integration
   - API endpoints for explanations
   - Dashboard embedding

**Deliverables**:
- [ ] Multiple export formats
- [ ] Model wrapper support
- [ ] Integration examples
- [ ] API documentation

**Reading Focus**: docs/roadmap.md (External Tool Support)

#### Week 45-46: Documentation & Case Studies

**Tasks**:
1. Comprehensive API docs:
   - Complete ExDoc coverage
   - Function examples
   - Type specifications

2. Tutorial series:
   - Getting started guide
   - Advanced techniques
   - Best practices

3. Case studies:
   ```elixir
   # examples/case_studies/
   # - healthcare_diagnosis.exs
   # - financial_credit_scoring.exs
   # - nlp_sentiment_analysis.exs
   # - computer_vision_classification.exs
   ```

4. Troubleshooting guide:
   - Common issues
   - Performance tuning
   - Debugging explanations

**Deliverables**:
- [ ] Complete API documentation
- [ ] Tutorial series
- [ ] 4+ case studies
- [ ] Troubleshooting guide
- [ ] v0.7.0 release

**Reading Focus**: docs/roadmap.md (Documentation and Examples)

---

### Phase 8: Advanced Features (v0.8.0+) - Q4 2026 and beyond

**Objective**: Research features and domain-specific tools

#### Week 47-48: Concept-based Explanations

**Tasks**:
1. Implement TCAV (Testing with Concept Activation Vectors):
   ```elixir
   # lib/crucible_xai/concepts/tcav.ex
   defmodule CrucibleXAI.Concepts.TCAV do
     def test_concept(model, concept_examples, random_examples, layer)
     def compute_cav(concept_activations, random_activations)
     def directional_derivative(model, instance, cav, layer)
   end
   ```

2. Concept bottleneck models:
   ```elixir
   def train_concept_bottleneck(model, concepts, training_data)
   def explain_via_concepts(model, instance, concepts)
   ```

**Deliverables**:
- [ ] TCAV implementation
- [ ] Concept bottleneck support
- [ ] Examples with concepts
- [ ] Research documentation

**Reading Focus**: docs/roadmap.md (Concept-based Explanations)

#### Week 49-50: Domain-Specific Tools

**Tasks**:
1. NLP-specific explanations:
   ```elixir
   # lib/crucible_xai/domain/nlp.ex
   defmodule CrucibleXAI.Domain.NLP do
     def token_importance(model, text, tokenizer)
     def attention_analysis(model, text)
     def semantic_similarity_explanation(model, text, similar_texts)
   end
   ```

2. Computer Vision:
   ```elixir
   # lib/crucible_xai/domain/vision.ex
   def saliency_map(model, image)
   def segmentation_mask(model, image, class_idx)
   def object_detection_explanation(model, image, detections)
   ```

3. Graph Neural Networks:
   ```elixir
   # lib/crucible_xai/domain/gnn.ex
   def node_importance(model, graph, node_idx)
   def edge_importance(model, graph, edge_idx)
   def subgraph_explanation(model, graph, target_nodes)
   ```

4. Time series:
   ```elixir
   # lib/crucible_xai/domain/time_series.ex
   def temporal_lime(model, time_series, opts \\ [])
   def temporal_shap(model, time_series, opts \\ [])
   def event_attribution(model, time_series, event_idx)
   ```

**Deliverables**:
- [ ] NLP tools
- [ ] Computer Vision tools
- [ ] GNN support
- [ ] Time series methods
- [ ] Domain examples

**Reading Focus**: docs/roadmap.md (Domain-Specific Tools)

#### Week 51-52: Fairness & Research Features

**Tasks**:
1. Fairness analysis integration:
   ```elixir
   # lib/crucible_xai/fairness.ex
   defmodule CrucibleXAI.Fairness do
     def disparate_impact_detection(model, data, sensitive_features)
     def bias_attribution(model, instance, sensitive_features)
     def fair_counterfactuals(instance, predict_fn, protected_features)
   end
   ```

2. Causal explanations:
   ```elixir
   # lib/crucible_xai/causal.ex
   def causal_attribution(model, instance, causal_graph)
   def do_calculus_explanation(model, intervention, causal_graph)
   ```

3. Research publications:
   - Write papers on novel techniques
   - Conference presentations
   - Academic collaborations

**Deliverables**:
- [ ] Fairness analysis tools
- [ ] Causal explanations
- [ ] Research publications
- [ ] v0.8.0+ releases

**Reading Focus**: docs/roadmap.md (Research Features, Fairness Analysis)

---

## Development Workflow

### Daily Workflow

1. **Morning**: Review required reading for current phase
2. **Development**: Implement features following TDD approach
3. **Testing**: Write property-based tests for mathematical properties
4. **Documentation**: Update ExDoc and examples
5. **Review**: End-of-day code review and refactoring

### Weekly Workflow

1. **Monday**: Plan week's tasks from buildout plan
2. **Tuesday-Thursday**: Development and testing
3. **Friday**: Code review, documentation, prepare next week
4. **Weekly retrospective**: Review progress, adjust timeline

### Testing Standards

- **Unit tests**: Cover all functions, edge cases, boundary conditions
- **Property-based tests**: Verify mathematical properties (SHAP efficiency, local fidelity)
- **Integration tests**: Test full explanation workflows
- **Performance tests**: Benchmark critical paths
- **Target coverage**: > 90% for production code

### Documentation Standards

- **Inline docs**: Every public function has @doc with examples
- **Module docs**: Comprehensive @moduledoc with overview
- **Type specs**: All public functions have @spec
- **Examples**: Real-world usage examples in docs
- **Guides**: High-level guides for common workflows

---

## Key Implementation Principles

### 1. Model-Agnostic Design

All explanation methods accept a prediction function, enabling use with any model:

```elixir
# Good: model-agnostic interface
predict_fn :: (input :: any()) -> prediction :: number() | Nx.Tensor.t()

explanation = CrucibleXAI.explain(
  instance: instance,
  predict_fn: predict_fn  # Works with any model
)

# Works with:
# - Axon neural networks
# - Scholar models
# - Custom Elixir models
# - ONNX models
# - Any black-box model
```

### 2. Nx Tensor Operations

Always use Nx tensors for numerical computations:

```elixir
# Good: Vectorized Nx operations
def euclidean_distance(samples, instance) do
  samples
  |> Nx.tensor()
  |> Nx.subtract(Nx.tensor(instance))
  |> Nx.pow(2)
  |> Nx.sum(axes: [1])
  |> Nx.sqrt()
end

# Bad: List operations
def euclidean_distance(samples, instance) do
  Enum.map(samples, fn sample ->
    sample
    |> Enum.zip(instance)
    |> Enum.map(fn {a, b} -> (a - b) ** 2 end)
    |> Enum.sum()
    |> :math.sqrt()
  end)
end
```

### 3. Pure Functions

All computations are pure functions with no side effects:

```elixir
# Returns new explanation, doesn't modify input
def explain(instance, predict_fn, opts) do
  samples = generate_samples(instance, opts)
  predictions = predict_fn.(samples)
  build_explanation(samples, predictions, opts)
end
```

### 4. Composability

Design functions to compose naturally:

```elixir
instance
|> CrucibleXAI.LIME.explain(predict_fn)
|> CrucibleXAI.Explanation.top_features(10)
|> CrucibleXAI.Visualization.plot()
```

### 5. Performance Optimization

Optimize for production use cases:

```elixir
# Batch predictions
predictions = predict_fn.(Nx.stack(samples))  # Single call

# Parallel explanations
instances
|> Task.async_stream(fn i -> explain(i, predict_fn) end)
|> Enum.map(fn {:ok, result} -> result end)

# GPU acceleration
Nx.default_backend(EXLA.Backend)
```

### 6. Configuration Flexibility

Support extensive configuration with sensible defaults:

```elixir
@default_opts [
  num_samples: 5000,
  kernel_width: 0.75,
  num_features: 10
]

def explain(instance, predict_fn, opts \\ []) do
  config = Keyword.merge(@default_opts, opts)
  # Implementation
end
```

---

## Quality Gates

### Phase 1 Gate (v0.1.0)
- [ ] LIME fully implemented with all sampling strategies
- [ ] Test coverage > 90%
- [ ] Documentation complete with examples
- [ ] `mix hex.build` succeeds
- [ ] Local fidelity > 0.9 on test cases
- [ ] Performance: 1000 explanations in < 30s (CPU)

### Phase 2 Gate (v0.2.0)
- [ ] SHAP methods implemented (KernelSHAP, SamplingShap, LinearSHAP)
- [ ] Feature attribution methods complete
- [ ] SHAP efficiency property validated
- [ ] Integration tests passing
- [ ] Performance: KernelSHAP in < 5s for 50 features

### Phase 3 Gate (v0.3.0)
- [ ] Global interpretability tools complete
- [ ] PDP, ICE, ALE implemented
- [ ] Visualization utilities functional
- [ ] LiveBook tutorials created
- [ ] Performance acceptable for production use

### Phase 4 Gate (v0.4.0)
- [ ] Counterfactual generation working
- [ ] Anchors implementation complete
- [ ] Example-based methods functional
- [ ] Diversity and feasibility constraints validated

### Phase 5 Gate (v0.5.0)
- [ ] Neural network methods complete (LRP, DeepLIFT, GradCAM)
- [ ] Axon integration seamless
- [ ] Saliency maps for vision models
- [ ] Attention visualization for transformers

### Phase 6 Gate (v0.6.0)
- [ ] Performance optimized (EXLA, batching, caching)
- [ ] Validation suite complete
- [ ] Faithfulness metrics > 0.85
- [ ] Production features ready

### Phase 7 Gate (v0.7.0)
- [ ] Crucible integration complete
- [ ] Export formats working
- [ ] Case studies published
- [ ] Community adoption metrics met

### Phase 8 Gate (v0.8.0+)
- [ ] Research features implemented
- [ ] Domain-specific tools available
- [ ] Publications submitted
- [ ] Innovation demonstrated

---

## Resources

### Elixir/Nx Resources
- [Nx Documentation](https://hexdocs.pm/nx)
- [Axon Documentation](https://hexdocs.pm/axon)
- [EXLA Documentation](https://hexdocs.pm/exla)
- [VegaLite for Elixir](https://hexdocs.pm/vega_lite)

### XAI Research Papers
- Ribeiro et al. (2016) - "Why Should I Trust You?" (LIME)
- Lundberg & Lee (2017) - A Unified Approach to Interpreting Model Predictions (SHAP)
- Sundararajan et al. (2017) - Axiomatic Attribution for Deep Networks (Integrated Gradients)
- Shrikumar et al. (2017) - Learning Important Features (DeepLIFT)
- Selvaraju et al. (2017) - Grad-CAM (Visual Explanations)

### XAI Books & Guides
- Molnar, C. (2022) - *Interpretable Machine Learning*
- Samek et al. (2019) - *Explainable AI: Interpreting, Explaining and Visualizing Deep Learning*

### Community
- ElixirForum ML section
- North Shore AI organization
- XAI research community
- Crucible framework contributors

---

## Success Criteria

### Technical Success
- All explanation methods mathematically correct
- High performance (GPU acceleration, batching)
- Production-ready reliability
- Comprehensive validation suite
- SHAP efficiency property: Σ φᵢ = f(x) - f(baseline)
- LIME local fidelity: R² > 0.9

### Adoption Success
- 1000+ Hex downloads
- 100+ GitHub stars
- 20+ production deployments
- 10+ community contributions
- Active community discussions
- Integration with major Elixir ML projects

### Research Success
- 2+ conference publications
- Novel XAI techniques in Elixir/Nx
- Academic collaborations
- Benchmark comparisons with Python libraries
- Contribution to XAI research community

### Community Success
- 30+ contributors
- Active issue resolution
- Third-party integrations
- Educational content (tutorials, videos, workshops)
- Industry partnerships

---

## Conclusion

This buildout plan provides a comprehensive roadmap from basic LIME implementation to advanced research features. By following this plan and thoroughly reading the required documentation, developers can build a world-class explainable AI library for the Elixir ecosystem.

The phased approach ensures:
- **Early value delivery** with LIME in Phase 1
- **Progressive capability building** through SHAP and attribution methods
- **Production readiness** with optimization and validation
- **Innovation** through research features and domain-specific tools

CrucibleXAI will enable the Elixir ML community to build trustworthy, interpretable AI systems with comprehensive explanation capabilities.

**Next Step**: Begin with Phase 1, Week 1-2 after completing all required reading.

---

*Document Version: 1.0*
*Last Updated: 2025-10-10*
*Maintainer: North Shore AI*

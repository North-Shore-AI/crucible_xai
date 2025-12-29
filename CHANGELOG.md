# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for Future Releases
- TreeSHAP for decision tree models
- Advanced visualizations for all methods
- CrucibleTrace integration
- Counterfactual explanations (DiCE)
- Neural network-specific methods (LRP, DeepLIFT, GradCAM)

## [0.4.0] - 2025-12-28

### Added - Pipeline Stage Integration

Integration with the Crucible IR pipeline framework, enabling CrucibleXAI to be used as a pipeline stage in larger ML reliability experiments.

#### New Modules

**CrucibleXAI.Stage**
- Pipeline stage implementation for Crucible framework integration
- `run/2` function accepting context with model function and instances
- `describe/1` function for stage introspection and metadata
- Support for LIME, SHAP (all variants), and feature importance methods
- Configurable via `experiment.reliability.xai` in context or direct options
- Parallel batch processing support
- Comprehensive error handling with graceful degradation

#### Stage Capabilities

**Input Requirements:**
- `model_fn` - Prediction function
- `instances` or `instance` - Data to explain
- `background_data` - Required for SHAP methods (optional for LIME)
- `experiment.reliability.xai` - Optional configuration via CrucibleIR

**Output:**
- Adds `:xai` key to context with explanation results
- Includes metadata (timestamp, instance count)
- Preserves all methods run and their results

**Supported Methods:**
- `:lime` - LIME explanations
- `:shap`, `:kernel_shap` - KernelSHAP approximation
- `:linear_shap` - Exact SHAP for linear models
- `:sampling_shap` - Monte Carlo SHAP approximation
- `:feature_importance` - Permutation importance

**Configuration Options:**
- `methods` - List of XAI methods to run
- `lime_opts` - LIME-specific options (num_samples, kernel, etc.)
- `shap_opts` - SHAP-specific options (num_samples, method, etc.)
- `feature_importance_opts` - Permutation importance options
- `parallel` - Enable parallel batch processing

#### Dependencies

- Added `{:crucible_ir, "~> 0.1.1"}` dependency
- Enables integration with Crucible experiment framework
- Provides standardized configuration via CrucibleIR.Reliability.* structs

#### Testing

- 25 new comprehensive tests for Stage module
- Tests for all supported XAI methods
- Error handling and edge case coverage
- Configuration extraction and option passing
- Metadata validation
- Total test count: 362+ tests (337 existing + 25 new)

#### Documentation

- Complete API documentation for Stage module
- Usage examples with context structure
- Integration guide for Crucible pipelines
- Method selection and configuration examples

#### Use Cases Enabled

**Pipeline Integration:**
- Use CrucibleXAI as a stage in multi-step ML experiments
- Combine with crucible_bench for statistical analysis
- Integrate with crucible_telemetry for metrics tracking
- Chain with other Crucible reliability mechanisms

**Experiment Workflows:**
- Standardized XAI analysis across experiments
- Reproducible explanation generation
- Automated explanation quality assessment
- Multi-method comparison in pipelines

**Example Usage:**

```elixir
# In a Crucible pipeline
context = %{
  model_fn: &MyModel.predict/1,
  instances: test_data,
  background_data: training_sample,
  experiment: %{
    reliability: %{
      xai: %{
        methods: [:lime, :shap],
        lime_opts: %{num_samples: 1000},
        parallel: true
      }
    }
  }
}

{:ok, updated_context} = CrucibleXAI.Stage.run(context)
# updated_context.xai contains LIME and SHAP explanations
```

### Code Quality Improvements

- Resolved all Credo issues including complexity refactoring
- Fixed all Dialyzer warnings and type specifications
- Refactored long/complex functions to reduce cyclomatic complexity
- Updated alias ordering across all modules for consistency
- Replaced `Enum.map |> Enum.join` with `Enum.map_join`
- Improved test isolation with logger level configuration

### Breaking Changes

None - fully backward compatible with v0.3.0. The Stage module is a new addition that doesn't affect existing LIME/SHAP/FeatureAttribution APIs.

### Quality Metrics

- 25+ new tests added
- Total test count: 362+ tests
- Zero compilation warnings
- Passes `mix credo --strict` with no issues
- Passes `mix dialyzer` with no warnings
- Full type specifications (@spec) for all Stage functions
- 100% documentation coverage for Stage module

## [0.3.0] - 2025-11-25

### Added - Validation & Quality Metrics Suite

A comprehensive validation framework for measuring explanation quality, reliability, and trustworthiness. This major enhancement enables production deployment with confidence and rigorous research validation.

#### New Modules

**CrucibleXAI.Validation.Faithfulness**
- Feature removal correlation testing
- Monotonicity verification for explanation reliability
- Spearman and Pearson correlation support
- Multiple baseline strategies (zero, mean, median)
- Per-feature importance validation
- Comprehensive faithfulness reports

**CrucibleXAI.Validation.Infidelity**
- Perturbation-based explanation error quantification
- Mean squared error between predicted and actual model changes
- Multiple perturbation strategies (Gaussian, uniform)
- Normalized and unnormalized scoring
- Cross-method comparison capabilities
- Sensitivity analysis across perturbation magnitudes

**CrucibleXAI.Validation.Sensitivity**
- Input perturbation sensitivity testing
- Hyperparameter sensitivity analysis
- Cross-method consistency verification
- Stability scoring (0-1 scale)
- Per-feature variation analysis
- Adaptive sampling strategies

**CrucibleXAI.Validation.Axioms**
- Completeness axiom testing (SHAP, Integrated Gradients)
- Symmetry axiom verification
- Dummy (null player) axiom validation
- Linearity axiom for linear models
- Comprehensive axiom validation suite
- Method-specific axiom testing

**CrucibleXAI.Validation (Main API)**
- `comprehensive_validation/4` - Full quality assessment
- `quick_validation/4` - Fast quality checks for production
- `benchmark_methods/4` - Compare multiple explanation methods
- Overall quality scoring (0-1 scale)
- Human-readable validation summaries
- Quality gate pass/fail determinations

#### Main API Enhancements

Added to `CrucibleXai` module:
- `validate_explanation/4` - Comprehensive validation
- `quick_validate/4` - Fast quality check
- `measure_faithfulness/4` - Faithfulness testing
- `compute_infidelity/4` - Infidelity measurement

#### Metrics & Scores

**Faithfulness Score**: -1 to 1 (higher is better)
- Measures correlation between feature importance and prediction change
- >0.9: Excellent, 0.7-0.9: Good, 0.5-0.7: Fair, <0.5: Poor

**Infidelity Score**: 0 to ∞ (lower is better)
- Quantifies explanation error via perturbation testing
- <0.02: Excellent, 0.02-0.05: Good, 0.05-0.10: Acceptable, >0.10: Poor

**Stability Score**: 0 to 1 (higher is better)
- Measures robustness to input perturbations
- >0.95: Excellent, 0.85-0.95: Good, 0.70-0.85: Acceptable, <0.70: Poor

**Quality Score**: 0 to 1 (higher is better)
- Weighted combination of all metrics (40% faithfulness + 40% infidelity + 20% axioms)
- ≥0.85: Production-ready, ≥0.70: Acceptable, ≥0.50: Use with caution, <0.50: Unreliable

### Documentation

- Complete API documentation for all validation modules
- Usage examples for each validation metric
- Production monitoring examples
- Method comparison examples
- Best practices guide for validation
- Integration with existing LIME/SHAP/Gradient methods

### Academic Foundation

Based on peer-reviewed research:
- Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations", NeurIPS
- Hooker et al. (2019) "A Benchmark for Interpretability Methods in Deep Neural Networks", NeurIPS
- Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks", ICML
- Shapley (1953) "A Value for N-person Games"

### Quality Metrics

- 60+ new tests added (faithfulness, infidelity, sensitivity, axioms)
- Total test count: 337+ tests
- Test coverage increased to >96%
- Zero compilation warnings
- Full type specifications (@spec) for all functions

### Breaking Changes

None - fully backward compatible with v0.2.1

### Use Cases Enabled

**Production Deployment**
- Automated quality gates for explanation deployment
- Real-time explanation quality monitoring
- Alerting for explanation quality degradation
- A/B testing of explanation strategies

**Research**
- Rigorous explanation method evaluation
- Comparative analysis across techniques
- Publication-quality validation metrics
- Reproducible validation experiments

**Compliance**
- Auditable explanation quality scores
- Evidence of explanation reliability
- Regulatory certification support
- Transparent quality assessment

### Performance

- Faithfulness: ~50ms per explanation
- Infidelity: ~100ms per explanation (100 perturbations)
- Sensitivity: ~2.5s per explanation (parallelizable)
- Axioms: ~10-100ms per explanation
- Quick validation: ~150ms per explanation

## [0.2.1] - 2025-10-29

### Added - SHAP Enhancements

#### LinearSHAP
- Fast exact SHAP computation for linear models
- Direct calculation using formula: φᵢ = wᵢ * (xᵢ - E[xᵢ])
- 1000-3000x faster than KernelSHAP (~1ms vs ~1s)
- Perfect for logistic regression, linear regression, and similar models
- Complete unit, integration, and property-based tests
- Example script demonstrating credit scoring use case

#### SamplingShap
- Monte Carlo approximation of SHAP values
- Random permutation sampling for feature attribution
- Faster than KernelSHAP with comparable accuracy
- Model-agnostic approach suitable for any model type
- Configurable number of permutation samples
- Full test coverage with property-based testing

### Documentation
- Added Example 11: LinearSHAP for Linear Models
- Updated SHAP module documentation with all methods
- Added usage examples and comparisons

#### Parallel Batch Processing
- Parallel execution of batch explanations using Task.async_stream for both LIME and SHAP
- Configurable concurrency control with `:max_concurrency` option
- Configurable timeout per instance (`:timeout` option)
- Graceful error handling with `:on_error` option (`:skip` or `:raise`)
- Backwards compatible - defaults to sequential processing
- Performance scaling with available CPU cores
- Order-preserving results
- Significant performance improvement (40-60%) for large batches on multi-core systems

#### Gradient-based Attribution Methods
- **Gradient × Input**: Simple fast method: attribution_i = (∂f/∂x_i) * x_i
- **Integrated Gradients**: Axiomatic method with completeness guarantee
- **SmoothGrad**: Noise-reduced attributions via averaging noisy gradients
- Full automatic differentiation using Nx.Defn.grad
- Configurable parameters for all gradient methods
- Complete mathematical formulas and research references
- 23 comprehensive tests (21 unit + 2 property-based)

#### Occlusion-based Attribution Methods
- **Feature Occlusion**: Measure importance by removing features individually
- **Sliding Window Occlusion**: Occlude windows of consecutive features
- **Occlusion Sensitivity**: Normalized sensitivity scores with optional absolute values
- **Batch Occlusion**: Parallel processing for multiple instances
- Model-agnostic (works with any black-box model, no gradients needed)
- Configurable baseline values for occlusion
- Configurable window size and stride for sliding windows
- Intuitive interpretation of feature importance
- 19 comprehensive tests (16 unit + 3 property-based)

#### Global Interpretability Methods
- **Partial Dependence Plots (PDP)**: Shows marginal effect of features
  - 1D PDP for single feature analysis
  - 2D PDP for feature interaction analysis
  - Auto-detects feature ranges or uses custom ranges
  - Configurable grid resolution
  - Robust handling of edge cases (nil values, min==max)
- **Individual Conditional Expectation (ICE)**: Shows per-instance prediction curves
  - One curve per instance revealing heterogeneity
  - Centered ICE for relative change visualization
  - Average of ICE equals PDP
  - Detects non-additive effects
- **Accumulated Local Effects (ALE)**: Robust alternative to PDP for correlated features
  - Avoids extrapolation to unrealistic feature combinations
  - Quantile-based binning for equal representation
  - Centered effects around zero
  - Better handles feature dependencies
- **H-Statistic**: Friedman's interaction detection
  - Measures interaction strength (0=none to 1=pure)
  - Pairwise interaction analysis
  - All-pairs scanning with find_all_interactions
  - Filtering and sorting by strength
  - Automatic interpretation (None/Weak/Moderate/Strong)
- Efficient grid generation and batch prediction
- 65 comprehensive tests (61 unit + 4 property-based)

### Test Coverage
- Added 13 tests for LinearSHAP (unit + property + integration)
- Added 12 tests for SamplingShap (unit + property + integration)
- Added 10 tests for LIME parallel batch processing
- Added 6 tests for SHAP parallel batch processing
- Added 23 tests for gradient attribution methods (21 unit + 2 property)
- Added 19 tests for occlusion attribution methods (16 unit + 3 property)
- Added 26 tests for PDP and ICE (24 unit + 2 property)
- Added 13 tests for ALE (11 unit + 2 property)
- Added 13 tests for H-statistic interactions (11 unit + 2 property)
- Total: 277 tests (11 doctests + 34 properties + 232 unit tests)
- 100% pass rate maintained
- >93% code coverage

### Performance
- LinearSHAP: <2ms per explanation (exact values)
- SamplingShap: ~100-500ms with 500-2000 samples (approximate)
- KernelSHAP: ~1s with 2000 coalitions (approximate)
- Gradient × Input: <1ms per attribution
- Integrated Gradients: ~5-50ms (depends on steps, default: 50)
- SmoothGrad: ~10-100ms (depends on samples, default: 50)
- Feature Occlusion: ~1-5ms per feature (model-agnostic)
- Sliding Window: ~1-10ms per window position
- PDP 1D: ~10-50ms depending on grid points and dataset size
- PDP 2D: ~50-200ms for grid combinations
- ICE: ~10-100ms depending on instances and grid points
- ALE: ~10-100ms depending on bins and dataset size
- H-Statistic: ~50-300ms per feature pair (requires 3 PDP computations)
- Parallel batch processing: 40-60% speed improvement

## [0.2.0] - 2025-10-20

### Added - Core XAI Implementation

#### LIME (Local Interpretable Model-agnostic Explanations)
- Complete LIME algorithm with local linear approximations
- Multiple sampling strategies: Gaussian, Uniform, Categorical, Combined
- Kernel functions: Exponential, Cosine with multiple distance metrics
- Interpretable models: Weighted Linear Regression and Ridge Regression
- Feature selection: Highest weights, Forward selection, Lasso-approximation
- Batch processing support for multiple instances
- `CrucibleXai.explain/3` and `CrucibleXai.explain_batch/3` API

#### SHAP (SHapley Additive exPlanations)
- KernelSHAP implementation with coalition sampling
- SHAP kernel weight calculation using game theory
- Shapley value computation via weighted regression
- Property validation: Additivity, Symmetry, Dummy properties
- Background data support for baseline computation
- `CrucibleXai.explain_shap/4` API
- Batch SHAP explanations

#### Feature Attribution
- Permutation Importance with multiple metrics (MSE, MAE, Accuracy)
- Statistical validation with mean and standard deviation
- Support for num_repeats configuration
- Top-k feature selection utility
- `CrucibleXai.feature_importance/3` API

#### Visualization
- HTML generation for LIME explanations
- HTML generation for SHAP values
- LIME vs SHAP comparison views
- Chart.js integration for interactive bar charts
- Light and dark theme support
- Custom feature naming
- File export functionality

### Test Coverage
- 141 tests total (111 unit + 19 property-based + 11 doctests)
- 100% pass rate
- 87.1% code coverage
- Property-based tests for mathematical correctness
- Integration tests for end-to-end workflows
- Shapley property validation tests

### Quality Assurance
- Zero compiler warnings (strict `--warnings-as-errors`)
- Dialyzer type checking (0 errors, 4 acceptable supertype warnings)
- Complete type specifications on all public functions
- Comprehensive documentation with examples
- All public API documented with doctests

### Documentation
- Complete README with quick start examples
- API documentation for all modules
- LIME vs SHAP comparison guide
- Visual algorithm explanations
- Performance benchmarks
- Use case examples (debugging, comparison, validation)
- Future direction technical specification

### Performance
- LIME: <50ms per explanation (5000 samples)
- SHAP: ~1s per explanation (2000 coalitions)
- R² scores: >0.95 for linear models
- Batch processing support

## [0.1.0] - 2025-10-10

### Added
- Initial project structure
- Core module architecture
- Documentation framework with ExDoc and Mermaid support
- Comprehensive README with usage examples
- Technical design documents:
  - Architecture overview
  - LIME implementation design
  - Feature attribution methods
  - Implementation roadmap
- MIT License
- Hex package configuration
- Basic testing framework

### Documentation
- README with comprehensive examples
- Architecture documentation
- LIME design document
- Feature attribution guide
- Development roadmap

[Unreleased]: https://github.com/North-Shore-AI/crucible_xai/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/North-Shore-AI/crucible_xai/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/North-Shore-AI/crucible_xai/releases/tag/v0.1.0

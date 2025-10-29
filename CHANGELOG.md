# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.3.0
- Occlusion-based attribution methods
- Global interpretability (Partial Dependence Plots, ICE)
- Feature interaction detection (H-statistic)
- CrucibleTrace integration
- TreeSHAP for decision tree models

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
- **Individual Conditional Expectation (ICE)**: Shows per-instance prediction curves
  - One curve per instance revealing heterogeneity
  - Centered ICE for relative change visualization
  - Average of ICE equals PDP
  - Detects non-additive effects
- Efficient grid generation and batch prediction
- 26 comprehensive tests (24 unit + 2 property-based)

### Test Coverage
- Added 13 tests for LinearSHAP (unit + property + integration)
- Added 12 tests for SamplingShap (unit + property + integration)
- Added 10 tests for LIME parallel batch processing
- Added 6 tests for SHAP parallel batch processing
- Added 23 tests for gradient attribution methods (21 unit + 2 property)
- Added 19 tests for occlusion attribution methods (16 unit + 3 property)
- Added 26 tests for global interpretability (24 unit + 2 property)
- Total: 251 tests (11 doctests + 30 properties + 210 unit tests)
- 100% pass rate maintained

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

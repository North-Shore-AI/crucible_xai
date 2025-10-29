# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned for v0.3.0
- Gradient-based attribution (Gradient×Input, Integrated Gradients)
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

### Test Coverage
- Added 13 tests for LinearSHAP (unit + property + integration)
- Added 12 tests for SamplingShap (unit + property + integration)
- Added 10 tests for LIME parallel batch processing
- Added 6 tests for SHAP parallel batch processing
- Total: 183 tests (11 doctests + 21 properties + 151 unit tests)
- 100% pass rate maintained

### Performance
- LinearSHAP: <2ms per explanation (exact values)
- SamplingShap: ~100-500ms with 500-2000 samples (approximate)
- KernelSHAP: ~1s with 2000 coalitions (approximate)

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

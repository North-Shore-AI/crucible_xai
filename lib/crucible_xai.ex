defmodule CrucibleXai do
  @moduledoc """
  CrucibleXAI - Explainable AI (XAI) Library for Elixir

  A comprehensive library for explaining machine learning model predictions
  using state-of-the-art interpretability techniques. Built on Nx for
  high-performance numerical computing.

  ## Features

  - **LIME** (Local Interpretable Model-agnostic Explanations)
    - Explain any black-box model locally with interpretable linear models
    - Multiple sampling strategies (Gaussian, Uniform, Categorical)
    - Flexible kernel functions for proximity weighting
    - Feature selection methods (highest weights, forward selection, Lasso)

  - **Model-Agnostic**: Works with any prediction function
  - **High Performance**: Built on Nx tensors for efficient computation
  - **Flexible**: Extensive configuration options
  - **Well-Tested**: Comprehensive test suite with property-based testing

  ## Quick Start

      # Explain a model prediction
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 1.0 end
      instance = [1.0, 2.0]

      explanation = CrucibleXai.explain(instance, predict_fn)

      # View explanation
      IO.puts(CrucibleXai.Explanation.to_text(explanation))

  ## Main Modules

  - `CrucibleXai.LIME` - LIME explanations
  - `CrucibleXai.SHAP` - SHAP (Shapley values) explanations
  - `CrucibleXai.Explanation` - Explanation structure and utilities
  - `CrucibleXai.Validation` - Explanation quality metrics and validation (v0.3.0+)
  - `CrucibleXai.LIME.Sampling` - Data perturbation strategies
  - `CrucibleXai.LIME.Kernels` - Proximity weighting functions
  - `CrucibleXai.LIME.InterpretableModels` - Linear regression models
  - `CrucibleXai.LIME.FeatureSelection` - Feature selection methods

  ## References

  - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
    "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
  """

  alias CrucibleXAI.{Explanation, FeatureAttribution, LIME, SHAP, Validation}

  @doc """
  Explain a model prediction using LIME.

  Convenience function that delegates to `CrucibleXai.LIME.explain/3`.

  ## Parameters
    * `instance` - The instance to explain
    * `predict_fn` - Function that takes input and returns prediction
    * `opts` - Options (see `CrucibleXai.LIME` for details)

  ## Returns
    `%Explanation{}` struct with feature weights and metadata

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> explanation = CrucibleXai.explain([5.0], predict_fn, num_samples: 100)
      iex> explanation.method
      :lime
  """
  @spec explain(
          list() | Nx.Tensor.t(),
          (any() -> number() | Nx.Tensor.t()),
          Keyword.t()
        ) :: Explanation.t()
  def explain(instance, predict_fn, opts \\ []) do
    LIME.explain(instance, predict_fn, opts)
  end

  @doc """
  Explain multiple instances.

  Convenience function that delegates to `CrucibleXai.LIME.explain_batch/3`.

  ## Parameters
    * `instances` - List of instances to explain
    * `predict_fn` - Prediction function
    * `opts` - Options

  ## Returns
    List of `%Explanation{}` structs
  """
  @spec explain_batch(
          list(),
          (any() -> number() | Nx.Tensor.t()),
          Keyword.t()
        ) :: list(Explanation.t())
  def explain_batch(instances, predict_fn, opts \\ []) do
    LIME.explain_batch(instances, predict_fn, opts)
  end

  @doc """
  Explain using SHAP (Shapley values).

  Convenience function that delegates to `CrucibleXAI.SHAP.explain/4`.

  ## Parameters
    * `instance` - The instance to explain
    * `background_data` - Background dataset for baseline
    * `predict_fn` - Prediction function
    * `opts` - Options (see `CrucibleXAI.SHAP` for details)

  ## Returns
    Map of feature_index => shapley_value

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> shap = CrucibleXai.explain_shap([5.0], [[0.0]], predict_fn, num_samples: 500)
      iex> is_map(shap)
      true
  """
  @spec explain_shap(list() | Nx.Tensor.t(), list(), function(), keyword()) :: %{
          integer() => float()
        }
  def explain_shap(instance, background_data, predict_fn, opts \\ []) do
    SHAP.explain(instance, background_data, predict_fn, opts)
  end

  @doc """
  Calculate feature importance using permutation importance.

  Convenience function that delegates to `CrucibleXAI.FeatureAttribution.permutation_importance/3`.

  ## Parameters
    * `predict_fn` - Prediction function
    * `validation_data` - List of {instance, label} tuples
    * `opts` - Options (see `CrucibleXAI.FeatureAttribution` for details)

  ## Returns
    Map of feature_index => %{importance: float, std_dev: float}

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> data = [{[1.0], 2.0}, {[2.0], 4.0}]
      iex> imp = CrucibleXai.feature_importance(predict_fn, data, num_repeats: 2)
      iex> is_map(imp)
      true
  """
  @spec feature_importance(
          (any() -> any()),
          [{list(), number()}, ...],
          Keyword.t()
        ) :: %{
          integer() => %{importance: float(), std_dev: float()}
        }
  def feature_importance(predict_fn, validation_data, opts \\ []) do
    FeatureAttribution.permutation_importance(predict_fn, validation_data, opts)
  end

  @doc """
  Validate an explanation comprehensively.

  Measures explanation quality across multiple dimensions: faithfulness,
  infidelity, sensitivity, and axiom compliance.

  New in v0.3.0.

  ## Parameters
    * `explanation` - Explanation struct to validate
    * `instance` - Instance that was explained
    * `predict_fn` - Prediction function
    * `opts` - Options (see `CrucibleXAI.Validation` for details)

  ## Returns
    Map with validation results and quality score

  ## Examples
      iex> explanation = CrucibleXai.explain([5.0, 10.0], fn [x, y] -> 2.0 * x + 3.0 * y end)
      iex> validation = CrucibleXai.validate_explanation(explanation, [5.0, 10.0], fn [x, y] -> 2.0 * x + 3.0 * y end)
      iex> is_map(validation)
      true
      iex> Map.has_key?(validation, :quality_score)
      true
  """
  @spec validate_explanation(
          Explanation.t(),
          list(),
          (any() -> any()),
          keyword()
        ) :: Validation.comprehensive_report()
  def validate_explanation(explanation, instance, predict_fn, opts \\ []) do
    Validation.comprehensive_validation(explanation, instance, predict_fn, opts)
  end

  @doc """
  Quick validation for production use.

  Fast quality check using faithfulness and infidelity metrics only.

  New in v0.3.0.

  ## Parameters
    * `explanation` - Explanation struct to validate
    * `instance` - Instance that was explained
    * `predict_fn` - Prediction function
    * `opts` - Options

  ## Returns
    Map with quality scores and pass/fail status

  ## Examples
      iex> explanation = CrucibleXai.explain([5.0], fn [x] -> x * 2.0 end)
      iex> quick = CrucibleXai.quick_validate(explanation, [5.0], fn [x] -> x * 2.0 end)
      iex> is_boolean(quick.passes_quality_gate)
      true
  """
  @spec quick_validate(Explanation.t(), list(), (any() -> any()), keyword()) ::
          Validation.quick_report()
  def quick_validate(explanation, instance, predict_fn, opts \\ []) do
    Validation.quick_validation(explanation, instance, predict_fn, opts)
  end

  @doc """
  Measure faithfulness of an explanation.

  Tests whether removing important features causes proportional prediction changes.

  New in v0.3.0.

  ## Parameters
    * `instance` - Instance explained
    * `explanation` - Explanation struct
    * `predict_fn` - Prediction function
    * `opts` - Options

  ## Returns
    Map with faithfulness score and details
  """
  @spec measure_faithfulness(list(), Explanation.t(), (any() -> any()), keyword()) ::
          Validation.Faithfulness.faithfulness_result()
  defdelegate measure_faithfulness(instance, explanation, predict_fn, opts \\ []),
    to: Validation.Faithfulness,
    as: :measure_faithfulness

  @doc """
  Compute infidelity score.

  Measures explanation error via perturbation-based testing.

  New in v0.3.0.

  ## Parameters
    * `instance` - Instance explained
    * `attributions` - Attribution map
    * `predict_fn` - Prediction function
    * `opts` - Options

  ## Returns
    Map with infidelity score (lower is better)
  """
  @spec compute_infidelity(list(), map(), (any() -> any()), keyword()) ::
          Validation.Infidelity.result()
  defdelegate compute_infidelity(instance, attributions, predict_fn, opts \\ []),
    to: Validation.Infidelity,
    as: :compute
end

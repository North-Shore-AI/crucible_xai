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
  - `CrucibleXai.Explanation` - Explanation structure and utilities
  - `CrucibleXai.LIME.Sampling` - Data perturbation strategies
  - `CrucibleXai.LIME.Kernels` - Proximity weighting functions
  - `CrucibleXai.LIME.InterpretableModels` - Linear regression models
  - `CrucibleXai.LIME.FeatureSelection` - Feature selection methods

  ## References

  - Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
    "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.
  """

  alias CrucibleXAI.{LIME, SHAP, FeatureAttribution, Explanation}

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
  @spec explain(any(), function(), keyword()) :: Explanation.t()
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
  @spec explain_batch(list(), function(), keyword()) :: list(Explanation.t())
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
  @spec feature_importance(function(), list({list(), number()}), keyword()) :: %{
          integer() => %{importance: float(), std_dev: float()}
        }
  def feature_importance(predict_fn, validation_data, opts \\ []) do
    FeatureAttribution.permutation_importance(predict_fn, validation_data, opts)
  end
end

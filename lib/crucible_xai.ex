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

  alias CrucibleXAI.{LIME, Explanation}

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
end

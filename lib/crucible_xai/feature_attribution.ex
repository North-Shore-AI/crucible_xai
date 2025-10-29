defmodule CrucibleXAI.FeatureAttribution do
  @moduledoc """
  Feature attribution methods for quantifying feature importance.

  This module provides multiple methods for understanding which features
  contribute most to model predictions. Different methods have different
  trade-offs in terms of speed, interpretability, and theoretical guarantees.

  ## Available Methods

  - **Permutation Importance**: Model-agnostic, measures performance degradation
  - **Gradient-based**: For differentiable models (future)
  - **Occlusion-based**: Measures prediction change when features are removed (future)

  ## Method Comparison

  | Method | Speed | Model Requirements | Interpretability |
  |--------|-------|-------------------|------------------|
  | Permutation | Medium | Any | High |
  | Gradient×Input | Fast | Differentiable | Medium |
  | Integrated Gradients | Slow | Differentiable | High |
  | Occlusion | Slow | Any | High |

  ## Examples

      # Permutation importance
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      validation_data = [{[1.0, 1.0], 5.0}, {[2.0, 2.0], 10.0}]

      importance = CrucibleXAI.FeatureAttribution.permutation_importance(
        predict_fn,
        validation_data,
        metric: :mse
      )
  """

  alias CrucibleXAI.FeatureAttribution.Permutation

  @doc """
  Calculate permutation importance for features.

  Delegates to `CrucibleXAI.FeatureAttribution.Permutation.calculate/3`.

  ## Parameters
    * `predict_fn` - Prediction function
    * `validation_data` - List of {instance, label} tuples
    * `opts` - Options:
      * `:metric` - Performance metric (`:mse`, `:mae`, `:accuracy`)
      * `:num_repeats` - Number of permutations per feature

  ## Returns
    Map of feature_index => %{importance: float, std_dev: float}

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> data = [{[1.0], 2.0}, {[2.0], 4.0}]
      iex> imp = CrucibleXAI.FeatureAttribution.permutation_importance(predict_fn, data, num_repeats: 2)
      iex> is_map(imp)
      true
  """
  @spec permutation_importance(
          (any() -> any()),
          [{list(), number()}, ...],
          Keyword.t()
        ) :: %{
          integer() => %{importance: float(), std_dev: float()}
        }
  def permutation_importance(predict_fn, validation_data, opts \\ []) do
    Permutation.calculate(predict_fn, validation_data, opts)
  end

  @doc """
  Get top k features by importance from any attribution method.

  ## Parameters
    * `importance_map` - Map of feature_index => stats
    * `k` - Number of top features

  ## Returns
    List of {feature_index, stats} sorted by importance

  ## Examples
      iex> importance = %{0 => %{importance: 0.3}, 1 => %{importance: 0.8}}
      iex> top = CrucibleXAI.FeatureAttribution.top_k(importance, 1)
      iex> {idx, _} = hd(top)
      iex> idx
      1
  """
  @spec top_k(%{integer() => map()}, pos_integer()) :: list({integer(), map()})
  def top_k(importance_map, k) do
    Permutation.top_k(importance_map, k)
  end
end

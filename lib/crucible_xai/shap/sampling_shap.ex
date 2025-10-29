defmodule CrucibleXAI.SHAP.SamplingShap do
  @moduledoc """
  SamplingShap: Monte Carlo approximation of SHAP values.

  SamplingShap approximates SHAP values by randomly sampling feature permutations
  and averaging marginal contributions. This is faster than KernelSHAP's weighted
  regression approach while still being model-agnostic.

  ## Algorithm

  For each feature i:
  1. Generate random permutations of all features
  2. For each permutation π:
     - Find features S that appear before i in π
     - Calculate marginal contribution: f(S ∪ {i}) - f(S)
  3. Average marginal contributions across all permutations

  ## Complexity

  - Time: O(n_samples * n_features * prediction_time)
  - Faster than KernelSHAP for similar accuracy levels
  - Accuracy improves with more samples

  ## References

  Lundberg, S. M., & Lee, S. I. (2017).
  A Unified Approach to Interpreting Model Predictions. NeurIPS.
  (Monte Carlo sampling approach)

  ## Examples

      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      shap_values = SamplingShap.explain(instance, background, predict_fn, num_samples: 1000)
      # => %{0 => ~2.0, 1 => ~3.0}
  """

  @doc """
  Compute approximate SHAP values using Monte Carlo sampling.

  ## Parameters
    * `instance` - Instance to explain (list of feature values)
    * `background_data` - Background dataset for computing feature baseline
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:num_samples` - Number of permutation samples (default: 1000)

  ## Returns
    Map of feature_index => approximate_shapley_value

  ## Examples

      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instance = [5.0]
      iex> background = [[0.0]]
      iex> shap = CrucibleXAI.SHAP.SamplingShap.explain(instance, background, predict_fn, num_samples: 100)
      iex> is_map(shap)
      true
      iex> map_size(shap)
      1
  """
  @spec explain(list(float()), list(list(float())), function(), keyword()) :: %{
          integer() => float()
        }
  def explain(instance, background_data, predict_fn, opts \\ []) do
    num_samples = Keyword.get(opts, :num_samples, 1000)
    n_features = length(instance)

    # Calculate background mean for baseline
    background_mean = calculate_background_mean(background_data)

    # For each feature, accumulate marginal contributions across permutations
    contributions =
      for _ <- 1..num_samples do
        # Generate random permutation
        permutation = generate_permutation(n_features)

        # Calculate marginal contribution for each feature in this permutation
        permutation
        |> Enum.with_index()
        |> Map.new(fn {feature_idx, position} ->
          # Features that appear before this one in the permutation
          preceding_features = Enum.slice(permutation, 0, position)

          # Marginal contribution of adding this feature
          contrib =
            marginal_contribution(
              feature_idx,
              preceding_features,
              instance,
              background_mean,
              predict_fn
            )

          {feature_idx, contrib}
        end)
      end

    # Average contributions across all samples
    Enum.reduce(0..(n_features - 1), %{}, fn feature_idx, acc ->
      total_contrib =
        contributions
        |> Enum.map(fn contrib_map -> Map.get(contrib_map, feature_idx, 0.0) end)
        |> Enum.sum()

      avg_contrib = total_contrib / num_samples

      Map.put(acc, feature_idx, avg_contrib)
    end)
  end

  @doc """
  Generate a random permutation of feature indices.

  ## Parameters
    * `n_features` - Number of features

  ## Returns
    List of feature indices in random order

  ## Examples

      iex> perm = CrucibleXAI.SHAP.SamplingShap.generate_permutation(3)
      iex> Enum.sort(perm)
      [0, 1, 2]
  """
  @spec generate_permutation(integer()) :: list(integer())
  def generate_permutation(n_features) do
    0..(n_features - 1)
    |> Enum.to_list()
    |> Enum.shuffle()
  end

  @doc """
  Calculate marginal contribution of adding a feature.

  Computes: f(S ∪ {feature}) - f(S)

  ## Parameters
    * `feature_idx` - Index of feature to add
    * `present_features` - Features already in the set S
    * `instance` - Instance being explained
    * `background_mean` - Mean values from background data
    * `predict_fn` - Prediction function

  ## Returns
    Float representing the marginal contribution
  """
  @spec marginal_contribution(
          integer(),
          list(integer()),
          list(float()),
          list(float()),
          function()
        ) ::
          float()
  def marginal_contribution(feature_idx, present_features, instance, background_mean, predict_fn) do
    # Build instance with feature included
    instance_with = build_instance(instance, background_mean, present_features ++ [feature_idx])

    # Build instance without feature
    instance_without = build_instance(instance, background_mean, present_features)

    # Marginal contribution
    pred_with = get_prediction(predict_fn, instance_with)
    pred_without = get_prediction(predict_fn, instance_without)

    pred_with - pred_without
  end

  # Private helpers

  defp calculate_background_mean(background_data) do
    n_features = length(hd(background_data))
    n_instances = length(background_data)

    for feature_idx <- 0..(n_features - 1) do
      feature_sum =
        background_data
        |> Enum.map(fn instance -> Enum.at(instance, feature_idx) end)
        |> Enum.sum()

      feature_sum / n_instances
    end
  end

  defp build_instance(instance, background_mean, present_feature_indices) do
    instance
    |> Enum.with_index()
    |> Enum.map(fn {value, idx} ->
      if idx in present_feature_indices do
        # Use actual value from instance
        value
      else
        # Use background mean
        Enum.at(background_mean, idx)
      end
    end)
  end

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end
end

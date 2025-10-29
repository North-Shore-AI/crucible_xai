defmodule CrucibleXAI.Global.ALE do
  @moduledoc """
  Accumulated Local Effects (ALE) for global interpretation.

  ALE is more robust than PDP when features are correlated. Instead of
  marginalizing over the entire dataset (like PDP), ALE computes local
  effects by looking at prediction changes within small neighborhoods.

  ## Why ALE > PDP for Correlated Features

  - **PDP problem**: With correlated features, PDP creates unrealistic instances
    (e.g., if age and experience are correlated, PDP might evaluate age=20 with
    experience=30 years, which never occurs in real data)

  - **ALE solution**: Only looks at local changes within bins of similar instances,
    avoiding extrapolation to unrealistic combinations

  ## Algorithm

  1. Divide feature range into bins (quantile-based for equal representation)
  2. For each bin:
     - Find instances in that bin
     - Compute effect = avg[f(upper_edge, x_other) - f(lower_edge, x_other)]
  3. Accumulate effects across bins
  4. Center by subtracting mean (for interpretability)

  ## References

  - Apley, D. W., & Zhu, J. (2020). "Visualizing the Effects of Predictor Variables
    in Black Box Supervised Learning Models"
  - Molnar, C. (2022). "Interpretable Machine Learning" (Chapter on ALE)

  ## Examples

      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

      ale = CrucibleXAI.Global.ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 5)
      # => %{bin_centers: [...], effects: [...], feature_index: 0}
  """

  @doc """
  Compute accumulated local effects for a feature.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset (list of instances)
    * `feature_index` - Index of feature to analyze
    * `opts` - Options:
      * `:num_bins` - Number of bins for discretization (default: 10)

  ## Returns
    Map with:
    * `:bin_centers` - Center value of each bin
    * `:effects` - Accumulated local effects for each bin
    * `:feature_index` - The feature that was analyzed

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
      iex> ale = CrucibleXAI.Global.ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 2)
      iex> is_map(ale)
      true
      iex> Map.has_key?(ale, :effects)
      true
  """
  @spec accumulated_local_effects(function(), list(list(float())), integer(), keyword()) :: map()
  def accumulated_local_effects(predict_fn, data, feature_index, opts \\ []) do
    num_bins = Keyword.get(opts, :num_bins, 10)

    # Get feature values
    feature_values = Enum.map(data, fn instance -> Enum.at(instance, feature_index) end)

    # Compute bin edges using quantiles
    bin_edges = compute_bin_edges(data, feature_index, num_bins)

    # For each bin, compute local effect
    local_effects =
      Enum.chunk_every(bin_edges, 2, 1, :discard)
      |> Enum.map(fn [bin_min, bin_max] ->
        # Find instances in this bin
        instances_in_bin =
          Enum.zip(data, feature_values)
          |> Enum.filter(fn {_instance, feat_val} ->
            feat_val >= bin_min and feat_val < bin_max
          end)
          |> Enum.map(fn {instance, _feat_val} -> instance end)

        # Compute average local effect for this bin
        local_effect_in_bin(predict_fn, instances_in_bin, feature_index, bin_min, bin_max)
      end)

    # Accumulate effects
    accumulated_effects =
      local_effects
      |> Enum.scan(0.0, fn effect, acc -> acc + effect end)

    # Center effects (subtract mean)
    mean_effect =
      if length(accumulated_effects) > 0 do
        Enum.sum(accumulated_effects) / length(accumulated_effects)
      else
        0.0
      end

    centered_effects = Enum.map(accumulated_effects, fn eff -> eff - mean_effect end)

    # Compute bin centers for plotting
    bin_centers =
      Enum.chunk_every(bin_edges, 2, 1, :discard)
      |> Enum.map(fn [bin_min, bin_max] -> (bin_min + bin_max) / 2.0 end)

    %{
      bin_centers: bin_centers,
      effects: centered_effects,
      feature_index: feature_index
    }
  end

  @doc """
  Compute bin edges using quantiles for equal representation.

  ## Parameters
    * `data` - Dataset
    * `feature_index` - Feature to bin
    * `num_bins` - Number of bins

  ## Returns
    List of bin edges (length = num_bins + 1)

  ## Examples

      iex> data = [[1.0, 5.0], [2.0, 6.0], [3.0, 7.0]]
      iex> edges = CrucibleXAI.Global.ALE.compute_bin_edges(data, 0, 2)
      iex> length(edges)
      3
  """
  @spec compute_bin_edges(list(list(float())), integer(), integer()) :: list(float())
  def compute_bin_edges(data, feature_index, num_bins) do
    # Extract and sort feature values
    feature_values =
      data
      |> Enum.map(fn instance -> Enum.at(instance, feature_index) end)
      |> Enum.sort()

    # Compute quantile-based edges
    n = length(feature_values)

    if num_bins == 1 do
      # Single bin: just min and max
      [Enum.min(feature_values), Enum.max(feature_values)]
    else
      # Quantile-based bins for equal representation
      for i <- 0..num_bins do
        quantile_index = floor(i * (n - 1) / num_bins)
        Enum.at(feature_values, quantile_index)
      end
      |> Enum.uniq()
    end
  end

  @doc """
  Compute local effect within a bin.

  For instances in the bin, average the difference:
  f(upper_edge, other_features) - f(lower_edge, other_features)

  ## Parameters
    * `predict_fn` - Prediction function
    * `instances_in_bin` - Instances that fall in this bin
    * `feature_index` - Feature being analyzed
    * `bin_min` - Lower edge of bin
    * `bin_max` - Upper edge of bin

  ## Returns
    Float representing average local effect in this bin
  """
  @spec local_effect_in_bin(function(), list(list(float())), integer(), float(), float()) ::
          float()
  def local_effect_in_bin(_predict_fn, [], _feature_index, _bin_min, _bin_max) do
    # Empty bin returns 0
    0.0
  end

  def local_effect_in_bin(predict_fn, instances_in_bin, feature_index, bin_min, bin_max) do
    # For each instance, compute difference when moving from bin_min to bin_max
    effects =
      Enum.map(instances_in_bin, fn instance ->
        # Prediction with feature at bin_min
        instance_at_min = List.replace_at(instance, feature_index, bin_min)
        pred_min = get_prediction(predict_fn, instance_at_min)

        # Prediction with feature at bin_max
        instance_at_max = List.replace_at(instance, feature_index, bin_max)
        pred_max = get_prediction(predict_fn, instance_at_max)

        # Local effect
        pred_max - pred_min
      end)

    # Average across instances in the bin
    if length(effects) > 0 do
      Enum.sum(effects) / length(effects)
    else
      0.0
    end
  end

  # Private helpers

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end
end

defmodule CrucibleXAI.OcclusionAttribution do
  @moduledoc """
  Occlusion-based attribution methods.

  These methods work by removing (occluding) features and measuring
  the change in model prediction. Unlike gradient methods, occlusion
  works with any model (differentiable or not).

  ## Methods

  - **Feature Occlusion**: Occlude each feature individually
  - **Sliding Window**: Occlude windows of consecutive features
  - **Occlusion Sensitivity**: Normalized sensitivity scores

  ## How It Works

  1. Get original prediction for the instance
  2. For each feature (or window), replace with baseline value
  3. Get prediction with feature occluded
  4. Attribution = original_prediction - occluded_prediction

  ## Advantages

  - Model-agnostic (no gradients needed)
  - Works with black-box models
  - Intuitive interpretation
  - Good for feature importance ranking

  ## References

  - Zeiler & Fergus (2014). "Visualizing and Understanding Convolutional Networks"
  - Zhou et al. (2016). "Learning Deep Features for Discriminative Localization"

  ## Examples

      # Simple feature occlusion
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 4.0]

      attributions = CrucibleXAI.OcclusionAttribution.feature_occlusion(instance, predict_fn)
      # => %{0 => 10.0, 1 => 12.0}
  """

  require Logger

  @doc """
  Compute attribution by occluding each feature individually.

  ## Parameters
    * `instance` - Instance to explain (list or Nx tensor)
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:baseline_value` - Value to use for occluded features (default: 0.0)

  ## Returns
    Map of feature_index => attribution_score

  ## Examples

      iex> predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      iex> instance = [5.0, 4.0]
      iex> attrs = CrucibleXAI.OcclusionAttribution.feature_occlusion(instance, predict_fn)
      iex> is_map(attrs)
      true
      iex> map_size(attrs)
      2
  """
  @spec feature_occlusion(list() | Nx.Tensor.t(), function(), keyword()) :: %{
          integer() => float()
        }
  def feature_occlusion(instance, predict_fn, opts \\ []) do
    baseline_value = Keyword.get(opts, :baseline_value, 0.0)

    # Convert to list if Nx tensor
    instance_list = to_list(instance)
    n_features = length(instance_list)

    # Get original prediction
    original_pred = get_prediction(predict_fn, instance)

    # For each feature, occlude it and measure change
    for feature_idx <- 0..(n_features - 1), into: %{} do
      # Create occluded instance
      occluded = List.replace_at(instance_list, feature_idx, baseline_value)

      # Get prediction with feature occluded
      occluded_pred = get_prediction(predict_fn, occluded)

      # Attribution is difference in prediction
      attribution = original_pred - occluded_pred

      {feature_idx, attribution}
    end
  end

  @doc """
  Compute attributions using sliding window occlusion.

  Useful for sequential data (time series, text) where groups of
  consecutive features should be analyzed together.

  ## Parameters
    * `instance` - Instance to explain (list or Nx tensor)
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:window_size` - Size of occlusion window (default: 2)
      * `:stride` - Step size for sliding window (default: 1)
      * `:baseline_value` - Value for occluded features (default: 0.0)

  ## Returns
    Map of window_position => attribution_score

  ## Examples

      iex> predict_fn = fn inst -> Enum.sum(inst) end
      iex> instance = [1.0, 2.0, 3.0, 4.0]
      iex> attrs = CrucibleXAI.OcclusionAttribution.sliding_window_occlusion(instance, predict_fn, window_size: 2)
      iex> is_map(attrs)
      true
  """
  @spec sliding_window_occlusion(list() | Nx.Tensor.t(), function(), keyword()) :: %{
          integer() => float()
        }
  def sliding_window_occlusion(instance, predict_fn, opts \\ []) do
    window_size = Keyword.get(opts, :window_size, 2)
    stride = Keyword.get(opts, :stride, 1)
    baseline_value = Keyword.get(opts, :baseline_value, 0.0)

    instance_list = to_list(instance)
    n_features = length(instance_list)

    # Get original prediction
    original_pred = get_prediction(predict_fn, instance)

    # Calculate number of windows
    max_start = max(0, n_features - window_size)
    window_positions = 0..max_start//stride

    # For each window position, occlude the window
    for position <- window_positions, into: %{} do
      # Create instance with window occluded
      occluded =
        instance_list
        |> Enum.with_index()
        |> Enum.map(fn {value, idx} ->
          if idx >= position and idx < position + window_size do
            baseline_value
          else
            value
          end
        end)

      # Get prediction with window occluded
      occluded_pred = get_prediction(predict_fn, occluded)

      # Attribution for this window
      attribution = original_pred - occluded_pred

      {position, attribution}
    end
  end

  @doc """
  Compute occlusion sensitivity scores.

  Similar to feature_occlusion but with optional normalization
  and absolute value options for better visualization.

  ## Parameters
    * `instance` - Instance to explain
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:baseline_value` - Value for occluded features (default: 0.0)
      * `:normalize` - Normalize to sum to 1.0 (default: false)
      * `:absolute` - Use absolute values (default: false)

  ## Returns
    Map of feature_index => sensitivity_score

  ## Examples

      iex> predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      iex> instance = [5.0, 4.0]
      iex> sens = CrucibleXAI.OcclusionAttribution.occlusion_sensitivity(instance, predict_fn)
      iex> is_map(sens)
      true
  """
  @spec occlusion_sensitivity(list() | Nx.Tensor.t(), function(), keyword()) :: %{
          integer() => float()
        }
  def occlusion_sensitivity(instance, predict_fn, opts \\ []) do
    normalize = Keyword.get(opts, :normalize, false)
    absolute = Keyword.get(opts, :absolute, false)

    # Get base attributions
    attributions = feature_occlusion(instance, predict_fn, opts)

    # Apply absolute value if requested
    attributions =
      if absolute do
        Map.new(attributions, fn {k, v} -> {k, abs(v)} end)
      else
        attributions
      end

    # Normalize if requested
    if normalize do
      normalize_attributions(attributions)
    else
      attributions
    end
  end

  defp normalize_attributions(attributions) do
    total = Enum.sum(Map.values(attributions))

    if total == 0.0 do
      attributions
    else
      Map.new(attributions, fn {k, v} -> {k, v / total} end)
    end
  end

  @doc """
  Compute occlusion attribution for multiple instances.

  ## Parameters
    * `instances` - List of instances to explain
    * `predict_fn` - Prediction function
    * `opts` - Options (same as feature_occlusion/3 plus):
      * `:parallel` - Enable parallel processing (default: false)
      * `:max_concurrency` - Max concurrent tasks

  ## Returns
    List of attribution maps

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> instances = [[1.0, 2.0], [3.0, 4.0]]
      iex> batch = CrucibleXAI.OcclusionAttribution.batch_occlusion(instances, predict_fn)
      iex> length(batch)
      2
  """
  @spec batch_occlusion(list(), function(), keyword()) :: list(%{integer() => float()})
  def batch_occlusion(instances, predict_fn, opts \\ []) do
    parallel = Keyword.get(opts, :parallel, false)

    if parallel do
      batch_occlusion_parallel(instances, predict_fn, opts)
    else
      batch_occlusion_sequential(instances, predict_fn, opts)
    end
  end

  # Private helpers

  defp batch_occlusion_sequential(instances, predict_fn, opts) do
    Enum.map(instances, fn instance ->
      feature_occlusion(instance, predict_fn, opts)
    end)
  end

  defp batch_occlusion_parallel(instances, predict_fn, opts) do
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())

    instances
    |> Task.async_stream(
      fn instance ->
        feature_occlusion(instance, predict_fn, opts)
      end,
      max_concurrency: max_concurrency,
      timeout: 30_000
    )
    |> Enum.map(fn {:ok, result} -> result end)
  end

  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)
  defp to_list(list) when is_list(list), do: list

  defp get_prediction(predict_fn, instance) when is_list(instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end

  defp get_prediction(predict_fn, %Nx.Tensor{} = instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end
end

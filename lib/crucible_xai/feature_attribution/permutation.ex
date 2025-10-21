defmodule CrucibleXAI.FeatureAttribution.Permutation do
  @moduledoc """
  Permutation importance for feature attribution.

  Measures the increase in model error when a feature's values are randomly
  shuffled. If a feature is important, shuffling it will significantly
  degrade model performance. If a feature is irrelevant, shuffling it
  will have little effect.

  ## Algorithm

  1. Measure baseline performance on validation set
  2. For each feature:
     a. Shuffle that feature's values across the dataset
     b. Measure performance on shuffled data
     c. Importance = baseline_score - shuffled_score
  3. Repeat multiple times and compute mean and standard deviation

  ## Advantages

  - Model-agnostic
  - Accounts for feature interactions
  - Easy to interpret (change in performance metric)

  ## Disadvantages

  - Requires validation dataset
  - Can be slow for many features
  - May create unrealistic feature combinations when features are correlated

  ## Examples

      iex> predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      iex> validation_data = [{[1.0, 1.0], 5.0}, {[2.0, 2.0], 10.0}]
      iex> importance = CrucibleXAI.FeatureAttribution.Permutation.calculate(predict_fn, validation_data, num_repeats: 2)
      iex> is_map(importance)
      true
  """

  @default_opts [
    metric: :mse,
    num_repeats: 10
  ]

  @doc """
  Calculate permutation importance for each feature.

  ## Parameters
    * `predict_fn` - Prediction function
    * `validation_data` - List of {features, label} tuples
    * `opts` - Options:
      * `:metric` - Performance metric (`:mse`, `:mae`, `:accuracy`) (default: `:mse`)
      * `:num_repeats` - Number of permutations per feature (default: 10)

  ## Returns
    Map of feature_index => %{importance: float, std_dev: float}

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> data = [{[1.0], 2.0}, {[2.0], 4.0}, {[3.0], 6.0}]
      iex> result = CrucibleXAI.FeatureAttribution.Permutation.calculate(predict_fn, data, num_repeats: 2)
      iex> Map.has_key?(result, 0)
      true
      iex> is_float(result[0][:importance])
      true
  """
  @spec calculate(function(), list({list(), number()}), keyword()) :: %{
          integer() => %{importance: float(), std_dev: float()}
        }
  def calculate(predict_fn, validation_data, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    metric = Keyword.get(opts, :metric)
    num_repeats = Keyword.get(opts, :num_repeats)

    # Calculate baseline performance
    baseline_score = calculate_score(predict_fn, validation_data, metric)

    # Get number of features from first instance
    {first_instance, _} = hd(validation_data)
    n_features = length(first_instance)

    # Calculate importance for each feature
    for feature_idx <- 0..(n_features - 1), into: %{} do
      importances =
        for _ <- 1..num_repeats do
          # Shuffle this feature
          shuffled_data = shuffle_feature(validation_data, feature_idx)

          # Measure performance on shuffled data
          shuffled_score = calculate_score(predict_fn, shuffled_data, metric)

          # For error metrics (MSE, MAE): importance = shuffled - baseline (increase in error)
          # For accuracy: importance = baseline - shuffled (decrease in accuracy)
          case metric do
            :accuracy -> baseline_score - shuffled_score
            _ -> shuffled_score - baseline_score
          end
        end

      # Calculate mean and std dev
      mean_importance = Enum.sum(importances) / length(importances)
      std_dev = calculate_std_dev(importances, mean_importance)

      {feature_idx, %{importance: mean_importance, std_dev: std_dev}}
    end
  end

  @doc """
  Get top k features by importance.

  ## Parameters
    * `importance_map` - Map from `calculate/3`
    * `k` - Number of top features to return

  ## Returns
    List of {feature_index, stats} tuples sorted by importance

  ## Examples
      iex> importance = %{0 => %{importance: 0.5, std_dev: 0.1}, 1 => %{importance: 0.8, std_dev: 0.2}}
      iex> top = CrucibleXAI.FeatureAttribution.Permutation.top_k(importance, 1)
      iex> length(top)
      1
      iex> {idx, _} = hd(top)
      iex> idx
      1
  """
  @spec top_k(%{integer() => map()}, pos_integer()) :: list({integer(), map()})
  def top_k(importance_map, k) do
    importance_map
    |> Enum.sort_by(fn {_idx, stats} -> stats[:importance] end, :desc)
    |> Enum.take(k)
  end

  # Private helper functions

  defp calculate_score(predict_fn, validation_data, metric) do
    predictions =
      Enum.map(validation_data, fn {instance, _label} ->
        result = predict_fn.(instance)

        case result do
          %Nx.Tensor{} -> Nx.to_number(result)
          num when is_number(num) -> num
        end
      end)

    labels = Enum.map(validation_data, fn {_instance, label} -> label end)

    case metric do
      :mse -> calculate_mse(predictions, labels)
      :mae -> calculate_mae(predictions, labels)
      :accuracy -> calculate_accuracy(predictions, labels)
      _ -> raise ArgumentError, "Unknown metric: #{inspect(metric)}"
    end
  end

  defp shuffle_feature(validation_data, feature_idx) do
    # Extract all values for this feature
    feature_values =
      Enum.map(validation_data, fn {instance, _label} ->
        Enum.at(instance, feature_idx)
      end)

    # Shuffle them
    shuffled_values = Enum.shuffle(feature_values)

    # Reconstruct data with shuffled feature
    validation_data
    |> Enum.zip(shuffled_values)
    |> Enum.map(fn {{instance, label}, shuffled_val} ->
      shuffled_instance = List.replace_at(instance, feature_idx, shuffled_val)
      {shuffled_instance, label}
    end)
  end

  defp calculate_mse(predictions, labels) do
    Enum.zip(predictions, labels)
    |> Enum.map(fn {pred, label} -> :math.pow(pred - label, 2) end)
    |> Enum.sum()
    |> Kernel./(length(predictions))
  end

  defp calculate_mae(predictions, labels) do
    Enum.zip(predictions, labels)
    |> Enum.map(fn {pred, label} -> abs(pred - label) end)
    |> Enum.sum()
    |> Kernel./(length(predictions))
  end

  defp calculate_accuracy(predictions, labels) do
    # For classification: count correct predictions
    # Round predictions to nearest integer for classification
    correct =
      Enum.zip(predictions, labels)
      |> Enum.count(fn {pred, label} ->
        round(pred) == round(label)
      end)

    correct / length(predictions)
  end

  defp calculate_std_dev(values, mean) do
    if length(values) < 2 do
      0.0
    else
      variance =
        values
        |> Enum.map(fn x -> :math.pow(x - mean, 2) end)
        |> Enum.sum()
        |> Kernel./(length(values) - 1)

      :math.sqrt(variance)
    end
  end
end

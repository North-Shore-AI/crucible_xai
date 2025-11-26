defmodule CrucibleXAI.Validation.Faithfulness do
  @moduledoc ~S"""
  Faithfulness metrics for explanation validation.

  Measures how well explanations reflect actual model behavior by testing
  whether removing important features causes proportional prediction changes.

  ## Overview

  Faithfulness validation uses the **feature removal test**: if an explanation
  claims feature X is important, removing X should significantly change the
  prediction. The correlation between attribution magnitude and prediction
  change quantifies faithfulness.

  ## Metrics

  - **Faithfulness Score**: Spearman/Pearson correlation between feature
    importance ranking and prediction change magnitude (range: -1 to 1,
    higher is better)
  - **Monotonicity**: Whether prediction changes increase monotonically as
    more features are removed (boolean)
  - **Violation Severity**: Average magnitude of monotonicity violations

  ## Usage

      # Validate LIME explanation
      explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

      result = Faithfulness.measure_faithfulness(
        instance,
        explanation,
        predict_fn
      )

      IO.puts("Faithfulness: #{result.faithfulness_score}")
      # => 0.87 (Good)

  ## References

  Based on:
  - Hooker et al. (2019) "A Benchmark for Interpretability Methods"
  - Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations"
  """

  alias CrucibleXAI.Explanation

  @type faithfulness_result :: %{
          faithfulness_score: float(),
          prediction_drops: [number()],
          feature_order: [integer()],
          monotonicity: boolean(),
          interpretation: String.t()
        }

  @type monotonicity_result :: %{
          is_monotonic: boolean(),
          violations: non_neg_integer(),
          violation_indices: [integer()],
          severity: float()
        }

  @type full_report :: %{
          faithfulness_score: float(),
          prediction_drops: [number()],
          feature_order: [integer()],
          monotonicity: boolean(),
          interpretation: String.t(),
          monotonicity_details: monotonicity_result(),
          summary: String.t()
        }

  @doc """
  Measure faithfulness via feature removal.

  ## Algorithm

  1. Sort features by absolute attribution (descending)
  2. Remove features incrementally (most important first)
  3. Measure prediction change at each step
  4. Compute correlation between attribution rank and prediction change

  ## Parameters

    * `instance` - Instance to test (list of feature values)
    * `explanation` - Explanation struct to validate
    * `predict_fn` - Model prediction function
    * `opts` - Options:
      * `:baseline_value` - Value for removed features (default: 0.0)
      * `:num_steps` - Number of removal steps (default: all features)
      * `:correlation_method` - `:pearson` or `:spearman` (default: `:spearman`)

  ## Returns

  Map with:
    * `:faithfulness_score` - Correlation coefficient (-1 to 1, higher better)
    * `:prediction_drops` - Prediction change at each removal step
    * `:feature_order` - Order features were removed (by importance)
    * `:monotonicity` - Whether drops are monotonic (boolean)
    * `:interpretation` - Human-readable assessment

  ## Examples

      # Perfect faithfulness for linear model
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 10.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0, 1 => 3.0},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)
      # => %{faithfulness_score: 1.0, monotonicity: true, ...}
  """
  @spec measure_faithfulness(list(), Explanation.t(), (any() -> any()), keyword()) ::
          faithfulness_result()
  def measure_faithfulness(instance, %Explanation{} = explanation, predict_fn, opts \\ []) do
    baseline_value = Keyword.get(opts, :baseline_value, 0.0)
    num_steps = Keyword.get(opts, :num_steps, map_size(explanation.feature_weights))
    correlation_method = Keyword.get(opts, :correlation_method, :spearman)

    # Get original prediction
    original_pred = get_prediction(predict_fn, instance)

    # Sort features by absolute attribution (descending)
    sorted_features =
      explanation.feature_weights
      |> Enum.sort_by(fn {_idx, weight} -> abs(weight) end, :desc)
      |> Enum.take(num_steps)

    # Remove features incrementally and measure prediction changes
    {prediction_drops, feature_order} =
      Enum.reduce(sorted_features, {[], []}, fn {feature_idx, _weight}, {drops, order} ->
        # Create instance with this feature and all previous ones removed
        modified_instance =
          instance
          |> Enum.with_index()
          |> Enum.map(fn {val, idx} ->
            if idx in order or idx == feature_idx do
              baseline_value
            else
              val
            end
          end)

        # Get prediction on modified instance
        modified_pred = get_prediction(predict_fn, modified_instance)
        drop = abs(original_pred - modified_pred)

        {drops ++ [drop], order ++ [feature_idx]}
      end)

    # Compute correlation between rank and prediction drops
    ranks = Enum.with_index(sorted_features, 1) |> Enum.map(fn {_, idx} -> idx end)
    faithfulness_score = compute_correlation(ranks, prediction_drops, correlation_method)

    # Check monotonicity
    monotonicity = check_monotonicity(prediction_drops)

    # Interpretation
    interpretation = interpret_faithfulness(faithfulness_score, monotonicity)

    %{
      faithfulness_score: faithfulness_score,
      prediction_drops: prediction_drops,
      feature_order: feature_order,
      monotonicity: monotonicity,
      interpretation: interpretation
    }
  end

  @doc """
  Test monotonicity property.

  Removing features in order of importance should cause monotonically
  increasing prediction changes (for regression) or decreasing confidence
  (for classification).

  ## Parameters

    * `instance` - Instance to test
    * `explanation` - Explanation struct
    * `predict_fn` - Model prediction function
    * `opts` - Same options as `measure_faithfulness/4`

  ## Returns

  Map with:
    * `:is_monotonic` - Whether drops are monotonic (boolean)
    * `:violations` - Number of monotonicity violations
    * `:violation_indices` - Step indices where violations occurred
    * `:severity` - Average violation magnitude

  ## Examples

      result = Faithfulness.monotonicity_test(instance, explanation, predict_fn)
      # => %{is_monotonic: true, violations: 0, ...}
  """
  @spec monotonicity_test(list(), Explanation.t(), (any() -> any()), keyword()) ::
          monotonicity_result()
  def monotonicity_test(instance, explanation, predict_fn, opts \\ []) do
    baseline_value = Keyword.get(opts, :baseline_value, 0.0)

    # Get prediction drops
    result =
      measure_faithfulness(
        instance,
        explanation,
        predict_fn,
        Keyword.put(opts, :baseline_value, baseline_value)
      )

    drops = result.prediction_drops

    # Check for violations
    {violations, violation_indices, severities} =
      drops
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.with_index()
      |> Enum.reduce({0, [], []}, fn {[prev, curr], idx}, {count, indices, sevs} ->
        if curr < prev do
          severity = prev - curr
          {count + 1, indices ++ [idx], sevs ++ [severity]}
        else
          {count, indices, sevs}
        end
      end)

    is_monotonic = violations == 0

    avg_severity =
      if length(severities) > 0, do: Enum.sum(severities) / length(severities), else: 0.0

    %{
      is_monotonic: is_monotonic,
      violations: violations,
      violation_indices: violation_indices,
      severity: avg_severity
    }
  end

  @doc """
  Generate comprehensive faithfulness report.

  Combines feature removal test and monotonicity analysis into a single
  detailed report with human-readable summary.

  ## Parameters

    * `instance` - Instance to test
    * `explanation` - Explanation struct
    * `predict_fn` - Model prediction function
    * `opts` - Same options as `measure_faithfulness/4`

  ## Returns

  Map combining faithfulness metrics, monotonicity details, and summary text.

  ## Examples

      report = Faithfulness.full_report(instance, explanation, predict_fn)
      IO.puts(report.summary)
  """
  @spec full_report(list(), Explanation.t(), (any() -> any()), keyword()) :: full_report()
  def full_report(instance, explanation, predict_fn, opts \\ []) do
    faithfulness = measure_faithfulness(instance, explanation, predict_fn, opts)
    monotonicity = monotonicity_test(instance, explanation, predict_fn, opts)

    Map.merge(faithfulness, %{
      monotonicity_details: monotonicity,
      summary: generate_summary(faithfulness, monotonicity)
    })
  end

  # Private helper functions

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num * 1.0
      _ -> raise ArgumentError, "Prediction function must return number or Nx.Tensor"
    end
  end

  defp compute_correlation(x, y, :spearman) do
    # Spearman rank correlation
    x_ranks = rank_data(x)
    y_ranks = rank_data(y)
    pearson_correlation(x_ranks, y_ranks)
  end

  defp compute_correlation(x, y, :pearson) do
    pearson_correlation(x, y)
  end

  defp pearson_correlation(x, y) when length(x) != length(y), do: 0.0
  defp pearson_correlation(x, _y) when length(x) < 2, do: 1.0

  defp pearson_correlation(x, y) do
    n = length(x)
    mean_x = Enum.sum(x) / n
    mean_y = Enum.sum(y) / n

    numerator =
      Enum.zip(x, y)
      |> Enum.map(fn {xi, yi} -> (xi - mean_x) * (yi - mean_y) end)
      |> Enum.sum()

    sum_sq_x = Enum.map(x, fn xi -> :math.pow(xi - mean_x, 2) end) |> Enum.sum()
    sum_sq_y = Enum.map(y, fn yi -> :math.pow(yi - mean_y, 2) end) |> Enum.sum()

    denominator = :math.sqrt(sum_sq_x * sum_sq_y)

    if denominator == 0.0 do
      0.0
    else
      numerator / denominator
    end
  end

  defp rank_data(data) do
    data
    |> Enum.with_index()
    |> Enum.sort_by(fn {val, _idx} -> val end)
    |> Enum.with_index()
    |> Enum.sort_by(fn {{_val, orig_idx}, _rank} -> orig_idx end)
    |> Enum.map(fn {{_val, _orig_idx}, rank} -> rank + 1 end)
  end

  defp check_monotonicity([]), do: true
  defp check_monotonicity([_]), do: true

  defp check_monotonicity(drops) do
    drops
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.all?(fn [prev, curr] -> curr >= prev end)
  end

  defp interpret_faithfulness(score, monotonic) do
    cond do
      score >= 0.9 and monotonic -> "Excellent - High faithfulness with perfect monotonicity"
      score >= 0.9 -> "Excellent - High faithfulness"
      score >= 0.7 -> "Good - Moderate faithfulness"
      score >= 0.5 -> "Fair - Weak faithfulness"
      score >= 0.3 -> "Poor - Low faithfulness"
      true -> "Very Poor - Unreliable explanation"
    end
  end

  defp generate_summary(faithfulness, monotonicity) do
    """
    Faithfulness Score: #{Float.round(faithfulness.faithfulness_score, 3)}
    Monotonicity: #{if monotonicity.is_monotonic, do: "Yes", else: "No (#{monotonicity.violations} violations)"}
    Assessment: #{faithfulness.interpretation}
    """
  end
end

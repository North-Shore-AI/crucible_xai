defmodule CrucibleXAI.Validation.Sensitivity do
  @moduledoc ~S"""
  Robustness and sensitivity testing for explanations.

  Measures explanation stability under input perturbations and hyperparameter
  variations. Stable explanations are more trustworthy and reliable.

  ## Metrics

  - **Stability Score**: 0-1 scale (1 = perfectly stable, 0 = very unstable)
  - **Mean Variation**: Average change in attributions across perturbations
  - **Max Variation**: Maximum attribution change observed
  - **Coefficient of Variation**: Normalized stability measure (σ/μ)

  ## Usage

      # Test input sensitivity
      explain_fn = fn inst -> CrucibleXai.explain(inst, predict_fn) end

      result = Sensitivity.input_sensitivity(
        instance,
        explain_fn,
        predict_fn,
        num_perturbations: 50
      )

      IO.puts("Stability: #{result.stability_score}")
      # => 0.92 (High stability)

  ## References

  Based on:
  - Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations"
  - Alvarez-Melis & Jaakkola (2018) "On the Robustness of Interpretability Methods"
  """

  alias CrucibleXAI.Explanation

  @type input_result :: %{
          stability_score: float(),
          mean_variation: float(),
          max_variation: float(),
          variation_by_feature: map(),
          coefficient_of_variation: float(),
          interpretation: String.t()
        }

  @doc """
  Test sensitivity to input perturbations.

  Measures how much explanations change when the input is slightly perturbed.
  Stable explanations should be similar for similar inputs.

  ## Algorithm

  1. Generate N small perturbations: x̃ᵢ ≈ x
  2. Compute explanations: φᵢ = explain(x̃ᵢ)
  3. Measure variation:
     - Mean absolute deviation: E[|φᵢ - φ̄|]
     - Max deviation: max|φᵢ - φ̄|
     - Coefficient of variation: σ/μ
  4. Return stability score: 1 - normalized_variation

  ## Parameters

    * `instance` - Instance to test
    * `explain_fn` - Explanation function (instance -> explanation or attributions)
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:num_perturbations` - Number of perturbations (default: 50)
      * `:noise_level` - Relative noise magnitude (default: 0.05)
      * `:noise_type` - `:gaussian` or `:uniform` (default: `:gaussian`)

  ## Returns

  Map with:
    * `:stability_score` - 0-1 (1 = perfectly stable)
    * `:mean_variation` - Average attribution change
    * `:max_variation` - Maximum attribution change
    * `:variation_by_feature` - Per-feature variation map
    * `:coefficient_of_variation` - Normalized stability (σ/μ)
    * `:interpretation` - Stability assessment string

  ## Examples

      explain_fn = fn inst -> %{0 => 2.0, 1 => 3.0} end  # Deterministic
      result = Sensitivity.input_sensitivity(instance, explain_fn, predict_fn)
      # => %{stability_score: 1.0, interpretation: "Excellent", ...}
  """
  @spec input_sensitivity(list(), (any() -> any()), (any() -> any()), keyword()) ::
          input_result()
  def input_sensitivity(instance, explain_fn, _predict_fn, opts \\ []) do
    num_perturbations = Keyword.get(opts, :num_perturbations, 50)
    noise_level = Keyword.get(opts, :noise_level, 0.05)
    noise_type = Keyword.get(opts, :noise_type, :gaussian)

    # Get original explanation
    original_attrs = extract_attributions(explain_fn.(instance))

    # Generate perturbations and compute explanations
    perturbed_attrs =
      1..num_perturbations
      |> Enum.map(fn _ ->
        perturbed = generate_perturbation(instance, noise_level, noise_type)
        extract_attributions(explain_fn.(perturbed))
      end)

    # Compute variation metrics
    all_attrs = [original_attrs | perturbed_attrs]
    feature_indices = Map.keys(original_attrs)

    variation_by_feature =
      feature_indices
      |> Enum.map(fn idx ->
        values = Enum.map(all_attrs, &Map.get(&1, idx, 0.0))
        mean = Enum.sum(values) / length(values)
        variance = compute_variance(values)
        std_dev = :math.sqrt(variance)

        cv = if abs(mean) > 1.0e-10, do: std_dev / abs(mean), else: 0.0

        {idx, %{mean: mean, std_dev: std_dev, coefficient_of_variation: cv}}
      end)
      |> Map.new()

    # Overall variation metrics
    all_std_devs = variation_by_feature |> Map.values() |> Enum.map(& &1.std_dev)
    mean_variation = Enum.sum(all_std_devs) / length(all_std_devs)
    max_variation = Enum.max(all_std_devs)

    all_cvs = variation_by_feature |> Map.values() |> Enum.map(& &1.coefficient_of_variation)
    mean_cv = Enum.sum(all_cvs) / length(all_cvs)

    # Stability score: 1 - normalized variation
    # Use coefficient of variation as the normalized measure
    stability_score = max(0.0, 1.0 - mean_cv)

    # Interpretation
    interpretation = interpret_stability(stability_score)

    %{
      stability_score: stability_score,
      mean_variation: mean_variation,
      max_variation: max_variation,
      variation_by_feature: variation_by_feature,
      coefficient_of_variation: mean_cv,
      interpretation: interpretation
    }
  end

  @doc """
  Test sensitivity to hyperparameters.

  Measures how sensitive explanations are to method-specific parameters
  (e.g., num_samples in LIME, num_coalitions in SHAP).

  ## Parameters

    * `instance` - Instance to test
    * `explain_fn` - Explanation function that accepts options
    * `param_ranges` - Map of parameter => list of values to test
    * `opts` - Additional options

  ## Returns

  Map with:
    * `:parameter_sensitivity` - Variation for each parameter
    * `:robust_parameters` - Parameters with low sensitivity
    * `:sensitive_parameters` - Parameters with high sensitivity
    * `:recommendations` - Tuning recommendations

  ## Examples

      explain_fn = fn inst, opts ->
        CrucibleXai.explain(inst, predict_fn, opts)
      end

      result = Sensitivity.parameter_sensitivity(
        instance,
        explain_fn,
        %{num_samples: [1000, 2000, 5000, 10000]},
        []
      )
      # => %{robust_parameters: [:num_samples], ...}
  """
  @spec parameter_sensitivity(list(), function(), map(), keyword()) :: map()
  def parameter_sensitivity(instance, explain_fn, param_ranges, _opts \\ []) do
    # For each parameter, vary it and measure explanation variation
    parameter_sensitivity =
      param_ranges
      |> Enum.map(fn {param_name, values} ->
        # Get explanations for each parameter value
        explanations =
          values
          |> Enum.map(fn val ->
            attrs = extract_attributions(explain_fn.(instance, [{param_name, val}]))
            {val, attrs}
          end)

        # Compute variation across parameter values
        all_attrs = Enum.map(explanations, fn {_val, attrs} -> attrs end)
        variation = compute_attribution_variation(all_attrs)

        {param_name, %{variation: variation, tested_values: values}}
      end)
      |> Map.new()

    # Classify parameters as robust or sensitive
    # Threshold: CV < 0.2 is robust
    {robust_parameters, sensitive_parameters} =
      parameter_sensitivity
      |> Enum.split_with(fn {_param, %{variation: var}} -> var < 0.2 end)

    robust_params = Enum.map(robust_parameters, fn {param, _} -> param end)
    sensitive_params = Enum.map(sensitive_parameters, fn {param, _} -> param end)

    # Generate recommendations
    recommendations =
      generate_parameter_recommendations(robust_params, sensitive_params, parameter_sensitivity)

    %{
      parameter_sensitivity: parameter_sensitivity,
      robust_parameters: robust_params,
      sensitive_parameters: sensitive_params,
      recommendations: recommendations
    }
  end

  @doc """
  Test cross-method consistency.

  Different explanation methods should agree on which features are important.
  High consistency indicates reliable feature identification.

  ## Parameters

    * `explanations` - List of explanation structs or attribution maps
    * `opts` - Options:
      * `:top_k` - Number of top features to compare (default: 5)
      * `:metric` - `:rank_correlation` or `:overlap` (default: `:rank_correlation`)

  ## Returns

  Map with:
    * `:consistency_score` - 0-1 (1 = perfect agreement)
    * `:pairwise_consistency` - Consistency between each pair of methods
    * `:top_features_overlap` - Overlap in top-k features
    * `:interpretation` - Consistency assessment

  ## Examples

      lime_exp = CrucibleXai.explain(instance, predict_fn)
      shap_exp = CrucibleXai.explain_shap(instance, background, predict_fn)

      result = Sensitivity.cross_method_consistency([lime_exp, shap_exp])
      # => %{consistency_score: 0.85, ...}
  """
  @spec cross_method_consistency(list(), keyword()) :: map()
  def cross_method_consistency(explanations, opts \\ []) do
    top_k = Keyword.get(opts, :top_k, 5)
    metric = Keyword.get(opts, :metric, :rank_correlation)

    # Extract attributions from all explanations
    all_attrs =
      explanations
      |> Enum.with_index()
      |> Enum.map(fn {exp, idx} ->
        attrs = extract_attributions(exp)
        {idx, attrs}
      end)
      |> Map.new()

    # Compute pairwise consistency
    method_indices = Map.keys(all_attrs)

    pairwise_consistency =
      for i <- method_indices, j <- method_indices, i < j do
        attrs_i = Map.get(all_attrs, i)
        attrs_j = Map.get(all_attrs, j)

        score =
          case metric do
            :rank_correlation -> compute_rank_correlation(attrs_i, attrs_j)
            :overlap -> compute_top_k_overlap(attrs_i, attrs_j, top_k)
          end

        {{i, j}, score}
      end
      |> Map.new()

    # Overall consistency score (mean of pairwise scores)
    consistency_score =
      if map_size(pairwise_consistency) > 0 do
        pairwise_consistency
        |> Map.values()
        |> Enum.sum()
        |> Kernel./(map_size(pairwise_consistency))
      else
        1.0
      end

    # Top features overlap
    top_features_by_method =
      all_attrs
      |> Enum.map(fn {idx, attrs} ->
        top_features =
          attrs
          |> Enum.sort_by(fn {_idx, val} -> abs(val) end, :desc)
          |> Enum.take(top_k)
          |> Enum.map(fn {idx, _val} -> idx end)

        {idx, top_features}
      end)
      |> Map.new()

    # Interpretation
    interpretation = interpret_consistency(consistency_score)

    %{
      consistency_score: consistency_score,
      pairwise_consistency: pairwise_consistency,
      top_features_overlap: top_features_by_method,
      interpretation: interpretation
    }
  end

  # Private helper functions

  defp extract_attributions(%Explanation{feature_weights: weights}), do: weights
  defp extract_attributions(attrs) when is_map(attrs), do: attrs

  defp extract_attributions(_), do: %{}

  defp generate_perturbation(instance, noise_level, :gaussian) do
    Enum.map(instance, fn val ->
      std = abs(val) * noise_level
      noise = :rand.normal(0.0, max(std, 0.01))
      val + noise
    end)
  end

  defp generate_perturbation(instance, noise_level, :uniform) do
    Enum.map(instance, fn val ->
      magnitude = abs(val) * noise_level
      noise = (:rand.uniform() - 0.5) * 2.0 * magnitude
      val + noise
    end)
  end

  defp compute_variance(data) when length(data) < 2, do: 0.0

  defp compute_variance(data) do
    n = length(data)
    mean = Enum.sum(data) / n

    sum_squared_diffs =
      data
      |> Enum.map(fn x -> :math.pow(x - mean, 2) end)
      |> Enum.sum()

    sum_squared_diffs / n
  end

  defp compute_attribution_variation(all_attrs) do
    # Get all feature indices
    all_indices =
      all_attrs
      |> Enum.flat_map(&Map.keys/1)
      |> Enum.uniq()

    # Compute CV for each feature
    cvs =
      all_indices
      |> Enum.map(fn idx ->
        values = Enum.map(all_attrs, &Map.get(&1, idx, 0.0))
        mean = Enum.sum(values) / length(values)
        variance = compute_variance(values)
        std_dev = :math.sqrt(variance)

        if abs(mean) > 1.0e-10, do: std_dev / abs(mean), else: 0.0
      end)

    # Mean CV across features
    if length(cvs) > 0, do: Enum.sum(cvs) / length(cvs), else: 0.0
  end

  defp compute_rank_correlation(attrs1, attrs2) do
    # Get common feature indices
    indices1 = Map.keys(attrs1)
    indices2 = Map.keys(attrs2)
    common_indices = Enum.filter(indices1, &(&1 in indices2))

    if length(common_indices) < 2 do
      0.0
    else
      # Rank features by absolute attribution
      ranks1 = rank_attributions(attrs1, common_indices)
      ranks2 = rank_attributions(attrs2, common_indices)

      # Compute Spearman correlation
      spearman_correlation(ranks1, ranks2)
    end
  end

  defp rank_attributions(attrs, indices) do
    indices
    |> Enum.map(fn idx -> {idx, abs(Map.get(attrs, idx, 0.0))} end)
    |> Enum.sort_by(fn {_idx, val} -> val end, :desc)
    |> Enum.with_index(1)
    |> Enum.map(fn {{idx, _val}, rank} -> {idx, rank} end)
    |> Map.new()
  end

  defp spearman_correlation(ranks1, ranks2) do
    indices = Map.keys(ranks1)
    n = length(indices)

    if n < 2 do
      0.0
    else
      rank_pairs =
        Enum.map(indices, fn idx ->
          {Map.get(ranks1, idx), Map.get(ranks2, idx)}
        end)

      sum_squared_diffs =
        rank_pairs
        |> Enum.map(fn {r1, r2} -> :math.pow(r1 - r2, 2) end)
        |> Enum.sum()

      1.0 - 6.0 * sum_squared_diffs / (n * (n * n - 1))
    end
  end

  defp compute_top_k_overlap(attrs1, attrs2, k) do
    top_k1 =
      attrs1
      |> Enum.sort_by(fn {_idx, val} -> abs(val) end, :desc)
      |> Enum.take(k)
      |> Enum.map(fn {idx, _val} -> idx end)
      |> MapSet.new()

    top_k2 =
      attrs2
      |> Enum.sort_by(fn {_idx, val} -> abs(val) end, :desc)
      |> Enum.take(k)
      |> Enum.map(fn {idx, _val} -> idx end)
      |> MapSet.new()

    intersection = MapSet.intersection(top_k1, top_k2)
    union = MapSet.union(top_k1, top_k2)

    if MapSet.size(union) > 0 do
      MapSet.size(intersection) / MapSet.size(union)
    else
      1.0
    end
  end

  defp generate_parameter_recommendations(robust_params, sensitive_params, param_sensitivity) do
    recommendations = []

    recommendations =
      if length(robust_params) > 0 do
        [
          "Parameters #{inspect(robust_params)} are robust - default values are safe"
          | recommendations
        ]
      else
        recommendations
      end

    recommendations =
      if length(sensitive_params) > 0 do
        sensitive_details =
          Enum.map(sensitive_params, fn param ->
            variation = param_sensitivity[param].variation
            "#{param} (CV: #{Float.round(variation, 3)})"
          end)
          |> Enum.join(", ")

        [
          "Parameters #{sensitive_details} are sensitive - careful tuning required"
          | recommendations
        ]
      else
        recommendations
      end

    if length(recommendations) == 0 do
      ["All parameters appear stable"]
    else
      recommendations
    end
  end

  defp interpret_stability(score) do
    cond do
      score >= 0.95 -> "Excellent - Very stable explanations"
      score >= 0.85 -> "Good - Stable explanations"
      score >= 0.70 -> "Acceptable - Moderately stable"
      score >= 0.50 -> "Poor - Unstable explanations"
      true -> "Very Poor - Highly unstable explanations"
    end
  end

  defp interpret_consistency(score) do
    cond do
      score >= 0.90 -> "Excellent - High cross-method agreement"
      score >= 0.75 -> "Good - Moderate agreement"
      score >= 0.60 -> "Acceptable - Some agreement"
      score >= 0.40 -> "Poor - Low agreement"
      true -> "Very Poor - Methods disagree significantly"
    end
  end
end

defmodule CrucibleXAI.Validation.Axioms do
  @moduledoc """
  Theoretical axiom verification for explanation methods.

  Tests whether explanations satisfy key mathematical properties that define
  good attribution methods. Based on Shapley value axioms and attribution theory.

  ## Supported Axioms

  1. **Completeness (Efficiency)**: Sum of attributions equals prediction difference
  2. **Symmetry**: Identical features receive identical attributions
  3. **Dummy (Null Player)**: Features with no impact receive zero attribution
  4. **Linearity**: For linear models, attributions match coefficients

  ## Applicable Methods

  - **SHAP**: Should satisfy completeness, symmetry, and dummy
  - **Integrated Gradients**: Should satisfy completeness
  - **LinearSHAP**: Should satisfy all axioms for linear models
  - **LIME**: Approximate, may violate axioms

  ## Usage

      shap_values = CrucibleXai.explain_shap(instance, background, predict_fn)

      result = Axioms.validate_all_axioms(
        shap_values,
        instance,
        predict_fn,
        method: :shap,
        baseline: background
      )

      IO.inspect(result.all_satisfied)
      # => true (for well-implemented SHAP)

  ## References

  Based on:
  - Shapley (1953) "A Value for N-person Games"
  - Lundberg & Lee (2017) "A Unified Approach to Interpreting Model Predictions"
  - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
  """

  @type completeness_test_result :: %{
          satisfies_completeness: boolean(),
          attribution_sum: number(),
          expected_sum: number(),
          error: number(),
          relative_error: float(),
          interpretation: String.t()
        }

  @type completeness_result ::
          completeness_test_result() | %{skipped: true, reason: String.t()}

  @type symmetry_test_result :: %{
          satisfies_symmetry: boolean(),
          violations: list(),
          max_violation: number(),
          interpretation: String.t()
        }

  @type symmetry_result :: symmetry_test_result() | %{skipped: true, reason: String.t()}

  @type dummy_result :: %{
          satisfies_dummy: boolean(),
          dummy_features: [integer()],
          violations: [integer()],
          interpretation: String.t()
        }

  @type linearity_test_result :: %{
          satisfies_linearity: boolean(),
          errors_by_feature: map(),
          expected_shap: map(),
          max_error: number(),
          interpretation: String.t()
        }

  @type linearity_result :: linearity_test_result() | %{skipped: true, reason: String.t()}

  @type axioms_result :: %{
          completeness: completeness_result(),
          symmetry: symmetry_result(),
          dummy: dummy_result(),
          linearity: linearity_result(),
          all_satisfied: boolean(),
          overall_score: float(),
          summary: String.t()
        }

  @doc """
  Test completeness (efficiency) axiom.

  For SHAP: Σφᵢ should equal f(x) - E[f(x)] (baseline prediction)
  For Integrated Gradients: Σφᵢ should equal f(x) - f(baseline)

  ## Parameters

    * `attributions` - Attribution map (feature_index => value)
    * `instance` - Instance that was explained
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:method` - :shap, :integrated_gradients, :other
      * `:baseline` - Baseline instance(s) or value
      * `:tolerance` - Acceptable error (default: 0.1)

  ## Returns

  Map with:
    * `:satisfies_completeness` - Boolean
    * `:attribution_sum` - Actual sum of attributions
    * `:expected_sum` - Expected sum (f(x) - baseline)
    * `:error` - Absolute difference
    * `:relative_error` - |error| / |expected|
    * `:interpretation` - Assessment string

  ## Examples

      # SHAP should satisfy completeness
      attributions = %{0 => 2.0, 1 => 3.0}
      instance = [5.0, 10.0]
      baseline = [[0.0, 0.0], [1.0, 1.0]]

      result = Axioms.test_completeness(
        attributions,
        instance,
        predict_fn,
        method: :shap,
        baseline: baseline
      )
      # => %{satisfies_completeness: true, error: 0.01, ...}
  """
  @spec test_completeness(map(), list(), (any() -> any()), keyword()) ::
          completeness_test_result()
  def test_completeness(attributions, instance, predict_fn, opts \\ []) do
    method = Keyword.get(opts, :method, :other)
    baseline = Keyword.get(opts, :baseline)
    tolerance = Keyword.get(opts, :tolerance, 0.1)

    # Compute sum of attributions
    attribution_sum =
      attributions
      |> Map.values()
      |> Enum.sum()

    # Compute expected sum based on method
    expected_sum =
      case method do
        :shap ->
          compute_shap_expected_sum(instance, baseline, predict_fn)

        :integrated_gradients ->
          compute_ig_expected_sum(instance, baseline, predict_fn)

        _ ->
          # Generic: f(x) - f(baseline) if baseline provided
          if baseline do
            baseline_inst = if is_list(hd(baseline)), do: hd(baseline), else: baseline
            pred_instance = get_prediction(predict_fn, instance)
            pred_baseline = get_prediction(predict_fn, baseline_inst)
            pred_instance - pred_baseline
          else
            # No baseline - can't verify
            attribution_sum
          end
      end

    # Compute error
    error = abs(attribution_sum - expected_sum)
    relative_error = if abs(expected_sum) > 1.0e-10, do: error / abs(expected_sum), else: 0.0

    # Check if satisfies axiom
    satisfies_completeness = error <= tolerance

    # Interpretation
    interpretation = interpret_completeness(satisfies_completeness, relative_error)

    %{
      satisfies_completeness: satisfies_completeness,
      attribution_sum: attribution_sum,
      expected_sum: expected_sum,
      error: error,
      relative_error: relative_error,
      interpretation: interpretation
    }
  end

  @doc """
  Test symmetry axiom.

  Features with identical marginal contributions should receive identical
  attributions. This is difficult to test in general, so we use a heuristic
  approach for symmetric features.

  ## Parameters

    * `attributions` - SHAP values or attributions
    * `instance` - Instance explained
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:symmetric_pairs` - List of {idx1, idx2} feature pairs to test
      * `:tolerance` - Acceptable difference (default: 0.1)

  ## Returns

  Map with:
    * `:satisfies_symmetry` - Boolean
    * `:violations` - List of feature pairs that violate symmetry
    * `:max_violation` - Maximum observed violation

  ## Examples

      # Test two features known to be symmetric
      attributions = %{0 => 2.0, 1 => 2.0}

      result = Axioms.test_symmetry(
        attributions,
        instance,
        predict_fn,
        symmetric_pairs: [{0, 1}]
      )
      # => %{satisfies_symmetry: true, violations: [], ...}
  """
  @spec test_symmetry(map(), list(), (any() -> any()), keyword()) :: symmetry_test_result()
  def test_symmetry(attributions, _instance, _predict_fn, opts \\ []) do
    symmetric_pairs = Keyword.get(opts, :symmetric_pairs, [])
    tolerance = Keyword.get(opts, :tolerance, 0.1)

    violations =
      symmetric_pairs
      |> Enum.filter(fn {idx1, idx2} ->
        attr1 = Map.get(attributions, idx1, 0.0)
        attr2 = Map.get(attributions, idx2, 0.0)
        abs(attr1 - attr2) > tolerance
      end)

    max_violation =
      if length(symmetric_pairs) > 0 do
        symmetric_pairs
        |> Enum.map(fn {idx1, idx2} ->
          attr1 = Map.get(attributions, idx1, 0.0)
          attr2 = Map.get(attributions, idx2, 0.0)
          abs(attr1 - attr2)
        end)
        |> Enum.max()
      else
        0.0
      end

    satisfies_symmetry = length(violations) == 0

    %{
      satisfies_symmetry: satisfies_symmetry,
      violations: violations,
      max_violation: max_violation,
      interpretation:
        if(satisfies_symmetry,
          do: "Symmetry axiom satisfied",
          else: "Symmetry violations detected"
        )
    }
  end

  @doc """
  Test dummy (null player) axiom.

  Features that don't affect model output should have zero attribution.

  ## Algorithm

  1. For each feature i:
     a. Vary feature i while fixing others
     b. Measure prediction change
  2. If Δf ≈ 0 for all variations, then φᵢ should ≈ 0

  ## Parameters

    * `attributions` - Attribution map
    * `instance` - Instance explained
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:num_variations` - Variations to test per feature (default: 10)
      * `:tolerance` - Attribution tolerance for dummy features (default: 0.1)
      * `:prediction_tolerance` - Prediction change tolerance (default: 0.01)

  ## Returns

  Map with:
    * `:satisfies_dummy` - Boolean
    * `:dummy_features` - Features identified as dummy
    * `:violations` - Dummy features with non-zero attribution
    * `:interpretation` - Assessment string

  ## Examples

      # Feature 2 is a dummy (doesn't affect prediction)
      predict_fn = fn [x, y, _z] -> 2.0 * x + 3.0 * y end
      attributions = %{0 => 2.0, 1 => 3.0, 2 => 0.0}

      result = Axioms.test_dummy(attributions, [5.0, 10.0, 7.0], predict_fn)
      # => %{satisfies_dummy: true, dummy_features: [2], ...}
  """
  @spec test_dummy(map(), list(), (any() -> any()), keyword()) :: dummy_result()
  def test_dummy(attributions, instance, predict_fn, opts \\ []) do
    num_variations = Keyword.get(opts, :num_variations, 10)
    tolerance = Keyword.get(opts, :tolerance, 0.1)
    prediction_tolerance = Keyword.get(opts, :prediction_tolerance, 0.01)

    # Identify dummy features
    feature_indices = 0..(length(instance) - 1) |> Enum.to_list()

    dummy_features =
      feature_indices
      |> Enum.filter(fn idx ->
        is_dummy_feature?(instance, idx, predict_fn, num_variations, prediction_tolerance)
      end)

    # Check if dummy features have near-zero attribution
    violations =
      dummy_features
      |> Enum.filter(fn idx ->
        abs(Map.get(attributions, idx, 0.0)) > tolerance
      end)

    satisfies_dummy = length(violations) == 0

    interpretation =
      if satisfies_dummy do
        "Dummy axiom satisfied - all dummy features have zero attribution"
      else
        "Dummy axiom violated - #{length(violations)} dummy features have non-zero attribution"
      end

    %{
      satisfies_dummy: satisfies_dummy,
      dummy_features: dummy_features,
      violations: violations,
      interpretation: interpretation
    }
  end

  @doc """
  Test linearity axiom (for linear models only).

  For linear model f(x) = wᵀx + b:
  SHAP values should exactly equal: φᵢ = wᵢ(xᵢ - E[xᵢ])

  ## Parameters

    * `shap_values` - SHAP attribution map
    * `instance` - Instance explained
    * `model_coefficients` - Map of feature_index => weight
    * `opts` - Options:
      * `:baseline` - Baseline values for features (E[x])
      * `:tolerance` - Acceptable error (default: 0.1)

  ## Returns

  Map with:
    * `:satisfies_linearity` - Boolean
    * `:errors_by_feature` - Error for each feature
    * `:max_error` - Maximum observed error
    * `:interpretation` - Assessment string

  ## Examples

      # Linear model: f(x) = 2x₁ + 3x₂
      coefficients = %{0 => 2.0, 1 => 3.0}
      instance = [5.0, 10.0]
      baseline = [0.0, 0.0]  # E[x]
      shap_values = %{0 => 10.0, 1 => 30.0}  # 2*(5-0), 3*(10-0)

      result = Axioms.test_linearity(
        shap_values,
        instance,
        coefficients,
        baseline: baseline
      )
      # => %{satisfies_linearity: true, ...}
  """
  @spec test_linearity(map(), list(), map(), keyword()) :: linearity_test_result()
  def test_linearity(shap_values, instance, model_coefficients, opts \\ []) do
    baseline = Keyword.get(opts, :baseline, List.duplicate(0.0, length(instance)))
    tolerance = Keyword.get(opts, :tolerance, 0.1)

    # Compute expected SHAP values: φᵢ = wᵢ(xᵢ - E[xᵢ])
    expected_shap =
      instance
      |> Enum.with_index()
      |> Enum.map(fn {x_i, i} ->
        w_i = Map.get(model_coefficients, i, 0.0)
        baseline_i = Enum.at(baseline, i, 0.0)
        expected = w_i * (x_i - baseline_i)
        {i, expected}
      end)
      |> Map.new()

    # Compute errors
    errors_by_feature =
      expected_shap
      |> Enum.map(fn {i, expected} ->
        actual = Map.get(shap_values, i, 0.0)
        error = abs(actual - expected)
        {i, error}
      end)
      |> Map.new()

    max_error =
      if map_size(errors_by_feature) > 0,
        do: Map.values(errors_by_feature) |> Enum.max(),
        else: 0.0

    satisfies_linearity = max_error <= tolerance

    interpretation =
      if satisfies_linearity do
        "Linearity axiom satisfied - attributions match model coefficients"
      else
        "Linearity axiom violated - max error: #{Float.round(max_error, 4)}"
      end

    %{
      satisfies_linearity: satisfies_linearity,
      errors_by_feature: errors_by_feature,
      max_error: max_error,
      expected_shap: expected_shap,
      interpretation: interpretation
    }
  end

  @doc """
  Comprehensive axiom validation suite.

  Runs all applicable axiom tests for the given method and returns a
  complete validation report.

  ## Parameters

    * `attributions` - Attribution map
    * `instance` - Instance explained
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:method` - Method type (:shap, :integrated_gradients, :lime, etc.)
      * `:baseline` - Baseline for completeness test
      * `:model_coefficients` - For linearity test (optional)
      * `:symmetric_pairs` - For symmetry test (optional)

  ## Returns

  Map with:
    * `:completeness` - Completeness test results
    * `:symmetry` - Symmetry test results (if applicable)
    * `:dummy` - Dummy test results
    * `:linearity` - Linearity test results (if applicable)
    * `:all_satisfied` - Whether all applicable axioms are satisfied
    * `:overall_score` - 0-1 score (fraction of axioms satisfied)
    * `:summary` - Human-readable summary

  ## Examples

      result = Axioms.validate_all_axioms(
        shap_values,
        instance,
        predict_fn,
        method: :shap,
        baseline: background
      )

      IO.puts(result.summary)
  """
  @spec validate_all_axioms(map(), list(), (any() -> any()), keyword()) :: axioms_result()
  def validate_all_axioms(attributions, instance, predict_fn, opts \\ []) do
    method = Keyword.get(opts, :method, :other)

    # Test completeness (always applicable for SHAP and IG)
    completeness =
      if method in [:shap, :integrated_gradients] do
        test_completeness(attributions, instance, predict_fn, opts)
      else
        %{skipped: true, reason: "Method does not guarantee completeness"}
      end

    # Test symmetry (if symmetric pairs provided)
    symmetry =
      if Keyword.has_key?(opts, :symmetric_pairs) do
        test_symmetry(attributions, instance, predict_fn, opts)
      else
        %{skipped: true, reason: "No symmetric feature pairs specified"}
      end

    # Test dummy (always applicable)
    dummy = test_dummy(attributions, instance, predict_fn, opts)

    # Test linearity (if model coefficients provided)
    linearity =
      if Keyword.has_key?(opts, :model_coefficients) do
        test_linearity(attributions, instance, Keyword.get(opts, :model_coefficients), opts)
      else
        %{skipped: true, reason: "Model coefficients not provided"}
      end

    # Compute overall satisfaction
    tests = [completeness, symmetry, dummy, linearity]

    satisfied_tests =
      tests
      |> Enum.reject(&Map.has_key?(&1, :skipped))
      |> Enum.count(fn test ->
        Map.get(test, :satisfies_completeness, false) or
          Map.get(test, :satisfies_symmetry, false) or
          Map.get(test, :satisfies_dummy, false) or
          Map.get(test, :satisfies_linearity, false)
      end)

    total_tests = tests |> Enum.reject(&Map.has_key?(&1, :skipped)) |> length()

    all_satisfied = satisfied_tests == total_tests and total_tests > 0
    overall_score = if total_tests > 0, do: satisfied_tests / total_tests, else: 1.0

    # Generate summary
    summary = generate_axiom_summary(completeness, symmetry, dummy, linearity, all_satisfied)

    %{
      completeness: completeness,
      symmetry: symmetry,
      dummy: dummy,
      linearity: linearity,
      all_satisfied: all_satisfied,
      overall_score: overall_score,
      summary: summary
    }
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

  defp compute_shap_expected_sum(instance, baseline, predict_fn) when is_list(baseline) do
    # Baseline is a list of instances (background dataset)
    # Expected: f(x) - E[f(background)]
    pred_instance = get_prediction(predict_fn, instance)

    baseline_preds =
      baseline
      |> Enum.map(&get_prediction(predict_fn, &1))

    baseline_mean = Enum.sum(baseline_preds) / length(baseline_preds)

    pred_instance - baseline_mean
  end

  defp compute_shap_expected_sum(instance, baseline, predict_fn) do
    # Baseline is a single value or instance
    pred_instance = get_prediction(predict_fn, instance)
    pred_baseline = get_prediction(predict_fn, baseline)

    pred_instance - pred_baseline
  end

  defp compute_ig_expected_sum(instance, baseline, predict_fn) do
    # For IG: f(x) - f(baseline)
    pred_instance = get_prediction(predict_fn, instance)
    pred_baseline = get_prediction(predict_fn, baseline)

    pred_instance - pred_baseline
  end

  defp is_dummy_feature?(instance, feature_idx, predict_fn, num_variations, tolerance) do
    # Get original prediction
    original_pred = get_prediction(predict_fn, instance)

    # Generate variations of the feature
    original_value = Enum.at(instance, feature_idx)

    variations =
      1..num_variations
      |> Enum.map(fn _ ->
        # Generate random value in reasonable range
        new_value = original_value + (:rand.uniform() - 0.5) * 10.0

        # Create modified instance
        modified =
          instance
          |> Enum.with_index()
          |> Enum.map(fn {val, idx} ->
            if idx == feature_idx, do: new_value, else: val
          end)

        # Get prediction
        get_prediction(predict_fn, modified)
      end)

    # Check if all predictions are close to original
    max_change =
      variations
      |> Enum.map(&abs(&1 - original_pred))
      |> Enum.max()

    max_change <= tolerance
  end

  defp interpret_completeness(true, _relative_error) do
    "Completeness axiom satisfied"
  end

  defp interpret_completeness(false, relative_error) do
    "Completeness axiom violated - relative error: #{Float.round(relative_error * 100, 2)}%"
  end

  defp generate_axiom_summary(completeness, symmetry, dummy, linearity, all_satisfied) do
    status = if all_satisfied, do: "✓ All axioms satisfied", else: "✗ Some axioms violated"

    completeness_str =
      unless Map.has_key?(completeness, :skipped) do
        if completeness.satisfies_completeness do
          "  ✓ Completeness: satisfied (error: #{Float.round(completeness.error, 4)})"
        else
          "  ✗ Completeness: violated (error: #{Float.round(completeness.error, 4)})"
        end
      else
        "  - Completeness: skipped"
      end

    symmetry_str =
      unless Map.has_key?(symmetry, :skipped) do
        if symmetry.satisfies_symmetry do
          "  ✓ Symmetry: satisfied"
        else
          "  ✗ Symmetry: #{length(symmetry.violations)} violations"
        end
      else
        "  - Symmetry: skipped"
      end

    dummy_str =
      unless Map.has_key?(dummy, :skipped) do
        if dummy.satisfies_dummy do
          "  ✓ Dummy: satisfied (#{length(dummy.dummy_features)} dummy features)"
        else
          "  ✗ Dummy: #{length(dummy.violations)} violations"
        end
      else
        "  - Dummy: skipped"
      end

    linearity_str =
      unless Map.has_key?(linearity, :skipped) do
        if linearity.satisfies_linearity do
          "  ✓ Linearity: satisfied"
        else
          "  ✗ Linearity: violated (max error: #{Float.round(linearity.max_error, 4)})"
        end
      else
        "  - Linearity: skipped"
      end

    """
    === Axiom Validation Summary ===
    #{status}

    #{completeness_str}
    #{symmetry_str}
    #{dummy_str}
    #{linearity_str}
    """
  end
end

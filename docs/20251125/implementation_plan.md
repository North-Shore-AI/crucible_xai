# Validation Metrics Suite - Implementation Plan

**Date**: November 25, 2025
**Version**: v0.3.0
**Status**: Implementation Ready
**Methodology**: Test-Driven Development (TDD)

---

## Implementation Overview

This document provides a detailed, step-by-step implementation plan for the Validation & Quality Metrics Suite. The implementation follows strict TDD methodology: RED (write failing tests) → GREEN (implement minimum code to pass) → REFACTOR (optimize and clean up).

---

## Environment Setup

### Prerequisites

```bash
# Ensure Elixir 1.14+ and OTP 25+ installed
elixir --version

# Navigate to project
cd /home/home/p/g/North-Shore-AI/crucible_xai

# Fetch dependencies
mix deps.get

# Verify tests pass
mix test

# Expected: 277 tests passing, 0 failures
```

### Create Module Structure

```bash
# Create validation directory
mkdir -p lib/crucible_xai/validation
mkdir -p test/crucible_xai/validation

# Create module files
touch lib/crucible_xai/validation.ex
touch lib/crucible_xai/validation/faithfulness.ex
touch lib/crucible_xai/validation/infidelity.ex
touch lib/crucible_xai/validation/sensitivity.ex
touch lib/crucible_xai/validation/axioms.ex

# Create test files
touch test/crucible_xai/validation_test.exs
touch test/crucible_xai/validation/faithfulness_test.exs
touch test/crucible_xai/validation/infidelity_test.exs
touch test/crucible_xai/validation/sensitivity_test.exs
touch test/crucible_xai/validation/axioms_test.exs
```

---

## Phase 1: Faithfulness Module (Week 1)

### Day 1: RED Phase - Feature Removal Tests

**File**: `test/crucible_xai/validation/faithfulness_test.exs`

```elixir
defmodule CrucibleXAI.Validation.FaithfulnessTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.Validation.Faithfulness
  alias CrucibleXAI.Explanation

  describe "measure_faithfulness/4" do
    test "perfect faithfulness for linear model with correct weights" do
      # Linear model: f(x) = 2x₁ + 3x₂
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end
      instance = [5.0, 10.0]

      # Correct explanation
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0, 1 => 3.0},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

      # Perfect correlation expected
      assert result.faithfulness_score >= 0.95
      assert result.monotonicity == true
      assert result.interpretation =~ "Excellent"
    end

    test "poor faithfulness for random attributions" do
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end
      instance = [5.0, 10.0]

      # Random wrong weights
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

      # Poor correlation expected
      assert result.faithfulness_score < 0.6
      assert result.interpretation =~ "Poor" or result.interpretation =~ "Weak"
    end

    test "handles different baseline strategies" do
      predict_fn = fn [x] -> x * 2.0 end
      instance = [5.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0},
        method: :lime
      }

      # Test with zero baseline
      result1 = Faithfulness.measure_faithfulness(
        instance,
        explanation,
        predict_fn,
        baseline_value: 0.0
      )

      # Test with mean baseline
      result2 = Faithfulness.measure_faithfulness(
        instance,
        explanation,
        predict_fn,
        baseline_value: 2.5
      )

      assert is_float(result1.faithfulness_score)
      assert is_float(result2.faithfulness_score)
      # Scores may differ but both should be valid
    end

    test "property: faithfulness_score in [-1, 1]" do
      predict_fn = fn [x, y] -> x + y end
      instance = [1.0, 2.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 1.0, 1 => 1.0},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

      assert result.faithfulness_score >= -1.0
      assert result.faithfulness_score <= 1.0
    end

    test "handles single feature" do
      predict_fn = fn [x] -> x * 3.0 end
      instance = [5.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 3.0},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

      # Single feature: perfect correlation or undefined
      assert is_map(result)
      assert Map.has_key?(result, :faithfulness_score)
    end
  end

  describe "monotonicity_test/4" do
    test "detects perfect monotonicity" do
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end
      instance = [5.0, 10.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0, 1 => 3.0},
        method: :lime
      }

      result = Faithfulness.monotonicity_test(instance, explanation, predict_fn)

      assert result.is_monotonic == true
      assert result.violations == 0
      assert result.violation_indices == []
    end

    test "detects monotonicity violations" do
      # Non-monotonic model
      predict_fn = fn [x] ->
        cond do
          x < 3.0 -> 10.0
          x < 7.0 -> 5.0
          true -> 15.0
        end
      end
      instance = [5.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0},
        method: :lime
      }

      result = Faithfulness.monotonicity_test(instance, explanation, predict_fn)

      # May have violations due to non-monotonic model
      assert is_boolean(result.is_monotonic)
      assert is_integer(result.violations)
      assert is_list(result.violation_indices)
    end

    test "calculates violation severity" do
      predict_fn = fn [x] -> if x > 5, do: x * 2, else: x * 3 end
      instance = [4.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.5},
        method: :lime
      }

      result = Faithfulness.monotonicity_test(instance, explanation, predict_fn)

      assert is_float(result.severity)
      assert result.severity >= 0.0
    end
  end

  describe "full_report/4" do
    test "generates comprehensive report" do
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end
      instance = [5.0, 10.0]
      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => 2.0, 1 => 3.0},
        method: :lime
      }

      report = Faithfulness.full_report(instance, explanation, predict_fn)

      assert Map.has_key?(report, :faithfulness_score)
      assert Map.has_key?(report, :monotonicity)
      assert Map.has_key?(report, :prediction_drops)
      assert Map.has_key?(report, :interpretation)
      assert is_binary(report.interpretation)
    end
  end
end
```

**Run tests (should fail)**:
```bash
mix test test/crucible_xai/validation/faithfulness_test.exs
# Expected: All tests fail (modules not implemented)
```

### Day 2: RED Phase - Additional Faithfulness Tests

Continue adding more test cases for edge cases, batch processing, and property-based tests.

```elixir
# Add to faithfulness_test.exs

describe "edge cases" do
  test "all features equal importance" do
    predict_fn = fn [x1, x2, x3] -> x1 + x2 + x3 end
    instance = [1.0, 1.0, 1.0]
    explanation = %Explanation{
      instance: instance,
      feature_weights: %{0 => 1.0, 1 => 1.0, 2 => 1.0},
      method: :lime
    }

    result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

    assert is_map(result)
    # All features equal: correlation may be perfect or undefined
  end

  test "handles zero predictions gracefully" do
    predict_fn = fn _ -> 0.0 end
    instance = [5.0]
    explanation = %Explanation{
      instance: instance,
      feature_weights: %{0 => 1.0},
      method: :lime
    }

    result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

    assert is_map(result)
    # Zero predictions: should not crash
  end
end

# Add property-based tests
describe "property-based tests" do
  use ExUnitProperties

  property "faithfulness score always in valid range" do
    check all(
      weights <- list_of(float(min: -10.0, max: 10.0), length: 3),
      instance <- list_of(float(min: -10.0, max: 10.0), length: 3)
    ) do
      predict_fn = fn [x1, x2, x3] ->
        Enum.at(weights, 0) * x1 +
        Enum.at(weights, 1) * x2 +
        Enum.at(weights, 2) * x3
      end

      explanation = %Explanation{
        instance: instance,
        feature_weights: %{0 => Enum.at(weights, 0), 1 => Enum.at(weights, 1), 2 => Enum.at(weights, 2)},
        method: :lime
      }

      result = Faithfulness.measure_faithfulness(instance, explanation, predict_fn)

      assert result.faithfulness_score >= -1.0
      assert result.faithfulness_score <= 1.0
    end
  end
end
```

### Days 3-4: GREEN Phase - Implement Faithfulness

**File**: `lib/crucible_xai/validation/faithfulness.ex`

```elixir
defmodule CrucibleXAI.Validation.Faithfulness do
  @moduledoc """
  Faithfulness metrics for explanation validation.

  Measures how well explanations reflect actual model behavior by testing
  whether removing important features causes proportional prediction changes.
  """

  alias CrucibleXAI.Explanation

  @doc """
  Measure faithfulness via feature removal.

  ## Algorithm
  1. Sort features by absolute attribution
  2. Remove features incrementally (most important first)
  3. Measure prediction change at each step
  4. Compute correlation between attribution rank and prediction change

  ## Parameters
    * `instance` - Instance to test
    * `explanation` - Explanation to validate
    * `predict_fn` - Model prediction function
    * `opts` - Options:
      * `:baseline_value` - Value for removed features (default: 0.0)
      * `:num_steps` - Number of removal steps (default: all features)
      * `:correlation_method` - :pearson or :spearman (default: :spearman)

  ## Returns
    Map with faithfulness metrics
  """
  @spec measure_faithfulness(list(), Explanation.t(), function(), keyword()) :: map()
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
    ranks = Enum.with_index(sorted_features) |> Enum.map(fn {_, idx} -> idx + 1 end)
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
  """
  @spec monotonicity_test(list(), Explanation.t(), function(), keyword()) :: map()
  def monotonicity_test(instance, explanation, predict_fn, opts \\ []) do
    baseline_value = Keyword.get(opts, :baseline_value, 0.0)

    # Get prediction drops
    result = measure_faithfulness(instance, explanation, predict_fn,
                                   Keyword.put(opts, :baseline_value, baseline_value))

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
    avg_severity = if length(severities) > 0, do: Enum.sum(severities) / length(severities), else: 0.0

    %{
      is_monotonic: is_monotonic,
      violations: violations,
      violation_indices: violation_indices,
      severity: avg_severity
    }
  end

  @doc """
  Generate comprehensive faithfulness report.
  """
  @spec full_report(list(), Explanation.t(), function(), keyword()) :: map()
  def full_report(instance, explanation, predict_fn, opts \\ []) do
    faithfulness = measure_faithfulness(instance, explanation, predict_fn, opts)
    monotonicity = monotonicity_test(instance, explanation, predict_fn, opts)

    Map.merge(faithfulness, %{
      monotonicity_details: monotonicity,
      summary: generate_summary(faithfulness, monotonicity)
    })
  end

  # Private helpers

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)
    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
      _ -> raise "Prediction function must return number or Nx.Tensor"
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
```

### Day 5: REFACTOR - Optimize and Polish

```bash
# Run tests
mix test test/crucible_xai/validation/faithfulness_test.exs

# Expected: All tests should now pass

# Check for warnings
mix compile --warnings-as-errors

# Format code
mix format

# Run dialyzer
mix dialyzer
```

**Refactoring tasks**:
1. Add @spec annotations for all functions
2. Optimize correlation computation for large datasets
3. Add comprehensive @doc documentation
4. Handle edge cases more gracefully
5. Add batch processing support

---

## Phase 2: Infidelity Module (Week 2)

### Days 1-2: RED Phase

Similar structure: write 12 failing tests for infidelity computation.

### Days 3-4: GREEN Phase

Implement the infidelity module with perturbation generation and error measurement.

### Day 5: REFACTOR

Optimize, add parallel perturbation processing, add caching.

---

## Phase 3: Sensitivity Module (Week 3)

### Days 1-2: RED Phase

Write 15 failing tests for input sensitivity, parameter sensitivity, and cross-method consistency.

### Days 3-4: GREEN Phase

Implement sensitivity analysis with adaptive sampling strategies.

### Day 5: REFACTOR

Optimize for batch validation, add progress reporting for long operations.

---

## Phase 4: Axioms & Integration (Week 4)

### Days 1-2: RED Phase

Write axiom tests, integration tests, and property-based tests.

### Days 3-4: GREEN Phase

Implement axiom validation and main API integration.

### Day 5: POLISH

Complete documentation, examples, performance benchmarking.

---

## Main API Integration

**File**: `lib/crucible_xai/validation.ex`

```elixir
defmodule CrucibleXAI.Validation do
  @moduledoc """
  Main API for explanation validation and quality metrics.

  Provides comprehensive validation tools to measure faithfulness,
  infidelity, sensitivity, and axiom compliance of explanations.

  ## Usage

      explanation = CrucibleXai.explain(instance, predict_fn)

      validation = CrucibleXAI.Validation.comprehensive_validation(
        explanation,
        instance,
        predict_fn
      )

      IO.inspect(validation.summary)
  """

  alias CrucibleXAI.Validation.{Faithfulness, Infidelity, Sensitivity, Axioms}
  alias CrucibleXAI.Explanation

  @doc """
  Comprehensive validation of an explanation.

  Runs all validation metrics and returns a complete report.

  ## Parameters
    * `explanation` - Explanation to validate
    * `instance` - Instance that was explained
    * `predict_fn` - Model prediction function
    * `opts` - Options for individual metrics

  ## Returns
    Complete validation report with all metrics
  """
  @spec comprehensive_validation(Explanation.t(), list(), function(), keyword()) :: map()
  def comprehensive_validation(%Explanation{} = explanation, instance, predict_fn, opts \\ []) do
    # Run all validation metrics
    faithfulness = Faithfulness.measure_faithfulness(instance, explanation, predict_fn, opts)

    attributions = explanation.feature_weights
    infidelity = Infidelity.compute(instance, attributions, predict_fn, opts)

    # Sensitivity (if requested)
    sensitivity =
      if Keyword.get(opts, :include_sensitivity, false) do
        explain_fn = fn inst ->
          CrucibleXai.explain(inst, predict_fn, Keyword.get(opts, :explain_opts, []))
        end
        Sensitivity.input_sensitivity(instance, explain_fn, predict_fn, opts)
      else
        %{skipped: true}
      end

    # Axioms (method-specific)
    axioms = Axioms.validate_all_axioms(attributions, instance, predict_fn,
                                        Keyword.put(opts, :method, explanation.method))

    # Generate overall quality score
    quality_score = calculate_quality_score(faithfulness, infidelity, axioms)

    %{
      faithfulness: faithfulness,
      infidelity: infidelity,
      sensitivity: sensitivity,
      axioms: axioms,
      quality_score: quality_score,
      summary: generate_validation_summary(faithfulness, infidelity, axioms, quality_score)
    }
  end

  @doc """
  Quick validation with essential metrics only.

  Runs faithfulness and infidelity (fast metrics).
  """
  @spec quick_validation(Explanation.t(), list(), function(), keyword()) :: map()
  def quick_validation(explanation, instance, predict_fn, opts \\ []) do
    faithfulness = Faithfulness.measure_faithfulness(instance, explanation, predict_fn, opts)
    infidelity = Infidelity.compute(instance, explanation.feature_weights, predict_fn, opts)

    %{
      faithfulness_score: faithfulness.faithfulness_score,
      infidelity_score: infidelity.infidelity_score,
      passes_quality_gate: passes_quality_gate?(faithfulness, infidelity)
    }
  end

  # Private helpers

  defp calculate_quality_score(faithfulness, infidelity, axioms) do
    # Weighted combination of metrics
    # Faithfulness: 40%, Infidelity: 40%, Axioms: 20%

    faith_score = normalize_score(faithfulness.faithfulness_score, :correlation)
    infid_score = normalize_score(infidelity.infidelity_score, :error)
    axiom_score = if Map.has_key?(axioms, :overall_score), do: axioms.overall_score, else: 0.8

    (0.4 * faith_score + 0.4 * infid_score + 0.2 * axiom_score)
  end

  defp normalize_score(score, :correlation) do
    # Correlation: -1 to 1 → 0 to 1
    (score + 1.0) / 2.0
  end

  defp normalize_score(score, :error) do
    # Error: 0 to ∞ → 1 to 0 (exponential decay)
    :math.exp(-score)
  end

  defp passes_quality_gate?(faithfulness, infidelity) do
    faithfulness.faithfulness_score >= 0.7 and infidelity.infidelity_score <= 0.1
  end

  defp generate_validation_summary(faithfulness, infidelity, axioms, quality_score) do
    """
    === Explanation Quality Report ===

    Overall Quality Score: #{Float.round(quality_score, 3)} / 1.0

    Faithfulness: #{Float.round(faithfulness.faithfulness_score, 3)}
      - #{faithfulness.interpretation}

    Infidelity: #{Float.round(infidelity.infidelity_score, 4)}
      - #{infidelity.interpretation}

    Axioms: #{if axioms.all_satisfied, do: "✓ All satisfied", else: "✗ Some violations"}

    Recommendation: #{recommend_action(quality_score)}
    """
  end

  defp recommend_action(score) when score >= 0.85, do: "Excellent - Safe for production"
  defp recommend_action(score) when score >= 0.70, do: "Good - Acceptable for most uses"
  defp recommend_action(score) when score >= 0.50, do: "Fair - Use with caution"
  defp recommend_action(_), do: "Poor - Explanation may be unreliable"
end
```

---

## Testing Commands

### Run Individual Module Tests

```bash
# Faithfulness
mix test test/crucible_xai/validation/faithfulness_test.exs

# Infidelity
mix test test/crucible_xai/validation/infidelity_test.exs

# Sensitivity
mix test test/crucible_xai/validation/sensitivity_test.exs

# Axioms
mix test test/crucible_xai/validation/axioms_test.exs
```

### Run All Validation Tests

```bash
mix test test/crucible_xai/validation/
```

### Run Full Test Suite

```bash
# All tests
mix test

# With coverage
mix test --cover

# Expected final results:
# - Total tests: 337 (277 existing + 60 new)
# - All passing
# - Coverage: >96%
```

### Quality Checks

```bash
# No warnings
mix compile --warnings-as-errors

# Type checking
mix dialyzer

# Code quality
mix credo --strict

# Format check
mix format --check-formatted
```

---

## Integration Tests

**File**: `test/crucible_xai/validation_test.exs`

```elixir
defmodule CrucibleXAI.ValidationTest do
  use ExUnit.Case, async: false

  alias CrucibleXAI.Validation

  describe "integration with LIME" do
    test "validates LIME explanation" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 10.0]

      explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

      validation = Validation.comprehensive_validation(
        explanation,
        instance,
        predict_fn
      )

      assert validation.quality_score > 0.8
      assert validation.faithfulness.faithfulness_score > 0.85
      assert validation.infidelity.infidelity_score < 0.1
    end
  end

  describe "integration with SHAP" do
    test "validates KernelSHAP explanation" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 10.0]
      background = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

      shap_values = CrucibleXai.explain_shap(instance, background, predict_fn,
                                              num_samples: 1000)

      # Convert to explanation format
      explanation = %CrucibleXAI.Explanation{
        instance: instance,
        feature_weights: shap_values,
        method: :shap
      }

      validation = Validation.comprehensive_validation(
        explanation,
        instance,
        predict_fn,
        method: :shap,
        baseline: background
      )

      # SHAP should satisfy additivity
      assert validation.axioms.completeness.satisfies_completeness == true
      assert validation.quality_score > 0.8
    end
  end

  describe "quick_validation" do
    test "provides fast quality check" do
      predict_fn = fn [x] -> x * 2.0 end
      instance = [5.0]

      explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 1000)

      result = Validation.quick_validation(explanation, instance, predict_fn)

      assert Map.has_key?(result, :faithfulness_score)
      assert Map.has_key?(result, :infidelity_score)
      assert Map.has_key?(result, :passes_quality_gate)
      assert is_boolean(result.passes_quality_gate)
    end
  end
end
```

---

## Performance Benchmarking

Create a benchmark script to measure performance:

**File**: `scripts/benchmark_validation.exs`

```elixir
# Benchmark validation performance

require Logger

defmodule ValidationBenchmark do
  def run do
    Logger.info("=== Validation Performance Benchmark ===")

    # Setup
    predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
    instance = [5.0, 10.0]
    explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

    # Benchmark faithfulness
    {time_faith, _result} = :timer.tc(fn ->
      CrucibleXAI.Validation.Faithfulness.measure_faithfulness(
        instance,
        explanation,
        predict_fn
      )
    end)
    Logger.info("Faithfulness: #{time_faith / 1000}ms")

    # Benchmark infidelity
    {time_infid, _result} = :timer.tc(fn ->
      CrucibleXAI.Validation.Infidelity.compute(
        instance,
        explanation.feature_weights,
        predict_fn,
        num_perturbations: 100
      )
    end)
    Logger.info("Infidelity: #{time_infid / 1000}ms")

    # Benchmark quick validation
    {time_quick, _result} = :timer.tc(fn ->
      CrucibleXAI.Validation.quick_validation(explanation, instance, predict_fn)
    end)
    Logger.info("Quick Validation: #{time_quick / 1000}ms")

    Logger.info("=== Benchmark Complete ===")
  end
end

ValidationBenchmark.run()
```

Run with:
```bash
mix run scripts/benchmark_validation.exs
```

---

## Documentation Requirements

### Module Documentation

Each module must have:
1. @moduledoc with overview and examples
2. @doc for all public functions
3. @spec type specifications
4. Examples in doctests where applicable

### Examples Directory

Create working examples:

```bash
mkdir -p examples/validation
touch examples/validation/01_basic_validation.exs
touch examples/validation/02_compare_methods.exs
touch examples/validation/03_production_monitoring.exs
```

---

## Completion Checklist

### Week 1: Faithfulness
- [ ] 15 tests written and passing
- [ ] Implementation complete
- [ ] Documentation complete
- [ ] Examples created
- [ ] Performance benchmarked

### Week 2: Infidelity
- [ ] 12 tests written and passing
- [ ] Implementation complete
- [ ] Documentation complete
- [ ] Parallel processing optimized

### Week 3: Sensitivity
- [ ] 15 tests written and passing
- [ ] Implementation complete
- [ ] Adaptive sampling implemented
- [ ] Cross-method comparison working

### Week 4: Integration
- [ ] 13 axiom tests passing
- [ ] 8 integration tests passing
- [ ] 7 property-based tests passing
- [ ] Main API complete
- [ ] All 337 tests passing
- [ ] Zero warnings
- [ ] Coverage >96%
- [ ] Documentation 100%
- [ ] Examples working
- [ ] README updated
- [ ] CHANGELOG updated

---

## Success Criteria

✅ **All 60 new tests pass**
✅ **Total: 337 tests, 0 failures**
✅ **Zero compilation warnings**
✅ **Test coverage >96%**
✅ **Dialyzer: 0 errors**
✅ **Complete API documentation**
✅ **5+ working examples**
✅ **Performance within targets**:
   - Faithfulness: <100ms
   - Infidelity: <150ms
   - Quick validation: <200ms

---

## Notes for Implementer

1. **Follow TDD strictly**: RED → GREEN → REFACTOR
2. **Run tests frequently**: After each function implementation
3. **Commit often**: After each passing test or module completion
4. **Document as you go**: Don't leave documentation for the end
5. **Benchmark early**: Identify performance issues early
6. **Use property-based testing**: Catch edge cases automatically
7. **Handle errors gracefully**: Add comprehensive error handling
8. **Optimize incrementally**: Don't pre-optimize, refactor in REFACTOR phase

---

**Document Version**: 1.0
**Status**: Ready for Implementation
**Estimated Total Effort**: 4 weeks (160 hours)
**Prerequisites**: Elixir 1.14+, OTP 25+, mix test passing

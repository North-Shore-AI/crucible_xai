defmodule CrucibleXAI.SHAP.LinearSHAPTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.SHAP.LinearSHAP

  describe "explain/4" do
    test "computes exact SHAP values for simple linear model" do
      # Model: f(x) = 2*x1 + 3*x2 + 1 (intercept)
      # For linear models: φᵢ = wᵢ * (xᵢ - E[xᵢ])

      coefficients = %{0 => 2.0, 1 => 3.0}
      intercept = 1.0
      instance = [1.0, 1.0]
      # Mean: [1.0, 1.0]
      background = [[0.0, 0.0], [2.0, 2.0]]

      # Expected SHAP values:
      # φ₀ = 2.0 * (1.0 - 1.0) = 0.0
      # φ₁ = 3.0 * (1.0 - 1.0) = 0.0
      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      assert_in_delta shap_values[0], 0.0, 0.001
      assert_in_delta shap_values[1], 0.0, 0.001
    end

    test "computes SHAP values when instance differs from mean" do
      # Model: f(x) = 2*x1 + 3*x2
      coefficients = %{0 => 2.0, 1 => 3.0}
      intercept = 0.0
      instance = [5.0, 3.0]
      # Mean: [0.0, 0.0]
      background = [[0.0, 0.0]]

      # Expected SHAP values:
      # φ₀ = 2.0 * (5.0 - 0.0) = 10.0
      # φ₁ = 3.0 * (3.0 - 0.0) = 9.0
      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      assert_in_delta shap_values[0], 10.0, 0.001
      assert_in_delta shap_values[1], 9.0, 0.001
    end

    test "verifies additivity property" do
      # SHAP values should sum to (prediction - baseline)
      coefficients = %{0 => 2.0, 1 => 3.0}
      intercept = 1.0
      instance = [3.0, 2.0]
      # Mean: [1.0, 1.0]
      background = [[0.0, 0.0], [2.0, 2.0]]

      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      # Prediction: 2*3 + 3*2 + 1 = 13
      # Baseline: 2*1 + 3*1 + 1 = 6
      # SHAP sum should be: 13 - 6 = 7
      shap_sum = Enum.sum(Map.values(shap_values))
      expected_diff = 7.0

      assert_in_delta shap_sum, expected_diff, 0.001
    end

    test "handles single feature" do
      coefficients = %{0 => 5.0}
      intercept = 2.0
      instance = [3.0]
      # Mean: 2.0
      background = [[1.0], [2.0], [3.0]]

      # φ₀ = 5.0 * (3.0 - 2.0) = 5.0
      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      assert_in_delta shap_values[0], 5.0, 0.001
    end

    test "handles negative coefficients" do
      coefficients = %{0 => -2.0, 1 => 3.0}
      intercept = 0.0
      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      # φ₀ = -2.0 * (1.0 - 0.0) = -2.0
      # φ₁ = 3.0 * (1.0 - 0.0) = 3.0
      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      assert_in_delta shap_values[0], -2.0, 0.001
      assert_in_delta shap_values[1], 3.0, 0.001
    end

    test "handles zero coefficients (dummy features)" do
      coefficients = %{0 => 2.0, 1 => 0.0, 2 => 3.0}
      intercept = 1.0
      # Feature 1 value doesn't matter
      instance = [1.0, 5.0, 2.0]
      background = [[0.0, 0.0, 0.0]]

      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

      # Feature 1 has zero coefficient, so SHAP value should be 0
      assert_in_delta shap_values[1], 0.0, 0.001

      # Other features should have non-zero SHAP
      assert_in_delta shap_values[0], 2.0, 0.001
      assert_in_delta shap_values[2], 6.0, 0.001
    end
  end

  describe "calculate_feature_means/1" do
    test "computes correct means for background data" do
      background = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

      means = LinearSHAP.calculate_feature_means(background)

      # Mean of feature 0: (1+3+5)/3 = 3.0
      # Mean of feature 1: (2+4+6)/3 = 4.0
      assert_in_delta means[0], 3.0, 0.001
      assert_in_delta means[1], 4.0, 0.001
    end

    test "handles single background instance" do
      background = [[5.0, 10.0]]

      means = LinearSHAP.calculate_feature_means(background)

      assert_in_delta means[0], 5.0, 0.001
      assert_in_delta means[1], 10.0, 0.001
    end

    test "handles single feature" do
      background = [[1.0], [2.0], [3.0]]

      means = LinearSHAP.calculate_feature_means(background)

      assert_in_delta means[0], 2.0, 0.001
    end
  end

  describe "property-based tests" do
    property "SHAP values always satisfy additivity for linear models" do
      check all(
              num_features <- integer(1..5),
              # Generate random coefficients
              coeffs <- list_of(float(min: -10.0, max: 10.0), length: num_features),
              intercept <- float(min: -5.0, max: 5.0),
              # Generate instance
              instance <- list_of(float(min: -10.0, max: 10.0), length: num_features),
              # Generate background data (simple list, uniqueness not required)
              bg_size <- integer(1..5),
              background <-
                list_of(list_of(float(min: -10.0, max: 10.0), length: num_features),
                  length: bg_size
                )
            ) do
        # Convert coefficients list to map
        coefficients = coeffs |> Enum.with_index() |> Map.new(fn {v, i} -> {i, v} end)

        shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)

        # Compute prediction
        prediction =
          Enum.zip(instance, coeffs)
          |> Enum.map(fn {x, w} -> x * w end)
          |> Enum.sum()
          |> Kernel.+(intercept)

        # Compute baseline
        feature_means = LinearSHAP.calculate_feature_means(background)

        baseline =
          Enum.map(0..(num_features - 1), fn i ->
            Map.get(coefficients, i, 0.0) * Map.get(feature_means, i, 0.0)
          end)
          |> Enum.sum()
          |> Kernel.+(intercept)

        # SHAP values should sum to (prediction - baseline)
        shap_sum = Enum.sum(Map.values(shap_values))
        expected_diff = prediction - baseline

        assert_in_delta shap_sum, expected_diff, 0.01
      end
    end
  end
end

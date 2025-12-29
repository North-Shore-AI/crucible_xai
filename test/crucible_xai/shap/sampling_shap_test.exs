defmodule CrucibleXAI.SHAP.SamplingShapTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.SHAP.KernelSHAP
  alias CrucibleXAI.SHAP.SamplingShap

  describe "explain/4" do
    test "approximates SHAP values for simple linear model" do
      # Model: f(x) = 2*x1 + 3*x2
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end

      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      # With enough samples, should approximate SHAP values
      shap_values = SamplingShap.explain(instance, background, predict_fn, num_samples: 1000)

      assert is_map(shap_values)
      assert map_size(shap_values) == 2

      # Exact SHAP values are: φ₀ = 2.0, φ₁ = 3.0
      # Allow some tolerance for Monte Carlo approximation
      assert_in_delta shap_values[0], 2.0, 0.5
      assert_in_delta shap_values[1], 3.0, 0.5
    end

    test "satisfies additivity property approximately" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 1.0 end
      instance = [3.0, 2.0]
      background = [[0.0, 0.0], [1.0, 1.0]]

      shap_values = SamplingShap.explain(instance, background, predict_fn, num_samples: 2000)

      # Prediction: 2*3 + 3*2 + 1 = 13
      # Baseline: 2*0.5 + 3*0.5 + 1 = 3.5
      # Expected diff: 9.5
      prediction = predict_fn.(instance)
      baseline = (predict_fn.([0.0, 0.0]) + predict_fn.([1.0, 1.0])) / 2
      expected_diff = prediction - baseline

      shap_sum = Enum.sum(Map.values(shap_values))

      # Should be close due to sampling approximation
      assert_in_delta shap_sum, expected_diff, 1.0
    end

    test "handles single feature" do
      predict_fn = fn [x] -> x * 5.0 end
      instance = [3.0]
      background = [[1.0]]

      shap_values = SamplingShap.explain(instance, background, predict_fn, num_samples: 500)

      assert map_size(shap_values) == 1
      # φ₀ should be approximately 5.0 * (3.0 - 1.0) = 10.0
      assert_in_delta shap_values[0], 10.0, 1.0
    end

    test "handles multiple features" do
      predict_fn = fn inst -> Enum.sum(inst) * 2.0 end
      instance = [1.0, 2.0, 3.0, 4.0]
      background = [[0.0, 0.0, 0.0, 0.0]]

      shap_values = SamplingShap.explain(instance, background, predict_fn, num_samples: 1000)

      assert map_size(shap_values) == 4
      # All SHAP values should be approximately 2.0 * feature_value
      Enum.each(0..3, fn i ->
        expected = 2.0 * Enum.at(instance, i)
        assert_in_delta Map.get(shap_values, i), expected, 1.0
      end)
    end

    test "faster than KernelSHAP for similar accuracy" do
      predict_fn = fn [x, y, z] -> x + 2.0 * y + 3.0 * z end
      instance = [1.0, 2.0, 3.0]
      background = [[0.0, 0.0, 0.0]]

      # Time SamplingShap
      {time_sampling, _result} =
        :timer.tc(fn ->
          SamplingShap.explain(instance, background, predict_fn, num_samples: 500)
        end)

      # Time KernelSHAP
      {time_kernel, _result} =
        :timer.tc(fn ->
          KernelSHAP.explain(instance, background, predict_fn, num_samples: 500)
        end)

      # SamplingShap should be comparable or faster
      # Allow up to 2x slower (it's simpler but may be less optimized)
      assert time_sampling < time_kernel * 2
    end
  end

  describe "generate_permutation/1" do
    test "generates valid permutation of feature indices" do
      n_features = 5
      perm = SamplingShap.generate_permutation(n_features)

      assert length(perm) == n_features
      # Should contain all indices 0..4
      assert Enum.sort(perm) == Enum.to_list(0..4)
    end

    test "generates different permutations on multiple calls" do
      n_features = 4
      perms = Enum.map(1..10, fn _ -> SamplingShap.generate_permutation(n_features) end)

      # Should have at least some different permutations
      unique_perms = Enum.uniq(perms)
      assert length(unique_perms) > 1
    end
  end

  describe "marginal_contribution/5" do
    test "calculates contribution of adding a feature" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 3.0]
      background_mean = [0.0, 0.0]

      # Adding feature 0 to empty set
      contrib0 = SamplingShap.marginal_contribution(0, [], instance, background_mean, predict_fn)
      # f({0}) - f({}) = 2*5 - 0 = 10
      assert_in_delta contrib0, 10.0, 0.01

      # Adding feature 1 to set containing feature 0
      contrib1 =
        SamplingShap.marginal_contribution(1, [0], instance, background_mean, predict_fn)

      # f({0,1}) - f({0}) = (2*5 + 3*3) - 2*5 = 19 - 10 = 9
      assert_in_delta contrib1, 9.0, 0.01
    end
  end

  describe "property-based tests" do
    property "SHAP values approximately sum to prediction minus baseline" do
      check all(
              num_features <- integer(1..4),
              # Simple linear coefficients
              coeffs <- list_of(float(min: -5.0, max: 5.0), length: num_features),
              instance <- list_of(float(min: 0.0, max: 5.0), length: num_features),
              background <- list_of(float(min: 0.0, max: 5.0), length: num_features)
            ) do
        # Linear model
        predict_fn = fn inst ->
          Enum.zip(inst, coeffs)
          |> Enum.map(fn {x, w} -> x * w end)
          |> Enum.sum()
        end

        shap_values =
          SamplingShap.explain(instance, [background], predict_fn, num_samples: 1000)

        prediction = predict_fn.(instance)
        baseline = predict_fn.(background)
        expected_diff = prediction - baseline

        shap_sum = Enum.sum(Map.values(shap_values))

        # Allow larger tolerance for Monte Carlo approximation
        assert_in_delta shap_sum, expected_diff, 2.0
      end
    end
  end
end

defmodule CrucibleXAI.SHAP.KernelSHAPTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.SHAP.KernelSHAP

  describe "generate_coalitions/2" do
    test "generates correct number of coalitions" do
      n_features = 5
      n_samples = 100

      coalitions = KernelSHAP.generate_coalitions(n_features, n_samples)

      assert Nx.shape(coalitions) == {n_samples, n_features}
    end

    test "coalitions are binary (0 or 1)" do
      coalitions = KernelSHAP.generate_coalitions(3, 50)
      values = coalitions |> Nx.to_flat_list() |> Enum.uniq() |> Enum.sort()

      # Should only contain 0 and 1
      assert values == [0, 1] or values == [0] or values == [1]
    end

    test "includes all features present and all features absent" do
      n_features = 4
      coalitions = KernelSHAP.generate_coalitions(n_features, 100)

      coalitions_list = Nx.to_list(coalitions)

      # Should include coalition with all 1s
      all_present = List.duplicate(1, n_features)
      assert all_present in coalitions_list

      # Should include coalition with all 0s
      all_absent = List.duplicate(0, n_features)
      assert all_absent in coalitions_list
    end

    test "handles single feature" do
      coalitions = KernelSHAP.generate_coalitions(1, 10)

      assert Nx.shape(coalitions) == {10, 1}
    end
  end

  describe "calculate_shap_kernel_weights/2" do
    test "calculates correct SHAP kernel weights" do
      n_features = 3
      # Test with specific coalitions
      coalitions =
        Nx.tensor([
          [0, 0, 0],
          # |S| = 0
          [1, 0, 0],
          # |S| = 1
          [1, 1, 0],
          # |S| = 2
          [1, 1, 1]
          # |S| = 3
        ])

      weights = KernelSHAP.calculate_shap_kernel_weights(coalitions, n_features)

      assert Nx.shape(weights) == {4}

      # Weights for empty and full coalitions should be very large (infinity approximation)
      weights_list = Nx.to_flat_list(weights)
      assert Enum.at(weights_list, 0) > 1000.0
      assert Enum.at(weights_list, 3) > 1000.0

      # Middle coalitions should have finite weights
      assert Enum.at(weights_list, 1) > 0 and Enum.at(weights_list, 1) < 1000.0
      assert Enum.at(weights_list, 2) > 0 and Enum.at(weights_list, 2) < 1000.0
    end

    test "handles single feature case" do
      coalitions = Nx.tensor([[0], [1]])
      weights = KernelSHAP.calculate_shap_kernel_weights(coalitions, 1)

      assert Nx.shape(weights) == {2}
    end

    test "all weights are positive" do
      coalitions = KernelSHAP.generate_coalitions(5, 100)
      weights = KernelSHAP.calculate_shap_kernel_weights(coalitions, 5)

      weights_list = Nx.to_flat_list(weights)
      assert Enum.all?(weights_list, fn w -> w > 0 end)
    end
  end

  describe "explain/4" do
    test "explains simple linear model" do
      # Model: f(x) = 2*x1 + 3*x2
      predict_fn = fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 end

      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 1000)

      assert is_map(shap_values)
      assert map_size(shap_values) == 2

      # SHAP values should be close to model coefficients for linear model
      # Since baseline is [0, 0]: prediction = 5, baseline = 0
      # SHAP values should sum to 5
      shap_sum = Enum.sum(Map.values(shap_values))
      assert_in_delta shap_sum, 5.0, 0.5
    end

    test "SHAP values sum to prediction minus baseline" do
      predict_fn = fn [x, y] -> x * x + y * y end
      instance = [3.0, 4.0]
      background = [[0.0, 0.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 2000)

      prediction = predict_fn.(instance)
      baseline = predict_fn.(hd(background))
      shap_sum = Enum.sum(Map.values(shap_values))

      # Core SHAP property: sum(SHAP values) = prediction - baseline
      assert_in_delta shap_sum, prediction - baseline, 1.0
    end

    test "handles multiple background samples" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [2.0, 2.0]
      background = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 1000)

      assert is_map(shap_values)
      assert map_size(shap_values) == 2
    end

    test "satisfies dummy property - unused features get zero value" do
      # Model that only uses first feature
      predict_fn = fn [x, _y] -> 2.0 * x end
      instance = [3.0, 5.0]
      background = [[0.0, 0.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 2000)

      # Feature 1 (y) should have near-zero SHAP value since it doesn't affect output
      assert_in_delta shap_values[1], 0.0, 0.5
      # Feature 0 (x) should have non-zero value
      assert abs(shap_values[0]) > 1.0
    end

    test "satisfies symmetry property" do
      # Model where both features have identical effect
      predict_fn = fn [x, y] -> x + y end
      instance = [2.0, 2.0]
      background = [[0.0, 0.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 2000)

      # Both features should have similar SHAP values
      assert_in_delta shap_values[0], shap_values[1], 0.3
    end

    test "handles single feature" do
      predict_fn = fn [x] -> 2.0 * x end
      instance = [5.0]
      background = [[0.0]]

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 100)

      assert map_size(shap_values) == 1
      # Should equal prediction - baseline
      assert_in_delta shap_values[0], 10.0, 0.5
    end
  end

  # Property-based tests
  property "SHAP values always sum to prediction minus baseline" do
    check all(n_features <- integer(1..4)) do
      instance = for _ <- 1..n_features, do: :rand.uniform() * 10.0
      background = [List.duplicate(0.0, n_features)]
      predict_fn = fn inst -> Enum.sum(inst) end

      shap_values = KernelSHAP.explain(instance, background, predict_fn, num_samples: 500)

      prediction = predict_fn.(instance)
      baseline = predict_fn.(hd(background))
      shap_sum = Enum.sum(Map.values(shap_values))

      # Core SHAP property
      assert_in_delta shap_sum, prediction - baseline, 0.5
    end
  end

  property "coalition weights are always positive" do
    check all(
            n_features <- integer(2..6),
            n_samples <- integer(10..100)
          ) do
      coalitions = KernelSHAP.generate_coalitions(n_features, n_samples)
      weights = KernelSHAP.calculate_shap_kernel_weights(coalitions, n_features)

      weights_list = Nx.to_flat_list(weights)
      assert Enum.all?(weights_list, fn w -> w > 0 end)
    end
  end
end

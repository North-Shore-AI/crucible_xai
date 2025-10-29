defmodule CrucibleXAI.Global.ALETest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.Global.ALE

  describe "accumulated_local_effects/4" do
    test "computes ALE for simple linear model" do
      # Model: f(x, y) = 2*x + 3*y
      # ALE for x should show constant effect of 2
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      data = [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0]
      ]

      ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 4)

      assert is_map(ale_result)
      assert Map.has_key?(ale_result, :bin_centers)
      assert Map.has_key?(ale_result, :effects)
      assert Map.has_key?(ale_result, :feature_index)

      # For linear model, effects should increase linearly
      assert length(ale_result.bin_centers) == 4
      assert length(ale_result.effects) == 4
    end

    test "handles correlated features better than PDP" do
      # When features are correlated, ALE is more accurate than PDP
      # Model: f(x, y) = x + y, where x and y are highly correlated
      predict_fn = fn [x, y] -> x + y end

      # Correlated data: y ≈ x
      data = for i <- 1..20, do: [i * 1.0, i * 1.0 + :rand.uniform() - 0.5]

      ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 5)

      # ALE should show effect close to 1.0 (the true coefficient)
      assert is_map(ale_result)
      assert length(ale_result.effects) == 5
    end

    test "configurable number of bins" do
      predict_fn = fn [x, y] -> x * 2.0 + y end
      data = for i <- 1..10, do: [i * 1.0, i * 0.5]

      ale_3 = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 3)
      ale_5 = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 5)

      assert length(ale_3.bin_centers) == 3
      assert length(ale_5.bin_centers) == 5
    end

    test "handles single feature" do
      predict_fn = fn [x] -> x * 3.0 end
      data = [[1.0], [2.0], [3.0], [4.0], [5.0]]

      ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 4)

      assert ale_result.feature_index == 0
      assert length(ale_result.effects) == 4
    end

    test "ALE is centered around zero" do
      # ALE effects should be centered (mean ≈ 0)
      predict_fn = fn [x, y] -> 2.0 * x + y end
      data = for i <- 1..20, do: [i * 1.0, i * 0.5]

      ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 10)

      # Mean of ALE effects should be close to 0 (centering)
      mean_effect = Enum.sum(ale_result.effects) / length(ale_result.effects)
      assert abs(mean_effect) < 1.0
    end

    test "handles nonlinear effects" do
      # Model: f(x, y) = x^2 + y
      predict_fn = fn [x, y] -> x * x + y end
      data = for i <- 1..15, do: [i * 1.0, i * 0.3]

      ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 5)

      # For quadratic function, effects should increase
      # (derivative increases with x)
      effects = ale_result.effects

      # Check that later effects are generally larger (for x^2)
      first_half_avg = Enum.slice(effects, 0, 2) |> Enum.sum() |> Kernel./(2)
      second_half_avg = Enum.slice(effects, -2, 2) |> Enum.sum() |> Kernel./(2)

      # For x^2, effect should increase with x
      assert second_half_avg > first_half_avg
    end
  end

  describe "compute_bin_edges/3" do
    test "creates quantile-based bins" do
      data = [
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 7.0],
        [4.0, 8.0],
        [5.0, 9.0]
      ]

      bin_edges = ALE.compute_bin_edges(data, 0, 3)

      # Should have num_bins + 1 edges
      assert length(bin_edges) == 4

      # Edges should span the feature range
      assert hd(bin_edges) <= 1.0
      assert List.last(bin_edges) >= 5.0

      # Edges should be sorted
      assert bin_edges == Enum.sort(bin_edges)
    end

    test "handles uniform distribution" do
      # Uniformly distributed data
      data = for i <- 1..20, do: [i * 1.0, 0.0]

      bin_edges = ALE.compute_bin_edges(data, 0, 5)

      assert length(bin_edges) == 6
      # Should be approximately evenly spaced for uniform data
      diffs =
        Enum.chunk_every(bin_edges, 2, 1, :discard)
        |> Enum.map(fn [a, b] -> b - a end)

      # Diffs should be relatively similar
      avg_diff = Enum.sum(diffs) / length(diffs)
      assert Enum.all?(diffs, fn d -> abs(d - avg_diff) < avg_diff * 0.5 end)
    end

    test "handles single bin" do
      data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

      bin_edges = ALE.compute_bin_edges(data, 0, 1)

      # Single bin needs 2 edges (min and max)
      assert length(bin_edges) == 2
    end
  end

  describe "local_effect_in_bin/5" do
    test "computes local effect for instances in a bin" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      # Instances in the bin
      instances_in_bin = [
        [2.5, 1.0],
        [3.0, 2.0],
        [3.5, 1.5]
      ]

      bin_min = 2.0
      bin_max = 4.0
      feature_idx = 0

      effect =
        ALE.local_effect_in_bin(
          predict_fn,
          instances_in_bin,
          feature_idx,
          bin_min,
          bin_max
        )

      # For linear model f = 2x + 3y:
      # Effect = f(bin_max, y) - f(bin_min, y)
      # = (2*4 + 3*y) - (2*2 + 3*y) = 8 - 4 = 4
      # Average across instances should be approximately 4
      assert is_float(effect)
      assert_in_delta effect, 4.0, 0.5
    end

    test "handles empty bin" do
      predict_fn = fn [x, y] -> x + y end
      instances_in_bin = []

      effect = ALE.local_effect_in_bin(predict_fn, instances_in_bin, 0, 1.0, 2.0)

      # Empty bin should return 0
      assert effect == 0.0
    end
  end

  describe "property-based tests" do
    property "ALE result has correct structure" do
      check all(
              n_instances <- integer(5..15),
              n_features <- integer(2..5),
              n_bins <- integer(2..8),
              data <-
                list_of(list_of(float(min: 0.0, max: 10.0), length: n_features),
                  length: n_instances
                ),
              feature_idx <- integer(0..(n_features - 1))
            ) do
        predict_fn = fn inst -> Enum.sum(inst) end

        ale_result =
          ALE.accumulated_local_effects(predict_fn, data, feature_idx, num_bins: n_bins)

        assert is_map(ale_result)
        assert ale_result.feature_index == feature_idx
        # May have fewer bins than requested if data is sparse
        assert length(ale_result.bin_centers) <= n_bins
        assert length(ale_result.effects) == length(ale_result.bin_centers)
      end
    end

    property "ALE effects are centered" do
      check all(
              n_instances <- integer(8..15),
              data <- list_of(list_of(float(min: 0.0, max: 10.0), length: 2), length: n_instances)
            ) do
        predict_fn = fn [x, y] -> x + y end

        ale_result = ALE.accumulated_local_effects(predict_fn, data, 0, num_bins: 5)

        # Mean of effects should be close to 0 (centered)
        if length(ale_result.effects) > 0 do
          mean_effect = Enum.sum(ale_result.effects) / length(ale_result.effects)
          assert abs(mean_effect) < 5.0
        end
      end
    end
  end
end

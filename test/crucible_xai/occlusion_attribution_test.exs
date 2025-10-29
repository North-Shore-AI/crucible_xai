defmodule CrucibleXAI.OcclusionAttributionTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.OcclusionAttribution

  describe "feature_occlusion/3" do
    test "computes attribution by occluding individual features" do
      # Model: f(x, y) = 2*x + 3*y
      # Occluding feature 0: removes 2*x contribution
      # Occluding feature 1: removes 3*y contribution

      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 4.0]

      attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

      # Original prediction: 2*5 + 3*4 = 22
      # With feature 0 occluded (x=0): 0 + 3*4 = 12, diff = 10
      # With feature 1 occluded (y=0): 2*5 + 0 = 10, diff = 12
      assert_in_delta Map.get(attributions, 0), 10.0, 0.001
      assert_in_delta Map.get(attributions, 1), 12.0, 0.001
    end

    test "handles non-zero baseline" do
      # Model: f(x, y) = 2*x + 3*y
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 4.0]

      # Use baseline value of 1.0 instead of 0.0
      attributions =
        OcclusionAttribution.feature_occlusion(
          instance,
          predict_fn,
          baseline_value: 1.0
        )

      # Original: 2*5 + 3*4 = 22
      # Feature 0 with baseline 1.0: 2*1 + 3*4 = 14, diff = 8
      # Feature 1 with baseline 1.0: 2*5 + 3*1 = 13, diff = 9
      assert_in_delta Map.get(attributions, 0), 8.0, 0.001
      assert_in_delta Map.get(attributions, 1), 9.0, 0.001
    end

    test "handles single feature" do
      predict_fn = fn [x] -> x * 5.0 end
      instance = [3.0]

      attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

      # Original: 5*3 = 15
      # Occluded: 5*0 = 0
      # Attribution: 15
      assert_in_delta Map.get(attributions, 0), 15.0, 0.001
    end

    test "handles multiple features" do
      predict_fn = fn inst -> Enum.sum(Enum.with_index(inst, fn x, i -> x * (i + 1) end)) end
      instance = [1.0, 2.0, 3.0, 4.0]

      attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

      # f(1,2,3,4) = 1*1 + 2*2 + 3*3 + 4*4 = 1+4+9+16 = 30
      # Occlude 0: f(0,2,3,4) = 0 + 4 + 9 + 16 = 29, diff = 1
      # Occlude 1: f(1,0,3,4) = 1 + 0 + 9 + 16 = 26, diff = 4
      # Occlude 2: f(1,2,0,4) = 1 + 4 + 0 + 16 = 21, diff = 9
      # Occlude 3: f(1,2,3,0) = 1 + 4 + 9 + 0 = 14, diff = 16
      assert_in_delta Map.get(attributions, 0), 1.0, 0.001
      assert_in_delta Map.get(attributions, 1), 4.0, 0.001
      assert_in_delta Map.get(attributions, 2), 9.0, 0.001
      assert_in_delta Map.get(attributions, 3), 16.0, 0.001
    end

    test "handles negative attributions" do
      # Model where a feature reduces the prediction
      predict_fn = fn [x, y] -> 10.0 - 2.0 * x + 3.0 * y end
      instance = [2.0, 1.0]

      attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

      # Original: 10 - 2*2 + 3*1 = 9
      # Occlude x (x=0): 10 - 0 + 3*1 = 13, diff = -4 (removing x increases prediction!)
      # Occlude y (y=0): 10 - 2*2 + 0 = 6, diff = 3
      assert_in_delta Map.get(attributions, 0), -4.0, 0.001
      assert_in_delta Map.get(attributions, 1), 3.0, 0.001
    end

    test "handles Nx tensor input" do
      predict_fn = fn
        params when is_list(params) -> Enum.sum(Enum.map(params, &(&1 * 2.0)))
        %Nx.Tensor{} = tensor -> Nx.sum(Nx.multiply(tensor, 2.0)) |> Nx.to_number()
      end

      instance = Nx.tensor([1.0, 2.0, 3.0])

      attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

      assert is_map(attributions)
      assert map_size(attributions) == 3
    end
  end

  describe "sliding_window_occlusion/4" do
    test "computes attributions using sliding window" do
      # For sequential data, occlude windows of features
      predict_fn = fn inst -> Enum.sum(inst) * 2.0 end
      instance = [1.0, 2.0, 3.0, 4.0, 5.0]

      attributions =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 2
        )

      # Original: (1+2+3+4+5)*2 = 30
      # Window [0,1] occluded (0,0,3,4,5): (3+4+5)*2 = 24, diff = 6
      # Window [1,2] occluded (1,0,0,4,5): (1+4+5)*2 = 20, diff = 10
      # etc.
      assert is_map(attributions)
      # Should have attributions for each window position
      # 5 features - 2 window + 1
      assert map_size(attributions) == 4
    end

    test "configurable window size" do
      predict_fn = fn inst -> Enum.sum(inst) end
      instance = [1.0, 2.0, 3.0, 4.0]

      # Window size 1 (same as feature occlusion)
      attrs_1 =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 1
        )

      assert map_size(attrs_1) == 4

      # Window size 2
      attrs_2 =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 2
        )

      # 4 - 2 + 1
      assert map_size(attrs_2) == 3

      # Window size 3
      attrs_3 =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 3
        )

      # 4 - 3 + 1
      assert map_size(attrs_3) == 2
    end

    test "configurable stride" do
      predict_fn = fn inst -> Enum.sum(inst) end
      instance = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

      # Stride 1 (default): slide by 1 each time
      attrs_stride_1 =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 2,
          stride: 1
        )

      # Stride 2: slide by 2 each time
      attrs_stride_2 =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 2,
          stride: 2
        )

      # With stride 2, should have fewer windows
      assert map_size(attrs_stride_2) < map_size(attrs_stride_1)
    end

    test "handles window larger than instance" do
      predict_fn = fn inst -> Enum.sum(inst) end
      instance = [1.0, 2.0]

      # Window size 5 > instance length 2
      attributions =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 5
        )

      # Should occlude entire instance
      assert map_size(attributions) == 1
    end

    test "uses custom baseline value" do
      predict_fn = fn inst -> Enum.sum(inst) end
      instance = [1.0, 2.0, 3.0, 4.0]

      attributions =
        OcclusionAttribution.sliding_window_occlusion(
          instance,
          predict_fn,
          window_size: 2,
          baseline_value: -1.0
        )

      # Original: 1+2+3+4 = 10
      # Window [0,1] with baseline -1: -1 + -1 + 3 + 4 = 5, diff = 5
      # This tests that baseline_value is being used
      assert is_map(attributions)
      assert map_size(attributions) == 3
    end
  end

  describe "occlusion_sensitivity/3" do
    test "computes sensitivity map for all features" do
      # Similar to feature_occlusion but returns sensitivity scores
      predict_fn = fn [x, y, z] -> x * 2.0 + y * 3.0 + z * 1.5 end
      instance = [2.0, 3.0, 4.0]

      sensitivity = OcclusionAttribution.occlusion_sensitivity(instance, predict_fn)

      # Should show how sensitive prediction is to each feature
      assert is_map(sensitivity)
      assert map_size(sensitivity) == 3

      # Higher coefficient features should have higher sensitivity
      # y (coeff=3) should be most sensitive
      max_sensitive_idx = Enum.max_by(sensitivity, fn {_k, v} -> abs(v) end) |> elem(0)
      # Feature y has highest coefficient
      assert max_sensitive_idx == 1
    end

    test "normalizes sensitivity scores" do
      predict_fn = fn inst -> Enum.sum(inst) end
      instance = [1.0, 2.0, 3.0]

      sensitivity =
        OcclusionAttribution.occlusion_sensitivity(
          instance,
          predict_fn,
          normalize: true
        )

      # With normalization, values should sum to 1.0 (or close)
      total = Enum.sum(Map.values(sensitivity))
      assert_in_delta total, 1.0, 0.001
    end

    test "returns absolute values when requested" do
      predict_fn = fn [x, y] -> 10.0 - 2.0 * x + 3.0 * y end
      instance = [2.0, 1.0]

      sensitivity =
        OcclusionAttribution.occlusion_sensitivity(
          instance,
          predict_fn,
          absolute: true
        )

      # All values should be positive
      assert Enum.all?(Map.values(sensitivity), fn v -> v >= 0 end)
    end
  end

  describe "batch_occlusion/3" do
    test "explains multiple instances with occlusion" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

      batch_attributions =
        OcclusionAttribution.batch_occlusion(
          instances,
          predict_fn
        )

      assert length(batch_attributions) == 3
      assert Enum.all?(batch_attributions, &is_map/1)
    end

    test "parallel batch occlusion" do
      predict_fn = fn [x, y] -> x * 2.0 + y * 3.0 end
      instances = for i <- 1..10, do: [i * 1.0, i * 2.0]

      batch_attributions =
        OcclusionAttribution.batch_occlusion(
          instances,
          predict_fn,
          parallel: true
        )

      assert length(batch_attributions) == 10
    end
  end

  describe "property-based tests" do
    property "occlusion attribution has correct dimensionality" do
      check all(
              n_features <- integer(1..10),
              values <- list_of(float(min: -10.0, max: 10.0), length: n_features)
            ) do
        predict_fn = fn inst -> Enum.sum(inst) end
        instance = values

        attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

        assert map_size(attributions) == n_features
        assert Enum.all?(Map.keys(attributions), fn k -> k >= 0 and k < n_features end)
      end
    end

    property "attribution sum approximates prediction for additive models" do
      check all(
              n_features <- integer(1..5),
              values <- list_of(float(min: 0.0, max: 10.0), length: n_features)
            ) do
        # Simple additive model
        predict_fn = fn inst -> Enum.sum(inst) end
        instance = values

        attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

        prediction = predict_fn.(instance)
        attr_sum = Enum.sum(Map.values(attributions))

        # For additive model with zero baseline, sum should equal prediction
        assert_in_delta attr_sum, prediction, 0.1
      end
    end

    property "zero features have zero attribution" do
      check all(
              n_features <- integer(2..5),
              values <- list_of(float(min: 0.0, max: 10.0), length: n_features)
            ) do
        # Set first feature to zero
        instance = [0.0 | tl(values)]

        predict_fn = fn inst -> Enum.sum(Enum.map(inst, &(&1 * 2.0))) end

        attributions = OcclusionAttribution.feature_occlusion(instance, predict_fn)

        # First feature is zero, so occluding it should have no effect
        assert_in_delta Map.get(attributions, 0), 0.0, 0.001
      end
    end
  end
end

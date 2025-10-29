defmodule CrucibleXAI.Global.ICETest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.Global.ICE

  describe "ice_curves/4" do
    test "computes ICE curves for each instance" do
      # Model: f(x, y) = 2*x + 3*y
      # Each instance should have its own curve
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      data = [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0]
      ]

      ice_result = ICE.ice_curves(predict_fn, data, 0, num_grid_points: 5)

      assert is_map(ice_result)
      assert Map.has_key?(ice_result, :grid_values)
      assert Map.has_key?(ice_result, :curves)
      assert Map.has_key?(ice_result, :feature_index)

      # Should have one curve per instance
      assert length(ice_result.curves) == 3
      # Each curve should have num_grid_points predictions
      assert Enum.all?(ice_result.curves, fn curve -> length(curve) == 5 end)
    end

    test "ICE curves show individual heterogeneity" do
      # Model where effect depends on other features
      predict_fn = fn [x, y] -> if y > 5.0, do: x * 2.0, else: x * 0.5 end

      data = [
        # y=3 < 5, so slope should be 0.5
        [1.0, 3.0],
        # y=7 > 5, so slope should be 2.0
        [1.0, 7.0]
      ]

      ice_result = ICE.ice_curves(predict_fn, data, 0, num_grid_points: 10)

      # Two instances should have different slopes
      curve_1 = Enum.at(ice_result.curves, 0)
      curve_2 = Enum.at(ice_result.curves, 1)

      # Curves should be different (heterogeneity)
      assert curve_1 != curve_2
    end

    test "handles single instance" do
      predict_fn = fn [x] -> x * 3.0 end
      data = [[2.0]]

      ice_result = ICE.ice_curves(predict_fn, data, 0, num_grid_points: 5)

      assert length(ice_result.curves) == 1
      assert length(hd(ice_result.curves)) == 5
    end

    test "custom grid range" do
      predict_fn = fn [x, y] -> x + y end
      data = [[1.0, 1.0], [2.0, 2.0]]

      ice_result =
        ICE.ice_curves(
          predict_fn,
          data,
          0,
          num_grid_points: 5,
          grid_range: {0.0, 10.0}
        )

      # Grid should use custom range
      assert Enum.min(ice_result.grid_values) >= 0.0
      assert Enum.max(ice_result.grid_values) <= 10.0
    end

    test "curves parallel for additive features" do
      # For additive model, ICE curves should be parallel
      # (same slope, different intercepts)
      predict_fn = fn [x, y] -> 2.0 * x + y end

      data = [
        # Different y values create different intercepts
        [1.0, 1.0],
        # Different x values too for proper grid
        [2.0, 3.0],
        [3.0, 5.0]
      ]

      ice_result = ICE.ice_curves(predict_fn, data, 0, num_grid_points: 10)

      # For linear additive model, all curves should have same slope
      # Calculate slopes for each curve
      slopes =
        ice_result.curves
        |> Enum.map(fn curve ->
          # Slope = (last - first) / (grid_max - grid_min)
          first = hd(curve)
          last = List.last(curve)
          grid_first = hd(ice_result.grid_values)
          grid_last = List.last(ice_result.grid_values)
          (last - first) / (grid_last - grid_first)
        end)

      # All slopes should be approximately 2.0 (the coefficient of x)
      assert Enum.all?(slopes, fn slope -> abs(slope - 2.0) < 0.5 end)
    end
  end

  describe "centered_ice/1" do
    test "centers ICE curves by subtracting first value" do
      ice_result = %{
        grid_values: [0.0, 1.0, 2.0],
        curves: [
          [10.0, 12.0, 14.0],
          [5.0, 7.0, 9.0],
          [8.0, 10.0, 12.0]
        ],
        feature_index: 0
      }

      centered = ICE.centered_ice(ice_result)

      # Each curve should start at 0.0
      assert Enum.all?(centered.curves, fn curve -> hd(curve) == 0.0 end)

      # First curve: [10-10, 12-10, 14-10] = [0, 2, 4]
      assert Enum.at(centered.curves, 0) == [0.0, 2.0, 4.0]
      # Second curve: [5-5, 7-5, 9-5] = [0, 2, 4]
      assert Enum.at(centered.curves, 1) == [0.0, 2.0, 4.0]
      # Third curve: [8-8, 10-8, 12-8] = [0, 2, 4]
      assert Enum.at(centered.curves, 2) == [0.0, 2.0, 4.0]
    end

    test "preserves grid values" do
      ice_result = %{
        grid_values: [1.0, 2.0, 3.0],
        curves: [[5.0, 10.0, 15.0]],
        feature_index: 1
      }

      centered = ICE.centered_ice(ice_result)

      assert centered.grid_values == ice_result.grid_values
      assert centered.feature_index == ice_result.feature_index
    end

    test "handles empty curves" do
      ice_result = %{
        grid_values: [1.0, 2.0],
        curves: [],
        feature_index: 0
      }

      centered = ICE.centered_ice(ice_result)

      assert centered.curves == []
    end
  end

  describe "average_ice_curves/1" do
    test "averages ICE curves to produce PDP" do
      ice_result = %{
        grid_values: [0.0, 1.0, 2.0],
        curves: [
          [10.0, 12.0, 14.0],
          [5.0, 7.0, 9.0],
          [8.0, 10.0, 12.0]
        ],
        feature_index: 0
      }

      pdp = ICE.average_ice_curves(ice_result)

      # Average of first point: (10+5+8)/3 = 7.67
      # Average of second point: (12+7+10)/3 = 9.67
      # Average of third point: (14+9+12)/3 = 11.67
      assert_in_delta Enum.at(pdp, 0), 7.67, 0.01
      assert_in_delta Enum.at(pdp, 1), 9.67, 0.01
      assert_in_delta Enum.at(pdp, 2), 11.67, 0.01
    end

    test "handles single curve" do
      ice_result = %{
        grid_values: [1.0, 2.0],
        curves: [[5.0, 10.0]],
        feature_index: 0
      }

      pdp = ICE.average_ice_curves(ice_result)

      # Average of single curve is the curve itself
      assert pdp == [5.0, 10.0]
    end
  end

  describe "property-based tests" do
    property "ICE has one curve per instance" do
      check all(
              n_instances <- integer(1..10),
              n_features <- integer(1..5),
              n_grid <- integer(3..10),
              data <-
                list_of(list_of(float(min: 0.0, max: 10.0), length: n_features),
                  length: n_instances
                ),
              feature_idx <- integer(0..(n_features - 1))
            ) do
        predict_fn = fn inst -> Enum.sum(inst) end

        ice_result = ICE.ice_curves(predict_fn, data, feature_idx, num_grid_points: n_grid)

        assert length(ice_result.curves) == n_instances
        assert Enum.all?(ice_result.curves, fn curve -> length(curve) == n_grid end)
      end
    end

    property "averaged ICE equals PDP" do
      check all(
              n_instances <- integer(2..8),
              data <- list_of(list_of(float(min: 0.0, max: 10.0), length: 2), length: n_instances)
            ) do
        predict_fn = fn [x, y] -> x + y end

        # Compute ICE
        ice_result = ICE.ice_curves(predict_fn, data, 0, num_grid_points: 5)

        # Compute PDP
        pdp_result =
          CrucibleXAI.Global.PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 5)

        # Average of ICE curves should equal PDP
        avg_ice = ICE.average_ice_curves(ice_result)

        # Should be approximately equal
        Enum.zip(avg_ice, pdp_result.predictions)
        |> Enum.each(fn {ice_avg, pdp_val} ->
          assert_in_delta ice_avg, pdp_val, 0.01
        end)
      end
    end
  end
end

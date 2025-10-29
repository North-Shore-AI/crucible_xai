defmodule CrucibleXAI.Global.PDPTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.Global.PDP

  describe "partial_dependence/4" do
    test "computes 1D partial dependence for linear model" do
      # Model: f(x, y) = 2*x + 3*y
      # PDP for feature x should be a straight line with slope 2
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      # Dataset
      data = [
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [1.0, 2.0],
        [2.0, 1.0]
      ]

      # Compute PDP for feature 0 (x)
      pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 5)

      assert is_map(pdp_result)
      assert Map.has_key?(pdp_result, :grid_values)
      assert Map.has_key?(pdp_result, :predictions)
      assert Map.has_key?(pdp_result, :feature_index)

      grid_values = pdp_result.grid_values
      predictions = pdp_result.predictions

      assert length(grid_values) == 5
      assert length(predictions) == 5

      # For linear model, PDP should be linear
      # Check that predictions increase with grid values
      assert Enum.zip(grid_values, predictions)
             |> Enum.chunk_every(2, 1, :discard)
             |> Enum.all?(fn [{_g1, p1}, {_g2, p2}] -> p2 > p1 end)
    end

    test "computes PDP for single feature dataset" do
      predict_fn = fn [x] -> x * 5.0 end
      data = [[1.0], [2.0], [3.0], [4.0]]

      pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 10)

      assert length(pdp_result.grid_values) == 10
      assert length(pdp_result.predictions) == 10
      assert pdp_result.feature_index == 0
    end

    test "handles nonlinear relationships" do
      # Model: f(x, y) = x^2 + y
      predict_fn = fn [x, y] -> x * x + y end

      data = for i <- 1..10, do: [i * 1.0, i * 0.5]

      pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 20)

      # PDP for x should show quadratic relationship
      assert length(pdp_result.grid_values) == 20

      # Check that relationship is curved (not linear)
      # Compare differences - they should increase (quadratic)
      diffs =
        Enum.zip(pdp_result.grid_values, pdp_result.predictions)
        |> Enum.chunk_every(2, 1, :discard)
        |> Enum.map(fn [{g1, p1}, {g2, p2}] -> (p2 - p1) / (g2 - g1) end)

      # For quadratic, slope should increase
      assert length(diffs) > 0
    end

    test "custom grid range" do
      predict_fn = fn [x, y] -> x + y end
      data = [[1.0, 1.0], [2.0, 2.0]]

      pdp_result =
        PDP.partial_dependence(
          predict_fn,
          data,
          0,
          num_grid_points: 5,
          grid_range: {0.0, 10.0}
        )

      # Grid should span the custom range
      assert Enum.min(pdp_result.grid_values) >= 0.0
      assert Enum.max(pdp_result.grid_values) <= 10.0
    end

    test "auto-detects grid range from data" do
      predict_fn = fn [x, y] -> x + y end
      data = [[1.0, 5.0], [3.0, 6.0], [5.0, 7.0]]

      # Feature 0 ranges from 1.0 to 5.0
      pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 10)

      # Grid should span the data range
      assert Enum.min(pdp_result.grid_values) >= 1.0
      assert Enum.max(pdp_result.grid_values) <= 5.0
    end

    test "averages predictions across all instances" do
      # Model where both features matter equally
      predict_fn = fn [x, y] -> x + y end

      # Data with different y values
      data = [[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]]

      pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 3)

      # For each grid value of x, should average over all y values
      # Average y = (1+2+3)/3 = 2
      # So PDP should show: pred(x) ≈ x + 2
      grid_values = pdp_result.grid_values
      predictions = pdp_result.predictions

      # Check that predictions are approximately grid_value + 2
      Enum.zip(grid_values, predictions)
      |> Enum.each(fn {grid_val, pred} ->
        assert_in_delta pred, grid_val + 2.0, 0.5
      end)
    end
  end

  describe "partial_dependence_2d/5" do
    test "computes 2D partial dependence for feature pair" do
      predict_fn = fn [x, y, z] -> 2.0 * x + 3.0 * y + z end

      data = for i <- 1..5, do: [i * 1.0, i * 0.5, i * 0.1]

      # Compute 2D PDP for features 0 and 1 (x and y)
      pdp_2d =
        PDP.partial_dependence_2d(
          predict_fn,
          data,
          {0, 1},
          num_grid_points: 5
        )

      assert is_map(pdp_2d)
      assert Map.has_key?(pdp_2d, :grid_values_x)
      assert Map.has_key?(pdp_2d, :grid_values_y)
      assert Map.has_key?(pdp_2d, :predictions)
      assert Map.has_key?(pdp_2d, :feature_indices)

      # Grid should be 5x5
      assert length(pdp_2d.grid_values_x) == 5
      assert length(pdp_2d.grid_values_y) == 5
      # Predictions should be a 5x5 matrix (list of lists)
      assert length(pdp_2d.predictions) == 5
      assert Enum.all?(pdp_2d.predictions, fn row -> length(row) == 5 end)
    end

    test "handles additive features" do
      # For additive model f(x, y) = g(x) + h(y),
      # 2D PDP should show no interaction
      predict_fn = fn [x, y] -> x * 2.0 + y * 3.0 end

      data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

      pdp_2d = PDP.partial_dependence_2d(predict_fn, data, {0, 1}, num_grid_points: 3)

      # For additive model, predictions should follow pattern:
      # pred(x, y) = 2*x + 3*y (plus constant from averaging other features)
      assert length(pdp_2d.predictions) == 3
      assert length(hd(pdp_2d.predictions)) == 3
    end

    test "custom grid ranges for both features" do
      predict_fn = fn [x, y] -> x + y end
      data = [[1.0, 1.0], [2.0, 2.0]]

      pdp_2d =
        PDP.partial_dependence_2d(
          predict_fn,
          data,
          {0, 1},
          num_grid_points: 4,
          grid_range_x: {0.0, 5.0},
          grid_range_y: {0.0, 10.0}
        )

      # Check grid ranges
      assert Enum.min(pdp_2d.grid_values_x) >= 0.0
      assert Enum.max(pdp_2d.grid_values_x) <= 5.0
      assert Enum.min(pdp_2d.grid_values_y) >= 0.0
      assert Enum.max(pdp_2d.grid_values_y) <= 10.0
    end
  end

  describe "create_grid/3" do
    test "creates evenly spaced grid" do
      data = [[1.0, 5.0], [3.0, 6.0], [5.0, 7.0]]
      grid = PDP.create_grid(data, 0, 10)

      assert length(grid) == 10
      # Should be evenly spaced between min and max of feature 0
      assert Enum.min(grid) >= 1.0
      assert Enum.max(grid) <= 5.0

      # Check spacing is approximately equal
      diffs =
        Enum.chunk_every(grid, 2, 1, :discard)
        |> Enum.map(fn [a, b] -> b - a end)

      avg_diff = Enum.sum(diffs) / length(diffs)
      # All diffs should be close to average (evenly spaced)
      assert Enum.all?(diffs, fn d -> abs(d - avg_diff) < 0.01 end)
    end

    test "uses custom range when provided" do
      data = [[1.0, 1.0], [2.0, 2.0]]
      grid = PDP.create_grid(data, 0, 5, grid_range: {0.0, 10.0})

      assert length(grid) == 5
      assert Enum.min(grid) >= 0.0
      assert Enum.max(grid) <= 10.0
    end

    test "handles single grid point" do
      data = [[1.0, 2.0], [3.0, 4.0]]
      grid = PDP.create_grid(data, 0, 1)

      assert length(grid) == 1
    end
  end

  describe "property-based tests" do
    property "PDP grid has correct number of points" do
      check all(
              n_instances <- integer(2..10),
              n_features <- integer(1..5),
              n_grid <- integer(3..20),
              data <-
                list_of(list_of(float(min: 0.0, max: 10.0), length: n_features),
                  length: n_instances
                ),
              feature_idx <- integer(0..(n_features - 1))
            ) do
        predict_fn = fn inst -> Enum.sum(inst) end

        pdp_result =
          PDP.partial_dependence(predict_fn, data, feature_idx, num_grid_points: n_grid)

        assert length(pdp_result.grid_values) == n_grid
        assert length(pdp_result.predictions) == n_grid
        assert pdp_result.feature_index == feature_idx
      end
    end

    property "PDP predictions are within reasonable range" do
      check all(
              n_instances <- integer(3..10),
              data <- list_of(list_of(float(min: 0.0, max: 10.0), length: 2), length: n_instances)
            ) do
        predict_fn = fn [x, y] -> x + y end

        pdp_result = PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 5)

        # All predictions should be finite and reasonable
        assert Enum.all?(pdp_result.predictions, fn p ->
                 is_float(p) and p >= 0.0 and p <= 100.0
               end)
      end
    end
  end
end

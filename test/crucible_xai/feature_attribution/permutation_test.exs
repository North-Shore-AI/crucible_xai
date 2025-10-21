defmodule CrucibleXAI.FeatureAttribution.PermutationTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.FeatureAttribution.Permutation

  describe "calculate/3" do
    test "calculates permutation importance for simple model" do
      # Simple model: f([x, y]) = 2*x + 3*y
      # Feature y should be more important (coefficient 3 vs 2)
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      validation_data = [
        {[1.0, 1.0], 5.0},
        {[2.0, 2.0], 10.0},
        {[3.0, 3.0], 15.0},
        {[1.0, 2.0], 8.0},
        {[2.0, 1.0], 7.0}
      ]

      importance =
        Permutation.calculate(predict_fn, validation_data, metric: :mse, num_repeats: 5)

      assert is_map(importance)
      assert map_size(importance) == 2

      # Both features should have positive importance
      assert importance[0][:importance] > 0
      assert importance[1][:importance] > 0

      # Feature 1 (y) should be more important than feature 0 (x)
      assert importance[1][:importance] > importance[0][:importance]
    end

    test "includes standard deviation in results" do
      predict_fn = fn [x] -> x * 2.0 end
      validation_data = [{[1.0], 2.0}, {[2.0], 4.0}, {[3.0], 6.0}]

      importance = Permutation.calculate(predict_fn, validation_data, num_repeats: 3)

      assert Map.has_key?(importance[0], :std_dev)
      assert is_float(importance[0][:std_dev])
    end

    test "handles different metrics" do
      predict_fn = fn [x, y] -> if x + y > 3, do: 1.0, else: 0.0 end

      validation_data = [
        {[1.0, 1.0], 0.0},
        {[2.0, 2.0], 1.0},
        {[3.0, 3.0], 1.0}
      ]

      # MSE
      imp_mse = Permutation.calculate(predict_fn, validation_data, metric: :mse, num_repeats: 3)
      assert is_map(imp_mse)

      # MAE
      imp_mae = Permutation.calculate(predict_fn, validation_data, metric: :mae, num_repeats: 3)
      assert is_map(imp_mae)

      # Accuracy (classification)
      imp_acc =
        Permutation.calculate(predict_fn, validation_data, metric: :accuracy, num_repeats: 3)

      assert is_map(imp_acc)
    end

    test "handles single feature" do
      predict_fn = fn [x] -> x * 3.0 end
      validation_data = [{[1.0], 3.0}, {[2.0], 6.0}, {[3.0], 9.0}, {[4.0], 12.0}]

      importance = Permutation.calculate(predict_fn, validation_data, num_repeats: 3)

      assert map_size(importance) == 1
      # Should have positive importance since feature is used
      assert importance[0][:importance] >= 0
    end

    test "returns zero importance for irrelevant features" do
      # Model only uses first feature
      predict_fn = fn [x, _y] -> x * 2.0 end

      validation_data = [
        {[1.0, 100.0], 2.0},
        {[2.0, 200.0], 4.0},
        {[3.0, 300.0], 6.0}
      ]

      importance =
        Permutation.calculate(predict_fn, validation_data, metric: :mse, num_repeats: 5)

      # Feature 0 should have high importance
      assert importance[0][:importance] > 0.5

      # Feature 1 should have near-zero importance
      assert importance[1][:importance] < 0.1
    end
  end

  describe "top_k/2" do
    test "returns top k features by importance" do
      importance = %{
        0 => %{importance: 0.5, std_dev: 0.1},
        1 => %{importance: 0.8, std_dev: 0.2},
        2 => %{importance: 0.3, std_dev: 0.05}
      }

      top = Permutation.top_k(importance, 2)

      assert length(top) == 2
      assert Enum.at(top, 0) == {1, %{importance: 0.8, std_dev: 0.2}}
      assert Enum.at(top, 1) == {0, %{importance: 0.5, std_dev: 0.1}}
    end

    test "returns all features if k >= number of features" do
      importance = %{
        0 => %{importance: 0.5, std_dev: 0.1},
        1 => %{importance: 0.3, std_dev: 0.05}
      }

      top = Permutation.top_k(importance, 10)

      assert length(top) == 2
    end
  end

  # Property-based tests
  property "importance values are non-negative" do
    check all(n_features <- integer(1..4)) do
      predict_fn = fn inst -> Enum.sum(inst) end

      validation_data =
        for _ <- 1..10 do
          instance = for _ <- 1..n_features, do: :rand.uniform() * 10.0
          label = predict_fn.(instance)
          {instance, label}
        end

      importance = Permutation.calculate(predict_fn, validation_data, num_repeats: 2)

      # All importance values should be >= 0
      assert Enum.all?(importance, fn {_idx, stats} ->
               stats[:importance] >= 0
             end)
    end
  end

  property "permutation preserves feature count" do
    check all(n_features <- integer(1..5)) do
      predict_fn = fn inst -> Enum.sum(inst) * 2.0 end

      validation_data =
        for _ <- 1..5 do
          instance = for _ <- 1..n_features, do: :rand.uniform()
          {instance, predict_fn.(instance)}
        end

      importance = Permutation.calculate(predict_fn, validation_data, num_repeats: 2)

      assert map_size(importance) == n_features
    end
  end
end

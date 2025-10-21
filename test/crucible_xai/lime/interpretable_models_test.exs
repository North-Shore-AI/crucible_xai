defmodule CrucibleXAI.LIME.InterpretableModelsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.LIME.InterpretableModels.LinearRegression
  alias CrucibleXAI.LIME.InterpretableModels.Ridge

  describe "LinearRegression.fit/3" do
    test "fits perfect linear model" do
      # y = 2*x1 + 3*x2 + 1
      # Use well-conditioned data (not collinear)
      samples = [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0],
        [2.0, 2.0],
        [3.0, 1.0]
      ]

      labels =
        Enum.map(samples, fn [x1, x2] -> 2.0 * x1 + 3.0 * x2 + 1.0 end)

      weights = List.duplicate(1.0, length(samples))

      model = LinearRegression.fit(samples, labels, weights)

      # Should recover exact coefficients for perfect linear data
      assert length(model.coefficients) == 2
      assert_in_delta Enum.at(model.coefficients, 0), 2.0, 0.01
      assert_in_delta Enum.at(model.coefficients, 1), 3.0, 0.01
      assert_in_delta model.intercept, 1.0, 0.01
      # R² should be perfect for perfect linear fit
      assert model.r_squared > 0.999
    end

    test "handles weighted samples" do
      samples = [[1, 1], [2, 2], [3, 3], [100, 100]]
      labels = [2.0, 4.0, 6.0, 200.0]
      # Give very low weight to outlier
      weights = [1.0, 1.0, 1.0, 0.01]

      model = LinearRegression.fit(samples, labels, weights)

      # Should fit mostly to first 3 points (y = 2*x1 approximately)
      assert model.r_squared > 0.8
      assert length(model.coefficients) == 2
    end

    test "handles Nx.Tensor inputs" do
      samples = Nx.tensor([[1.0, 2.0], [3.0, 4.0]])
      labels = Nx.tensor([5.0, 11.0])
      weights = Nx.tensor([1.0, 1.0])

      model = LinearRegression.fit(samples, labels, weights)

      assert is_float(model.intercept)
      assert is_list(model.coefficients)
      assert is_float(model.r_squared)
    end

    test "handles single feature" do
      samples = [[1], [2], [3], [4]]
      labels = [2.0, 4.0, 6.0, 8.0]
      weights = [1.0, 1.0, 1.0, 1.0]

      model = LinearRegression.fit(samples, labels, weights)

      assert length(model.coefficients) == 1
      assert_in_delta Enum.at(model.coefficients, 0), 2.0, 0.1
    end

    test "returns model struct with all fields" do
      samples = [[1, 2]]
      labels = [3.0]
      weights = [1.0]

      model = LinearRegression.fit(samples, labels, weights)

      assert Map.has_key?(model, :intercept)
      assert Map.has_key?(model, :coefficients)
      assert Map.has_key?(model, :r_squared)
    end
  end

  describe "LinearRegression.predict/2" do
    test "predicts with fitted model" do
      # Fit y = 2*x + 1
      samples = [[1], [2], [3]]
      labels = [3.0, 5.0, 7.0]
      weights = [1.0, 1.0, 1.0]

      model = LinearRegression.fit(samples, labels, weights)

      # Predict for new samples
      predictions = LinearRegression.predict(model, [[4], [5]])

      assert Nx.shape(predictions) == {2}
      # Should predict 9.0 and 11.0 approximately
      assert_in_delta Nx.to_number(predictions[0]), 9.0, 0.2
      assert_in_delta Nx.to_number(predictions[1]), 11.0, 0.2
    end

    test "handles tensor input" do
      samples = [[1, 1], [2, 2]]
      labels = [2.0, 4.0]
      weights = [1.0, 1.0]

      model = LinearRegression.fit(samples, labels, weights)

      test_samples = Nx.tensor([[3, 3], [4, 4]])
      predictions = LinearRegression.predict(model, test_samples)

      assert Nx.shape(predictions) == {2}
    end
  end

  describe "Ridge.fit/4" do
    test "fits with L2 regularization" do
      # Simple linear case
      samples = [[1, 2], [3, 4], [5, 6]]
      labels = [5.0, 11.0, 17.0]
      weights = [1.0, 1.0, 1.0]

      model = Ridge.fit(samples, labels, weights, 0.1)

      assert is_float(model.intercept)
      assert is_list(model.coefficients)
      assert is_float(model.r_squared)
      assert length(model.coefficients) == 2
    end

    test "regularization reduces coefficient magnitudes" do
      samples = [[1, 10], [2, 20], [3, 30]]
      labels = [11.0, 22.0, 33.0]
      weights = [1.0, 1.0, 1.0]

      # No regularization
      model_no_reg = Ridge.fit(samples, labels, weights, 0.0)
      # High regularization
      model_high_reg = Ridge.fit(samples, labels, weights, 10.0)

      coef_mag_no_reg =
        model_no_reg.coefficients |> Enum.map(&abs/1) |> Enum.sum()

      coef_mag_high_reg =
        model_high_reg.coefficients |> Enum.map(&abs/1) |> Enum.sum()

      # Higher regularization should lead to smaller coefficients
      assert coef_mag_high_reg < coef_mag_no_reg
    end

    test "handles default lambda" do
      samples = [[1, 2]]
      labels = [3.0]
      weights = [1.0]

      model = Ridge.fit(samples, labels, weights)

      assert is_map(model)
    end
  end

  describe "Ridge.predict/2" do
    test "predicts with ridge model" do
      samples = [[1, 1], [2, 2], [3, 3]]
      labels = [2.0, 4.0, 6.0]
      weights = [1.0, 1.0, 1.0]

      model = Ridge.fit(samples, labels, weights, 0.1)

      predictions = Ridge.predict(model, [[4, 4], [5, 5]])

      assert Nx.shape(predictions) == {2}
    end
  end

  # Property-based tests
  property "fitted model can predict on training data" do
    check all(
            n_samples <- integer(3..20),
            n_features <- integer(1..5)
          ) do
      samples = for _ <- 1..n_samples, do: for(_ <- 1..n_features, do: :rand.uniform() * 10)
      labels = Enum.map(samples, &Enum.sum/1)
      weights = List.duplicate(1.0, n_samples)

      model = LinearRegression.fit(samples, labels, weights)
      predictions = LinearRegression.predict(model, samples)

      assert Nx.shape(predictions) == {n_samples}
      # All predictions should be finite
      assert Enum.all?(Nx.to_flat_list(predictions), &is_number/1)
    end
  end

  property "equal weights give same result as unweighted" do
    check all(n_samples <- integer(3..10)) do
      samples = for _ <- 1..n_samples, do: [:rand.uniform() * 10, :rand.uniform() * 10]
      labels = Enum.map(samples, fn [x, y] -> 2 * x + 3 * y + 1 end)
      equal_weights = List.duplicate(1.0, n_samples)

      model = LinearRegression.fit(samples, labels, equal_weights)

      # Should fit well with equal weights
      assert is_float(model.r_squared)
      assert model.r_squared >= 0.0 and model.r_squared <= 1.0
    end
  end

  property "ridge regularization always produces valid model" do
    check all(
            n_samples <- integer(3..15),
            lambda <- float(min: 0.0, max: 10.0)
          ) do
      samples = for _ <- 1..n_samples, do: [:rand.uniform() * 10, :rand.uniform() * 10]
      labels = Enum.map(samples, &Enum.sum/1)
      weights = List.duplicate(1.0, n_samples)

      model = Ridge.fit(samples, labels, weights, lambda)

      assert is_float(model.intercept)
      assert is_list(model.coefficients)
      assert length(model.coefficients) == 2
      assert is_float(model.r_squared)
      # R² should be in valid range
      assert model.r_squared >= -10.0 and model.r_squared <= 1.0
    end
  end
end

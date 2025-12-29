defmodule CrucibleXAI.LIMETest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.Explanation
  alias CrucibleXAI.LIME

  describe "explain/3" do
    test "explains simple linear model" do
      # Model: f(x) = 2*x1 + 3*x2 + 1
      predict_fn = fn instance ->
        [x1, x2] = instance
        2.0 * x1 + 3.0 * x2 + 1.0
      end

      instance = [1.0, 2.0]

      explanation = LIME.explain(instance, predict_fn, num_samples: 100)

      assert %Explanation{} = explanation
      assert explanation.instance == instance
      assert explanation.method == :lime
      assert is_map(explanation.feature_weights)
      assert is_float(explanation.intercept)
      assert is_float(explanation.score)
      # Should have reasonable R² for local linear approximation
      assert explanation.score > 0.8
    end

    test "handles different number of features" do
      predict_fn = fn instance -> Enum.sum(instance) end

      # 1 feature
      exp1 = LIME.explain([5.0], predict_fn, num_samples: 50)
      assert map_size(exp1.feature_weights) <= 1

      # 3 features
      exp3 = LIME.explain([1.0, 2.0, 3.0], predict_fn, num_samples: 50)
      assert map_size(exp3.feature_weights) <= 3

      # 5 features
      exp5 = LIME.explain([1.0, 2.0, 3.0, 4.0, 5.0], predict_fn, num_samples: 50)
      assert map_size(exp5.feature_weights) <= 5
    end

    test "respects num_features option" do
      predict_fn = fn instance -> Enum.sum(instance) end
      instance = [1.0, 2.0, 3.0, 4.0, 5.0]

      explanation = LIME.explain(instance, predict_fn, num_features: 2, num_samples: 100)

      # Should select only 2 features
      assert map_size(explanation.feature_weights) == 2
    end

    test "uses specified kernel" do
      predict_fn = fn [x] -> x * 2.0 end
      instance = [5.0]

      # Should work with different kernels
      exp_exp = LIME.explain(instance, predict_fn, kernel: :exponential, num_samples: 50)
      exp_cos = LIME.explain(instance, predict_fn, kernel: :cosine, num_samples: 50)

      assert exp_exp.method == :lime
      assert exp_cos.method == :lime
    end

    test "uses specified feature selection method" do
      predict_fn = fn instance -> Enum.sum(instance) end
      instance = [1.0, 2.0, 3.0]

      exp_hw =
        LIME.explain(instance, predict_fn, feature_selection: :highest_weights, num_samples: 50)

      exp_fs =
        LIME.explain(instance, predict_fn, feature_selection: :forward_selection, num_samples: 50)

      assert exp_hw.method == :lime
      assert exp_fs.method == :lime
    end

    test "includes metadata" do
      predict_fn = fn [x] -> x * 2.0 end
      instance = [5.0]

      explanation = LIME.explain(instance, predict_fn, num_samples: 200)

      assert is_map(explanation.metadata)
      assert explanation.metadata.num_samples == 200
      assert Map.has_key?(explanation.metadata, :kernel)
      assert Map.has_key?(explanation.metadata, :feature_selection)
    end

    test "handles nonlinear models locally" do
      # Nonlinear model: f(x) = x²
      predict_fn = fn [x] -> x * x end
      instance = [3.0]

      explanation = LIME.explain(instance, predict_fn, num_samples: 200)

      assert %Explanation{} = explanation
      # Local linear approximation should still work
      assert is_float(explanation.score)
      # Might not be perfect since model is nonlinear
      assert explanation.score > 0.5
    end

    test "works with classification predictions" do
      # Binary classifier: predict class 1 if sum > 5, else class 0
      predict_fn = fn instance ->
        if Enum.sum(instance) > 5.0, do: 1.0, else: 0.0
      end

      # Sum = 6, predicts class 1
      instance = [3.0, 3.0]

      explanation = LIME.explain(instance, predict_fn, num_samples: 100)

      assert %Explanation{} = explanation
      assert is_map(explanation.feature_weights)
    end
  end

  describe "explain_batch/3" do
    test "explains multiple instances" do
      predict_fn = fn instance -> Enum.sum(instance) * 2.0 end

      instances = [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
      ]

      explanations = LIME.explain_batch(instances, predict_fn, num_samples: 50)

      assert length(explanations) == 3
      assert Enum.all?(explanations, fn exp -> exp.method == :lime end)
      assert Enum.all?(explanations, fn exp -> is_map(exp.feature_weights) end)
    end

    test "handles empty batch" do
      predict_fn = fn instance -> Enum.sum(instance) end

      explanations = LIME.explain_batch([], predict_fn)

      assert explanations == []
    end

    test "handles single instance batch" do
      predict_fn = fn [x] -> x * 2.0 end

      explanations = LIME.explain_batch([[5.0]], predict_fn, num_samples: 50)

      assert length(explanations) == 1
      assert hd(explanations).method == :lime
    end
  end

  describe "consistency and quality" do
    test "similar instances get similar explanations" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      exp1 = LIME.explain([1.0, 2.0], predict_fn, num_samples: 500)
      exp2 = LIME.explain([1.01, 2.01], predict_fn, num_samples: 500)

      # Feature importance ranking should be similar
      top1 = Explanation.top_features(exp1, 2) |> Enum.map(fn {idx, _} -> idx end)
      top2 = Explanation.top_features(exp2, 2) |> Enum.map(fn {idx, _} -> idx end)

      # Same features should be important
      assert top1 == top2
    end

    test "local fidelity: prediction close to model for instance" do
      predict_fn = fn [x, y] -> x * x + y * y end
      instance = [3.0, 4.0]

      explanation = LIME.explain(instance, predict_fn, num_samples: 500)

      # Reconstruct prediction using explanation
      predicted_value =
        Enum.reduce(explanation.feature_weights, explanation.intercept, fn {idx, weight}, acc ->
          acc + weight * Enum.at(instance, idx)
        end)

      actual_value = predict_fn.(instance)

      # Local approximation should be reasonably close
      # Note: won't be perfect for nonlinear functions, but should be in the ballpark
      assert_in_delta predicted_value, actual_value, abs(actual_value) * 0.5
    end
  end

  # Property-based tests
  property "always returns valid explanation structure" do
    check all(
            n_features <- integer(1..5),
            num_samples <- integer(50..200)
          ) do
      instance = for _ <- 1..n_features, do: :rand.uniform() * 10.0
      predict_fn = fn inst -> Enum.sum(inst) end

      explanation = LIME.explain(instance, predict_fn, num_samples: num_samples)

      # Verify structure
      assert %Explanation{} = explanation
      assert explanation.instance == instance
      assert explanation.method == :lime
      assert is_map(explanation.feature_weights)
      assert is_float(explanation.intercept) or is_nil(explanation.intercept)
      assert is_float(explanation.score)
      assert is_map(explanation.metadata)
    end
  end

  property "feature weights are finite numbers" do
    check all(n_features <- integer(1..6)) do
      instance = for _ <- 1..n_features, do: :rand.uniform() * 10.0
      predict_fn = fn inst -> Enum.sum(inst) * 2.0 end

      explanation = LIME.explain(instance, predict_fn, num_samples: 100)

      # All weights should be finite
      assert Enum.all?(explanation.feature_weights, fn {_idx, weight} ->
               is_float(weight) and not nan?(weight) and not infinity?(weight)
             end)
    end
  end

  defp nan?(x) when is_float(x), do: :erlang.float_to_binary(x) == "nan"
  defp infinity?(x), do: abs(x) > 1.0e308
end

defmodule CrucibleXAI.SHAPTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.SHAP

  describe "explain/4" do
    @tag :capture_log
    test "explains simple model with SHAP" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      shap_values = SHAP.explain(instance, background, predict_fn, num_samples: 1000)

      assert is_map(shap_values)
      assert map_size(shap_values) == 2
    end

    @tag :capture_log
    test "uses default method (kernel_shap)" do
      predict_fn = fn [x] -> x * 2.0 end
      instance = [5.0]
      background = [[0.0]]

      shap_values = SHAP.explain(instance, background, predict_fn)

      assert is_map(shap_values)
    end

    @tag :capture_log
    test "handles different numbers of features" do
      predict_fn = fn inst -> Enum.sum(inst) end

      # 1 feature
      shap1 = SHAP.explain([5.0], [[0.0]], predict_fn, num_samples: 100)
      assert map_size(shap1) == 1

      # 3 features
      shap3 = SHAP.explain([1.0, 2.0, 3.0], [[0.0, 0.0, 0.0]], predict_fn, num_samples: 500)
      assert map_size(shap3) == 3
    end
  end

  describe "explain_batch/4" do
    @tag :capture_log
    test "explains multiple instances" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0], [3.0]]
      background = [[0.0]]

      shap_list = SHAP.explain_batch(instances, background, predict_fn, num_samples: 500)

      assert length(shap_list) == 3
      assert Enum.all?(shap_list, &is_map/1)
    end

    @tag :capture_log
    test "handles empty batch" do
      predict_fn = fn [x] -> x end
      background = [[0.0]]

      shap_list = SHAP.explain_batch([], background, predict_fn)

      assert shap_list == []
    end
  end

  describe "verify_additivity/5" do
    @tag :capture_log
    test "verifies additivity property holds" do
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      shap_values = SHAP.explain(instance, background, predict_fn, num_samples: 2000)

      is_valid = SHAP.verify_additivity(shap_values, instance, background, predict_fn, 0.5)

      assert is_valid == true
    end

    @tag :capture_log
    test "detects when additivity is violated" do
      predict_fn = fn [x, y] -> x + y end
      instance = [5.0, 5.0]
      background = [[0.0, 0.0]]

      # Create fake SHAP values that don't sum correctly
      fake_shap = %{0 => 1.0, 1 => 1.0}

      is_valid = SHAP.verify_additivity(fake_shap, instance, background, predict_fn, 0.5)

      # Should be false since 1 + 1 = 2, but prediction - baseline = 10
      assert is_valid == false
    end
  end

  # Property-based test
  property "SHAP values have valid structure" do
    check all(n_features <- integer(1..4)) do
      instance = for _ <- 1..n_features, do: :rand.uniform() * 10.0
      background = [List.duplicate(0.0, n_features)]
      predict_fn = fn inst -> Enum.sum(inst) * 2.0 end

      shap_values = SHAP.explain(instance, background, predict_fn, num_samples: 500)

      # Should return map with correct number of features
      assert is_map(shap_values)
      assert map_size(shap_values) == n_features
      # All values should be floats
      assert Enum.all?(shap_values, fn {_idx, val} -> is_float(val) end)
    end
  end
end

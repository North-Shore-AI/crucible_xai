defmodule CrucibleXAI.FeatureAttributionTest do
  use ExUnit.Case, async: true
  doctest CrucibleXAI.FeatureAttribution

  alias CrucibleXAI.FeatureAttribution

  test "permutation_importance/3 delegates correctly" do
    predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
    validation_data = [{[1.0, 1.0], 5.0}, {[2.0, 2.0], 10.0}]

    importance =
      FeatureAttribution.permutation_importance(predict_fn, validation_data, num_repeats: 2)

    assert is_map(importance)
    assert map_size(importance) == 2
  end

  test "top_k/2 returns top features" do
    importance = %{
      0 => %{importance: 0.5, std_dev: 0.1},
      1 => %{importance: 0.8, std_dev: 0.2},
      2 => %{importance: 0.3, std_dev: 0.05}
    }

    top = FeatureAttribution.top_k(importance, 2)

    assert length(top) == 2
    {first_idx, _} = hd(top)
    assert first_idx == 1
  end
end

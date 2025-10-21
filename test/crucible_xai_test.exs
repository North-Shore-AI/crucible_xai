defmodule CrucibleXaiTest do
  use ExUnit.Case, async: true
  doctest CrucibleXai

  alias CrucibleXAI.Explanation

  @tag :capture_log
  test "explain/3 works via convenience function" do
    predict_fn = fn [x] -> x * 2.0 end
    instance = [5.0]

    explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 100)

    assert %Explanation{} = explanation
    assert explanation.method == :lime
    assert explanation.instance == instance
  end

  @tag :capture_log
  test "explain_batch/3 works via convenience function" do
    predict_fn = fn [x] -> x * 3.0 end
    instances = [[1.0], [2.0], [3.0]]

    explanations = CrucibleXai.explain_batch(instances, predict_fn, num_samples: 50)

    assert length(explanations) == 3
    assert Enum.all?(explanations, fn exp -> exp.method == :lime end)
  end
end

defmodule CrucibleXAI.SHAP.BatchTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.SHAP

  describe "explain_batch/4 with parallel processing" do
    @tag :capture_log
    test "processes batch in parallel and returns all explanations" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0], [3.0], [4.0], [5.0]]
      background = [[0.0]]

      shap_list =
        SHAP.explain_batch(instances, background, predict_fn,
          num_samples: 500,
          parallel: true
        )

      assert length(shap_list) == 5
      assert Enum.all?(shap_list, &is_map/1)
    end

    @tag :capture_log
    test "parallel processing is faster than sequential for SHAP" do
      predict_fn = fn [x, y] -> x * 2.0 + y * 3.0 end
      # Create a batch of 10 instances
      instances = for i <- 1..10, do: [i * 1.0, i * 2.0]
      background = [[0.0, 0.0]]

      # Time sequential processing
      {time_seq, _results} =
        :timer.tc(fn ->
          SHAP.explain_batch(instances, background, predict_fn,
            num_samples: 300,
            parallel: false
          )
        end)

      # Time parallel processing
      {time_par, _results} =
        :timer.tc(fn ->
          SHAP.explain_batch(instances, background, predict_fn,
            num_samples: 300,
            parallel: true
          )
        end)

      # Parallel should be faster (allow some margin)
      assert time_par < time_seq * 0.8
    end

    @tag :capture_log
    test "respects max_concurrency option for SHAP" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = for i <- 1..5, do: [i * 1.0]
      background = [[0.0]]

      # Should not crash with limited concurrency
      shap_list =
        SHAP.explain_batch(instances, background, predict_fn,
          num_samples: 500,
          parallel: true,
          max_concurrency: 2
        )

      assert length(shap_list) == 5
    end

    @tag :capture_log
    test "handles errors in SHAP batch processing gracefully" do
      # Prediction function that fails for large values
      predict_fn = fn
        [x] when x > 100.0 -> raise "SHAP prediction failed"
        [x] -> x * 2.0
      end

      instances = [[10.0], [20.0], [150.0], [30.0]]
      background = [[0.0]]

      # With parallel processing and error skipping
      results =
        SHAP.explain_batch(instances, background, predict_fn,
          num_samples: 500,
          parallel: true,
          on_error: :skip
        )

      # Should have 3 successful explanations
      assert length(results) == 3
      assert Enum.all?(results, &is_map/1)
    end

    @tag :capture_log
    test "sequential SHAP processing when parallel: false" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0], [3.0]]
      background = [[0.0]]

      shap_list =
        SHAP.explain_batch(instances, background, predict_fn,
          num_samples: 500,
          parallel: false
        )

      assert length(shap_list) == 3
    end

    @tag :capture_log
    test "parallel SHAP preserves order of results" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = for i <- 1..5, do: [i * 1.0]
      background = [[0.0]]

      shap_list =
        SHAP.explain_batch(instances, background, predict_fn,
          num_samples: 500,
          parallel: true
        )

      # Results should be in same order as inputs (can verify by checking values)
      assert length(shap_list) == 5
      # Each SHAP value should approximately equal 2 * instance_value
      # (for this simple linear model)
      Enum.zip(instances, shap_list)
      |> Enum.each(fn {[x], shap_map} ->
        shap_val = Map.get(shap_map, 0)
        # Allow some tolerance for SHAP approximation
        assert_in_delta shap_val, x * 2.0, 1.0
      end)
    end
  end
end

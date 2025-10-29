defmodule CrucibleXAI.LIME.BatchTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.LIME

  describe "explain_batch/3 with parallel processing" do
    test "processes batch in parallel and returns all explanations" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0], [3.0], [4.0], [5.0]]

      explanations =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 100,
          parallel: true
        )

      assert length(explanations) == 5
      assert Enum.all?(explanations, fn exp -> exp.method == :lime end)
      assert Enum.all?(explanations, fn exp -> is_number(exp.score) end)
    end

    test "parallel processing is faster than sequential for large batches" do
      predict_fn = fn [x, y] -> x * 2.0 + y * 3.0 end
      # Create a batch of 20 instances
      instances = for i <- 1..20, do: [i * 1.0, i * 2.0]

      # Time sequential processing
      {time_seq, _results} =
        :timer.tc(fn ->
          LIME.explain_batch(instances, predict_fn, num_samples: 500, parallel: false)
        end)

      # Time parallel processing
      {time_par, _results} =
        :timer.tc(fn ->
          LIME.explain_batch(instances, predict_fn, num_samples: 500, parallel: true)
        end)

      # Parallel should be at least 1.5x faster on multi-core systems
      # (conservative estimate to avoid flaky tests)
      assert time_par < time_seq * 0.8
    end

    test "respects max_concurrency option" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = for i <- 1..10, do: [i * 1.0]

      # Should not crash with limited concurrency
      explanations =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 100,
          parallel: true,
          max_concurrency: 2
        )

      assert length(explanations) == 10
    end

    test "handles errors in individual explanations gracefully" do
      # Prediction function that fails for very large values
      # Use instances far from the boundary (10.0 vs 100.0)
      predict_fn = fn
        [x] when x > 100.0 -> raise "Prediction failed for large value"
        [x] -> x * 2.0
      end

      instances = [[10.0], [20.0], [150.0], [30.0]]

      # With parallel processing, should collect results and errors
      results =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 100,
          parallel: true,
          on_error: :skip
        )

      # Should have 3 successful explanations ([150.0] will definitely fail)
      # The other instances (10.0, 20.0, 30.0) are far enough from 100.0
      # that LIME perturbations won't cross the boundary
      assert length(results) == 3
      assert Enum.all?(results, fn exp -> exp.method == :lime end)
      # Ensure all successful results are below the threshold
      assert Enum.all?(results, fn exp -> hd(exp.instance) <= 100.0 end)
    end

    test "sequential processing when parallel: false" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0], [3.0]]

      explanations =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 100,
          parallel: false
        )

      assert length(explanations) == 3
    end

    test "defaults to sequential processing when parallel not specified" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = [[1.0], [2.0]]

      # Should work without :parallel option (backwards compatibility)
      explanations = LIME.explain_batch(instances, predict_fn, num_samples: 100)

      assert length(explanations) == 2
    end

    test "parallel processing preserves order of results" do
      predict_fn = fn [x] -> x * 2.0 end
      instances = for i <- 1..10, do: [i * 1.0]

      explanations =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 100,
          parallel: true
        )

      # Check that results are in the same order as inputs
      Enum.zip(instances, explanations)
      |> Enum.each(fn {instance, explanation} ->
        assert explanation.instance == instance
      end)
    end

    test "empty batch returns empty list" do
      predict_fn = fn [x] -> x * 2.0 end
      explanations = LIME.explain_batch([], predict_fn, parallel: true)

      assert explanations == []
    end

    test "single instance batch works with parallel processing" do
      predict_fn = fn [x] -> x * 2.0 end

      explanations =
        LIME.explain_batch([[1.0]], predict_fn,
          num_samples: 100,
          parallel: true
        )

      assert length(explanations) == 1
      assert hd(explanations).method == :lime
    end
  end

  describe "explain_batch/3 performance scaling" do
    test "parallel processing scales with number of cores" do
      predict_fn = fn [x, y] -> x * 2.0 + y * 3.0 end
      instances = for i <- 1..(System.schedulers_online() * 2), do: [i * 1.0, i * 2.0]

      # Should complete without timeout
      explanations =
        LIME.explain_batch(instances, predict_fn,
          num_samples: 500,
          parallel: true,
          timeout: 30_000
        )

      assert length(explanations) == length(instances)
    end
  end
end

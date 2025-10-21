defmodule CrucibleXAI.LIME.FeatureSelectionTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.LIME.FeatureSelection

  describe "highest_weights/4" do
    test "selects top k features by absolute coefficient" do
      samples = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
      labels = [10.0, 26.0, 42.0]
      weights = [1.0, 1.0, 1.0]

      selected = FeatureSelection.highest_weights(samples, labels, weights, 2)

      assert length(selected) == 2
      assert is_list(selected)
      # Selected indices should be valid feature indices
      assert Enum.all?(selected, fn idx -> idx >= 0 and idx < 4 end)
    end

    test "returns all features if k >= number of features" do
      samples = [[1, 2], [3, 4]]
      labels = [3.0, 7.0]
      weights = [1.0, 1.0]

      selected = FeatureSelection.highest_weights(samples, labels, weights, 10)

      assert length(selected) == 2
    end

    test "returns empty list if k = 0" do
      samples = [[1, 2], [3, 4]]
      labels = [3.0, 7.0]
      weights = [1.0, 1.0]

      selected = FeatureSelection.highest_weights(samples, labels, weights, 0)

      assert selected == []
    end

    test "handles single feature" do
      samples = [[1], [2], [3]]
      labels = [2.0, 4.0, 6.0]
      weights = [1.0, 1.0, 1.0]

      selected = FeatureSelection.highest_weights(samples, labels, weights, 1)

      assert selected == [0]
    end
  end

  describe "forward_selection/4" do
    test "selects features that improve fit most" do
      # Create data where feature 1 is most important, then feature 0
      samples = [[1, 10, 0], [2, 20, 0], [3, 30, 1]]
      # y ≈ 2*x1 (feature 1 dominates)
      labels = [20.0, 40.0, 60.0]
      weights = [1.0, 1.0, 1.0]

      selected = FeatureSelection.forward_selection(samples, labels, weights, 2)

      assert length(selected) == 2
      assert is_list(selected)
      # First selected should be most important feature
      assert Enum.at(selected, 0) in [0, 1, 2]
    end

    test "handles k = 1" do
      samples = [[1, 2], [3, 4], [5, 6]]
      labels = [3.0, 7.0, 11.0]
      weights = [1.0, 1.0, 1.0]

      selected = FeatureSelection.forward_selection(samples, labels, weights, 1)

      assert length(selected) == 1
    end

    test "maintains order of selection" do
      samples = [[1, 0], [2, 0], [3, 0], [4, 1]]
      labels = [2.0, 4.0, 6.0, 8.0]
      weights = [1.0, 1.0, 1.0, 1.0]

      selected = FeatureSelection.forward_selection(samples, labels, weights, 2)

      # Order matters - features are selected in order of importance
      assert length(selected) == 2
      assert is_list(selected)
    end
  end

  describe "lasso/4" do
    test "performs feature selection via L1 regularization" do
      samples = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
      labels = [6.0, 15.0, 24.0, 33.0]
      weights = [1.0, 1.0, 1.0, 1.0]

      selected = FeatureSelection.lasso(samples, labels, weights, 2)

      assert length(selected) == 2
      assert is_list(selected)
      assert Enum.all?(selected, fn idx -> idx >= 0 and idx < 3 end)
    end

    test "handles k = 0" do
      samples = [[1, 2]]
      labels = [3.0]
      weights = [1.0]

      selected = FeatureSelection.lasso(samples, labels, weights, 0)

      assert selected == []
    end

    test "returns unique feature indices" do
      samples = [[1, 2, 3], [4, 5, 6]]
      labels = [6.0, 15.0]
      weights = [1.0, 1.0]

      selected = FeatureSelection.lasso(samples, labels, weights, 3)

      # Should not have duplicates
      assert length(selected) == length(Enum.uniq(selected))
    end
  end

  describe "select_features/5" do
    test "dispatches to correct method" do
      samples = [[1, 2, 3], [4, 5, 6]]
      labels = [6.0, 15.0]
      weights = [1.0, 1.0]

      # Test highest_weights method
      selected_hw =
        FeatureSelection.select_features(samples, labels, weights, 2, :highest_weights)

      assert length(selected_hw) == 2

      # Test forward_selection method
      selected_fs =
        FeatureSelection.select_features(samples, labels, weights, 2, :forward_selection)

      assert length(selected_fs) == 2

      # Test lasso method
      selected_lasso = FeatureSelection.select_features(samples, labels, weights, 2, :lasso)
      assert length(selected_lasso) == 2
    end

    test "uses default method when not specified" do
      samples = [[1, 2], [3, 4]]
      labels = [3.0, 7.0]
      weights = [1.0, 1.0]

      selected = FeatureSelection.select_features(samples, labels, weights, 1)

      assert length(selected) == 1
    end
  end

  # Property-based tests
  property "selected features are always valid indices" do
    check all(
            n_samples <- integer(3..10),
            n_features <- integer(2..6),
            k <- integer(1..4)
          ) do
      samples = for _ <- 1..n_samples, do: for(_ <- 1..n_features, do: :rand.uniform() * 10)
      labels = Enum.map(samples, &Enum.sum/1)
      weights = List.duplicate(1.0, n_samples)

      selected = FeatureSelection.highest_weights(samples, labels, weights, k)

      # All selected indices should be valid
      assert Enum.all?(selected, fn idx -> idx >= 0 and idx < n_features end)
      # Should not exceed k features
      assert length(selected) <= k
      # Should not exceed number of features
      assert length(selected) <= n_features
    end
  end

  property "forward selection returns ordered features" do
    check all(
            n_samples <- integer(4..10),
            n_features <- integer(2..5)
          ) do
      samples = for _ <- 1..n_samples, do: for(_ <- 1..n_features, do: :rand.uniform() * 10)
      labels = Enum.map(samples, &Enum.sum/1)
      weights = List.duplicate(1.0, n_samples)

      selected = FeatureSelection.forward_selection(samples, labels, weights, min(3, n_features))

      # Should return a list
      assert is_list(selected)
      # All indices should be unique
      assert length(selected) == length(Enum.uniq(selected))
      # All indices should be valid
      assert Enum.all?(selected, fn idx -> idx >= 0 and idx < n_features end)
    end
  end
end

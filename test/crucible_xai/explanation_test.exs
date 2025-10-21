defmodule CrucibleXAI.ExplanationTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.Explanation

  describe "struct creation" do
    test "creates explanation with all fields" do
      explanation = %Explanation{
        instance: [1.0, 2.0, 3.0],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        intercept: 1.0,
        score: 0.95,
        method: :lime,
        metadata: %{num_samples: 5000}
      }

      assert explanation.instance == [1.0, 2.0, 3.0]
      assert explanation.method == :lime
      assert explanation.score == 0.95
    end

    test "creates explanation with minimal fields" do
      explanation = %Explanation{
        instance: [1.0, 2.0],
        feature_weights: %{0 => 0.5},
        method: :lime
      }

      assert explanation.instance == [1.0, 2.0]
      assert is_nil(explanation.intercept)
      assert is_nil(explanation.score)
    end
  end

  describe "top_features/2" do
    test "returns top k features by absolute weight" do
      explanation = %Explanation{
        instance: [1, 2, 3, 4, 5],
        feature_weights: %{0 => 0.5, 1 => -0.8, 2 => 0.3, 3 => 0.9, 4 => -0.2},
        method: :lime
      }

      top = Explanation.top_features(explanation, 3)

      assert length(top) == 3
      # Should be sorted by absolute value: 3 (0.9), 1 (-0.8), 0 (0.5)
      assert Enum.at(top, 0) == {3, 0.9}
      assert Enum.at(top, 1) == {1, -0.8}
      assert Enum.at(top, 2) == {0, 0.5}
    end

    test "returns all features if k > number of features" do
      explanation = %Explanation{
        instance: [1, 2],
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :lime
      }

      top = Explanation.top_features(explanation, 10)

      assert length(top) == 2
    end

    test "handles empty feature weights" do
      explanation = %Explanation{
        instance: [],
        feature_weights: %{},
        method: :lime
      }

      top = Explanation.top_features(explanation, 5)

      assert top == []
    end
  end

  describe "positive_features/1" do
    test "returns features with positive weights" do
      explanation = %Explanation{
        instance: [1, 2, 3],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        method: :lime
      }

      positive = Explanation.positive_features(explanation)

      assert length(positive) == 2
      assert {0, 0.5} in positive
      assert {2, 0.8} in positive
    end

    test "returns empty list when no positive features" do
      explanation = %Explanation{
        instance: [1, 2],
        feature_weights: %{0 => -0.5, 1 => -0.3},
        method: :lime
      }

      positive = Explanation.positive_features(explanation)

      assert positive == []
    end
  end

  describe "negative_features/1" do
    test "returns features with negative weights" do
      explanation = %Explanation{
        instance: [1, 2, 3],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        method: :lime
      }

      negative = Explanation.negative_features(explanation)

      assert length(negative) == 1
      assert {1, -0.3} in negative
    end

    test "sorts by absolute value descending" do
      explanation = %Explanation{
        instance: [1, 2, 3],
        feature_weights: %{0 => -0.2, 1 => -0.8, 2 => -0.5},
        method: :lime
      }

      negative = Explanation.negative_features(explanation)

      assert Enum.at(negative, 0) == {1, -0.8}
      assert Enum.at(negative, 1) == {2, -0.5}
      assert Enum.at(negative, 2) == {0, -0.2}
    end
  end

  describe "to_text/1" do
    test "generates text representation" do
      explanation = %Explanation{
        instance: [1.0, 2.0, 3.0],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        intercept: 1.0,
        score: 0.95,
        method: :lime,
        metadata: %{}
      }

      text = Explanation.to_text(explanation)

      assert is_binary(text)
      assert text =~ "LIME"
      assert text =~ "0.95"
      assert text =~ "Feature"
    end

    test "handles missing optional fields" do
      explanation = %Explanation{
        instance: [1.0, 2.0],
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :shap
      }

      text = Explanation.to_text(explanation)

      assert is_binary(text)
      assert text =~ "SHAP"
    end
  end

  describe "to_map/1" do
    test "converts explanation to JSON-serializable map" do
      explanation = %Explanation{
        instance: [1.0, 2.0, 3.0],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        intercept: 1.0,
        score: 0.95,
        method: :lime,
        metadata: %{num_samples: 5000}
      }

      map = Explanation.to_map(explanation)

      assert is_map(map)
      assert map.instance == [1.0, 2.0, 3.0]
      assert map.method == "lime"
      assert map.score == 0.95
      # Feature weights should be converted to string keys for JSON
      assert is_map(map.feature_weights)
    end

    test "handles nil fields" do
      explanation = %Explanation{
        instance: [1.0],
        feature_weights: %{0 => 0.5},
        method: :lime
      }

      map = Explanation.to_map(explanation)

      assert map.intercept == nil
      assert map.score == nil
    end
  end

  describe "feature_importance/1" do
    test "returns list of {feature_index, importance} sorted by absolute value" do
      explanation = %Explanation{
        instance: [1, 2, 3],
        feature_weights: %{0 => 0.5, 1 => -0.8, 2 => 0.3},
        method: :lime
      }

      importance = Explanation.feature_importance(explanation)

      assert length(importance) == 3
      assert Enum.at(importance, 0) == {1, 0.8}
      assert Enum.at(importance, 1) == {0, 0.5}
      assert Enum.at(importance, 2) == {2, 0.3}
    end
  end
end

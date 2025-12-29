defmodule CrucibleXAI.VisualizationTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.{Explanation, Visualization}

  describe "to_html/2" do
    test "generates HTML for LIME explanation" do
      explanation = %Explanation{
        instance: [1.0, 2.0, 3.0],
        feature_weights: %{0 => 0.5, 1 => -0.3, 2 => 0.8},
        intercept: 1.0,
        score: 0.95,
        method: :lime,
        metadata: %{num_samples: 5000}
      }

      html = Visualization.to_html(explanation)

      assert is_binary(html)
      assert html =~ "<!DOCTYPE html>"
      assert html =~ "LIME"
      assert html =~ "Feature"
      assert html =~ "0.95"
    end

    test "generates HTML for SHAP values" do
      shap_values = %{0 => 2.0, 1 => 3.0, 2 => -1.0}

      html = Visualization.shap_to_html(shap_values, [1.0, 1.0, 1.0])

      assert is_binary(html)
      assert html =~ "<!DOCTYPE html>"
      assert html =~ "SHAP"
    end

    test "handles feature names" do
      explanation = %Explanation{
        instance: [25.0, 50_000.0],
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :lime
      }

      feature_names = %{0 => "Age", 1 => "Income"}
      html = Visualization.to_html(explanation, feature_names: feature_names)

      assert html =~ "Age"
      assert html =~ "Income"
    end
  end

  describe "save_html/3" do
    test "saves HTML to file" do
      explanation = %Explanation{
        instance: [1.0, 2.0],
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :lime
      }

      path = "/tmp/crucible_xai_test_#{:rand.uniform(10000)}.html"

      {:ok, saved_path} = Visualization.save_html(explanation, path)

      assert File.exists?(saved_path)
      assert saved_path == path

      # Cleanup
      File.rm(path)
    end
  end

  describe "comparison_html/3" do
    test "generates comparison HTML for LIME vs SHAP" do
      lime_exp = %Explanation{
        instance: [1.0, 2.0],
        feature_weights: %{0 => 0.5, 1 => 0.3},
        method: :lime,
        score: 0.95
      }

      shap_values = %{0 => 0.6, 1 => 0.4}

      html = Visualization.comparison_html(lime_exp, shap_values, [1.0, 2.0])

      assert is_binary(html)
      assert html =~ "LIME"
      assert html =~ "SHAP"
      assert html =~ "Comparison"
    end
  end
end

#!/usr/bin/env elixir

# Visualization Example
# This example demonstrates how to generate HTML visualizations

IO.puts("=" |> String.duplicate(80))
IO.puts("Visualization Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# Define a prediction function
predict_fn = fn [age, income, credit_score] ->
  0.4 * age + 0.35 * income + 0.25 * credit_score + 10.0
end

# Instance to explain
instance = [35.0, 75.0, 750.0]

feature_names = %{
  0 => "Age (years)",
  1 => "Income (k$)",
  2 => "Credit Score"
}

IO.puts("Explaining loan approval prediction")
IO.puts("Instance: #{inspect(instance)}")
IO.puts("Features: #{inspect(Map.values(feature_names))}")
IO.puts("")

# Generate LIME explanation
IO.puts("Generating LIME explanation...")
lime_explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 5000)
IO.puts("LIME R² score: #{Float.round(lime_explanation.score, 4)}")
IO.puts("")

# Generate SHAP explanation
IO.puts("Generating SHAP explanation...")

background = [
  [25.0, 50.0, 700.0],
  [35.0, 75.0, 750.0],
  [45.0, 100.0, 800.0]
]

shap_values = CrucibleXai.explain_shap(instance, background, predict_fn, num_samples: 1000)
IO.puts("SHAP values generated")
IO.puts("")

# Generate HTML visualizations
output_dir = "/home/home/p/g/n/North-Shore-AI/crucible_xai/examples/output"
File.mkdir_p!(output_dir)

# 1. LIME visualization
IO.puts("Generating visualizations...")
lime_html = CrucibleXAI.Visualization.to_html(lime_explanation, feature_names: feature_names)
lime_path = Path.join(output_dir, "lime_explanation.html")
File.write!(lime_path, lime_html)
IO.puts("✓ LIME visualization saved to: #{lime_path}")

# 2. SHAP visualization
shap_html =
  CrucibleXAI.Visualization.shap_html(shap_values, instance, feature_names: feature_names)

shap_path = Path.join(output_dir, "shap_explanation.html")
File.write!(shap_path, shap_html)
IO.puts("✓ SHAP visualization saved to: #{shap_path}")

# 3. Comparison visualization
comparison_html =
  CrucibleXAI.Visualization.comparison_html(
    lime_explanation,
    shap_values,
    instance,
    feature_names: feature_names
  )

comparison_path = Path.join(output_dir, "comparison.html")
File.write!(comparison_path, comparison_html)
IO.puts("✓ Comparison visualization saved to: #{comparison_path}")

# 4. Use convenience function
convenience_path = Path.join(output_dir, "lime_convenience.html")

CrucibleXAI.Visualization.save_html(lime_explanation, convenience_path,
  feature_names: feature_names
)

IO.puts("✓ Convenience method visualization saved to: #{convenience_path}")

IO.puts("")
IO.puts("Visualization files created in: #{output_dir}")
IO.puts("")

IO.puts("Generated Files:")
IO.puts("  1. lime_explanation.html - Interactive LIME feature importance chart")
IO.puts("  2. shap_explanation.html - Interactive SHAP values chart")
IO.puts("  3. comparison.html - Side-by-side comparison of LIME and SHAP")
IO.puts("  4. lime_convenience.html - LIME using convenience function")
IO.puts("")

IO.puts("To view the visualizations:")
IO.puts("  Open any HTML file in a web browser")
IO.puts("  Example: firefox #{lime_path}")
IO.puts("")

IO.puts("Features of the visualizations:")
IO.puts("  - Interactive bar charts using Chart.js")
IO.puts("  - Positive features shown in green, negative in red")
IO.puts("  - Custom feature names displayed")
IO.puts("  - Responsive design")
IO.puts("  - Light/dark theme compatible")
IO.puts("")

IO.puts("Example completed successfully!")
IO.puts("=" |> String.duplicate(80))

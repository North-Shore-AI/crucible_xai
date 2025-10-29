#!/usr/bin/env elixir

# Complete Workflow Example
# This example demonstrates a full XAI workflow from model training to visualization

IO.puts("=" |> String.duplicate(80))
IO.puts("Complete XAI Workflow Example")
IO.puts("=" |> String.duplicate(80))
IO.puts("")

# ============================================================================
# STEP 1: Define the Model and Data
# ============================================================================

IO.puts("STEP 1: Model Setup")
IO.puts("-" |> String.duplicate(80))

# Simulate a credit scoring model
credit_model = fn [age, income, credit_history, debt_ratio] ->
  # Normalize inputs (simplified)
  norm_age = age / 100.0
  norm_income = income / 100.0
  norm_credit = credit_history / 10.0
  norm_debt = debt_ratio / 100.0

  # Model computation
  score =
    40.0 * norm_age +
      50.0 * norm_income +
      60.0 * norm_credit -
      30.0 * norm_debt +
      500.0

  # Clamp to 300-850 range (typical credit score range)
  max(300.0, min(850.0, score))
end

feature_names = %{
  0 => "Age (years)",
  1 => "Income (k$)",
  2 => "Credit History (years)",
  3 => "Debt-to-Income Ratio (%)"
}

IO.puts("Model: Credit Score Predictor")
IO.puts("Features: #{Map.values(feature_names) |> Enum.join(", ")}")
IO.puts("Output: Credit score (300-850)")
IO.puts("✓ Model defined")
IO.puts("")

# ============================================================================
# STEP 2: Create Test Dataset
# ============================================================================

IO.puts("STEP 2: Create Test Dataset")
IO.puts("-" |> String.duplicate(80))

# Create diverse test instances
test_data = [
  # [age, income, credit_history, debt_ratio]
  {[25.0, 45.0, 2.0, 35.0], "Young, low income, new credit"},
  {[35.0, 75.0, 8.0, 25.0], "Mid-career, good income, established"},
  {[45.0, 120.0, 15.0, 15.0], "Mature, high income, excellent history"},
  {[28.0, 55.0, 5.0, 45.0], "Young professional, high debt"},
  {[50.0, 90.0, 20.0, 20.0], "Senior, stable finances"}
]

IO.puts("Test dataset: #{length(test_data)} instances")

Enum.with_index(test_data, 1)
|> Enum.each(fn {{instance, description}, idx} ->
  score = credit_model.(instance)
  IO.puts("  #{idx}. #{description}: Score = #{Float.round(score, 0)}")
end)

IO.puts("✓ Test data created")
IO.puts("")

# ============================================================================
# STEP 3: Generate Explanations
# ============================================================================

IO.puts("STEP 3: Generate Explanations")
IO.puts("-" |> String.duplicate(80))

instances = Enum.map(test_data, fn {instance, _} -> instance end)

# LIME explanations
IO.puts("Generating LIME explanations...")
lime_explanations = CrucibleXai.explain_batch(instances, credit_model, num_samples: 3000)
IO.puts("✓ LIME explanations completed")

# SHAP explanations
IO.puts("Generating SHAP explanations...")

background = [
  [30.0, 60.0, 5.0, 30.0],
  [40.0, 80.0, 10.0, 25.0],
  [50.0, 100.0, 15.0, 20.0]
]

shap_explanations =
  Enum.map(instances, fn instance ->
    CrucibleXai.explain_shap(instance, background, credit_model, num_samples: 1000)
  end)

IO.puts("✓ SHAP explanations completed")
IO.puts("")

# ============================================================================
# STEP 4: Feature Importance Analysis
# ============================================================================

IO.puts("STEP 4: Global Feature Importance")
IO.puts("-" |> String.duplicate(80))

# Create validation dataset
validation_data =
  Enum.map(instances, fn instance ->
    {instance, credit_model.(instance)}
  end)

# Calculate permutation importance
feature_importance =
  CrucibleXai.feature_importance(
    credit_model,
    validation_data,
    metric: :mse,
    num_repeats: 10
  )

IO.puts("Permutation Importance Results:")

feature_importance
|> Enum.sort_by(fn {_idx, %{importance: imp}} -> -imp end)
|> Enum.each(fn {idx, %{importance: imp, std_dev: std}} ->
  IO.puts("  #{Map.get(feature_names, idx)}: #{Float.round(imp, 4)} (±#{Float.round(std, 4)})")
end)

IO.puts("")

# ============================================================================
# STEP 5: Analyze Individual Cases
# ============================================================================

IO.puts("STEP 5: Detailed Instance Analysis")
IO.puts("-" |> String.duplicate(80))

# Analyze the first three instances in detail
Enum.zip([
  Enum.take(test_data, 3),
  Enum.take(lime_explanations, 3),
  Enum.take(shap_explanations, 3)
])
|> Enum.with_index(1)
|> Enum.each(fn {{{instance, description}, lime_exp, shap_vals}, idx} ->
  score = credit_model.(instance)

  IO.puts("")
  IO.puts("Instance #{idx}: #{description}")
  IO.puts("  Input: #{inspect(instance)}")
  IO.puts("  Credit Score: #{Float.round(score, 0)}")
  IO.puts("")

  IO.puts("  LIME Analysis (R²=#{Float.round(lime_exp.score, 4)}):")
  top_lime = CrucibleXAI.Explanation.top_features(lime_exp, 4)

  Enum.each(top_lime, fn {fidx, weight} ->
    direction = if weight > 0, do: "↑", else: "↓"
    IO.puts("    #{direction} #{Map.get(feature_names, fidx)}: #{Float.round(weight, 2)}")
  end)

  IO.puts("  SHAP Analysis:")

  shap_vals
  |> Enum.sort_by(fn {_, val} -> -abs(val) end)
  |> Enum.each(fn {fidx, val} ->
    direction = if val > 0, do: "↑", else: "↓"
    IO.puts("    #{direction} #{Map.get(feature_names, fidx)}: #{Float.round(val, 2)}")
  end)
end)

IO.puts("")

# ============================================================================
# STEP 6: Generate Visualizations
# ============================================================================

IO.puts("")
IO.puts("STEP 6: Generate Visualizations")
IO.puts("-" |> String.duplicate(80))

output_dir = "/home/home/p/g/n/North-Shore-AI/crucible_xai/examples/output/workflow"
File.mkdir_p!(output_dir)

# Generate visualization for first instance
{first_instance, first_desc} = hd(test_data)
first_lime = hd(lime_explanations)
first_shap = hd(shap_explanations)

# Save LIME visualization
lime_path = Path.join(output_dir, "workflow_lime.html")
CrucibleXAI.Visualization.save_html(first_lime, lime_path, feature_names: feature_names)
IO.puts("✓ LIME visualization: #{lime_path}")

# Save comparison
comparison_html =
  CrucibleXAI.Visualization.comparison_html(
    first_lime,
    first_shap,
    first_instance,
    feature_names: feature_names
  )

comparison_path = Path.join(output_dir, "workflow_comparison.html")
File.write!(comparison_path, comparison_html)
IO.puts("✓ Comparison visualization: #{comparison_path}")
IO.puts("")

# ============================================================================
# STEP 7: Summary and Recommendations
# ============================================================================

IO.puts("STEP 7: Summary and Recommendations")
IO.puts("-" |> String.duplicate(80))

# Calculate average quality metrics
avg_r2 =
  lime_explanations
  |> Enum.map(& &1.score)
  |> Enum.sum()
  |> Kernel./(length(lime_explanations))

IO.puts("Quality Metrics:")
IO.puts("  Average LIME R² score: #{Float.round(avg_r2, 4)}")
IO.puts("  Instances analyzed: #{length(test_data)}")
IO.puts("  Visualizations generated: 2")
IO.puts("")

IO.puts("Key Findings:")
top_features = CrucibleXAI.FeatureAttribution.top_k(feature_importance, 2)
IO.puts("  Most important features (global):")

Enum.each(top_features, fn {idx, _} ->
  IO.puts("    - #{Map.get(feature_names, idx)}")
end)

IO.puts("")

IO.puts("Recommendations:")
IO.puts("  1. Monitor feature importance over time")
IO.puts("  2. Investigate cases with low R² scores")
IO.puts("  3. Use SHAP for precise attribution requirements")
IO.puts("  4. Use LIME for quick explanations and debugging")
IO.puts("  5. Compare LIME and SHAP for consistency checks")
IO.puts("")

IO.puts("=" |> String.duplicate(80))
IO.puts("Complete Workflow Example Finished Successfully!")
IO.puts("=" |> String.duplicate(80))
IO.puts("")
IO.puts("This workflow demonstrated:")
IO.puts("  ✓ Model setup and data preparation")
IO.puts("  ✓ Batch LIME and SHAP explanations")
IO.puts("  ✓ Global feature importance analysis")
IO.puts("  ✓ Individual instance analysis")
IO.puts("  ✓ Visualization generation")
IO.puts("  ✓ Quality metrics and recommendations")

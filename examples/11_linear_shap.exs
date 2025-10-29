# Example 11: LinearSHAP for Linear Models
#
# This example demonstrates LinearSHAP, a fast exact SHAP computation
# specifically optimized for linear models. LinearSHAP is orders of
# magnitude faster than KernelSHAP (~1ms vs ~1s) and provides exact
# rather than approximate SHAP values.
#
# Run: mix run examples/11_linear_shap.exs

alias CrucibleXAI.SHAP

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("  Example 11: LinearSHAP - Fast Exact SHAP for Linear Models")
IO.puts(String.duplicate("=", 80) <> "\n")

# ============================================================================
# Scenario: Credit Scoring Model
# ============================================================================
# We have a simple linear credit scoring model:
# Score = 0.3*income + 0.25*credit_history + 0.2*employment_years + 0.15*debt_ratio + 50

IO.puts("Scenario: Credit Scoring Model")
IO.puts("Model: Score = 0.3*income + 0.25*credit_history + 0.2*employment + 0.15*debt_ratio + 50")
IO.puts("")

# Model coefficients
coefficients = %{
  # income (in $10k)
  0 => 0.3,
  # credit_history (months)
  1 => 0.25,
  # employment_years
  2 => 0.2,
  # debt_ratio (%)
  3 => 0.15
}

intercept = 50.0

# Applicant to explain
# $80k income, 60 months credit, 5 years employed, 30% debt
applicant = [8.0, 60.0, 5.0, 30.0]

# Background data (average applicants)
background = [
  [5.0, 48.0, 3.0, 35.0],
  [6.0, 55.0, 4.0, 32.0],
  [7.0, 50.0, 6.0, 28.0],
  [5.5, 45.0, 2.0, 40.0],
  [6.5, 52.0, 5.0, 30.0]
]

# Feature names for readability
feature_names = %{
  0 => "Income ($10k)",
  1 => "Credit History (months)",
  2 => "Employment (years)",
  3 => "Debt Ratio (%)"
}

IO.puts("Applicant Profile:")

Enum.each(Enum.with_index(applicant), fn {value, idx} ->
  IO.puts("  #{Map.get(feature_names, idx)}: #{value}")
end)

IO.puts("")

# ============================================================================
# Part 1: LinearSHAP Explanation
# ============================================================================
IO.puts("Part 1: LinearSHAP (Fast & Exact)")
IO.puts(String.duplicate("-", 80))

{time_linear, shap_linear} =
  :timer.tc(fn ->
    SHAP.explain(applicant, background, nil,
      method: :linear_shap,
      coefficients: coefficients,
      intercept: intercept
    )
  end)

IO.puts("Computation time: #{time_linear / 1000} ms")
IO.puts("")
IO.puts("SHAP Values (feature contributions):")

shap_linear
|> Enum.sort_by(fn {_idx, value} -> abs(value) end, :desc)
|> Enum.each(fn {idx, value} ->
  sign = if value >= 0, do: "+", else: ""

  IO.puts(
    "  #{String.pad_trailing(Map.get(feature_names, idx), 30)}: #{sign}#{Float.round(value, 3)}"
  )
end)

# Calculate prediction and baseline
feature_means = CrucibleXAI.SHAP.LinearSHAP.calculate_feature_means(background)

prediction =
  Enum.with_index(applicant)
  |> Enum.map(fn {value, idx} -> value * Map.get(coefficients, idx, 0.0) end)
  |> Enum.sum()
  |> Kernel.+(intercept)

baseline =
  Enum.map(0..3, fn idx ->
    Map.get(coefficients, idx, 0.0) * Map.get(feature_means, idx, 0.0)
  end)
  |> Enum.sum()
  |> Kernel.+(intercept)

shap_sum = Enum.sum(Map.values(shap_linear))

IO.puts("")
IO.puts("Verification:")
IO.puts("  Applicant Score: #{Float.round(prediction, 2)}")
IO.puts("  Average Score: #{Float.round(baseline, 2)}")
IO.puts("  Difference: #{Float.round(prediction - baseline, 2)}")
IO.puts("  SHAP Sum: #{Float.round(shap_sum, 2)}")
IO.puts("  ✓ Additivity satisfied: #{abs(shap_sum - (prediction - baseline)) < 0.01}")

# ============================================================================
# Part 2: Compare with KernelSHAP
# ============================================================================
IO.puts("")
IO.puts("Part 2: Comparison with KernelSHAP")
IO.puts(String.duplicate("-", 80))

# Define prediction function for KernelSHAP
predict_fn = fn [income, credit, employ, debt] ->
  0.3 * income + 0.25 * credit + 0.2 * employ + 0.15 * debt + 50.0
end

{time_kernel, shap_kernel} =
  :timer.tc(fn ->
    SHAP.explain(applicant, background, predict_fn, num_samples: 2000)
  end)

IO.puts("KernelSHAP computation time: #{time_kernel / 1000} ms")
IO.puts("LinearSHAP computation time: #{time_linear / 1000} ms")
IO.puts("Speed improvement: #{Float.round(time_kernel / time_linear, 1)}x faster")
IO.puts("")
IO.puts("Comparison of SHAP values:")

Enum.each(0..3, fn idx ->
  linear_val = Map.get(shap_linear, idx)
  kernel_val = Map.get(shap_kernel, idx)
  diff = abs(linear_val - kernel_val)

  IO.puts(
    "  #{String.pad_trailing(Map.get(feature_names, idx), 30)}: " <>
      "Linear=#{Float.round(linear_val, 3)}, " <>
      "Kernel=#{Float.round(kernel_val, 3)}, " <>
      "Diff=#{Float.round(diff, 3)}"
  )
end)

IO.puts("")
IO.puts("Note: LinearSHAP gives exact values, KernelSHAP gives approximations.")

# ============================================================================
# Part 3: Multiple Applicants
# ============================================================================
IO.puts("")
IO.puts("Part 3: Explaining Multiple Applicants")
IO.puts(String.duplicate("-", 80))

applicants = [
  # High income, excellent credit
  [9.0, 72.0, 8.0, 20.0],
  # Low income, poor credit
  [4.0, 36.0, 1.0, 45.0],
  # Average applicant
  [6.5, 55.0, 4.5, 30.0]
]

applicant_labels = ["High Quality", "High Risk", "Average"]

{batch_time, batch_shap} =
  :timer.tc(fn ->
    Enum.map(applicants, fn app ->
      SHAP.explain(app, background, nil,
        method: :linear_shap,
        coefficients: coefficients,
        intercept: intercept
      )
    end)
  end)

IO.puts("Batch processing time: #{batch_time / 1000} ms")
IO.puts("Average per applicant: #{batch_time / 1000 / length(applicants)} ms")
IO.puts("")

Enum.zip([applicant_labels, applicants, batch_shap])
|> Enum.each(fn {label, app, shap} ->
  score =
    Enum.with_index(app)
    |> Enum.map(fn {val, idx} -> val * Map.get(coefficients, idx) end)
    |> Enum.sum()
    |> Kernel.+(intercept)

  top_feature =
    shap
    |> Enum.max_by(fn {_idx, val} -> abs(val) end)
    |> then(fn {idx, val} -> {Map.get(feature_names, idx), val} end)

  IO.puts("#{label} (Score: #{Float.round(score, 1)}):")

  IO.puts(
    "  Top contributing feature: #{elem(top_feature, 0)} (#{Float.round(elem(top_feature, 1), 2)})"
  )
end)

# ============================================================================
# Key Takeaways
# ============================================================================
IO.puts("")
IO.puts("Key Takeaways:")
IO.puts(String.duplicate("-", 80))
IO.puts("1. LinearSHAP is 100-1000x faster than KernelSHAP for linear models")
IO.puts("2. LinearSHAP provides exact SHAP values, not approximations")
IO.puts("3. Perfect for production systems with linear models (logistic regression, etc.)")
IO.puts("4. SHAP values always satisfy additivity: sum = prediction - baseline")
IO.puts("5. Use LinearSHAP when you have access to model coefficients")
IO.puts("")

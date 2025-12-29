defmodule CrucibleXAI.Validation do
  @moduledoc ~S"""
  Main API for explanation validation and quality metrics.

  Provides comprehensive validation tools to measure faithfulness, infidelity,
  sensitivity, and axiom compliance of explanations.

  ## Overview

  This module implements state-of-the-art validation metrics for XAI,
  enabling you to:

  - **Measure Faithfulness**: Do explanations reflect actual model behavior?
  - **Quantify Infidelity**: How accurate are the explanations?
  - **Test Sensitivity**: Are explanations robust to perturbations?
  - **Verify Axioms**: Do explanations satisfy theoretical properties?

  ## Quick Start

      # Generate explanation
      explanation = CrucibleXai.explain(instance, predict_fn, num_samples: 2000)

      # Comprehensive validation
      validation = CrucibleXAI.Validation.comprehensive_validation(
        explanation,
        instance,
        predict_fn
      )

      IO.puts(validation.summary)

      # Quick quality check
      quick = CrucibleXAI.Validation.quick_validation(
        explanation,
        instance,
        predict_fn
      )

      if quick.passes_quality_gate do
        IO.puts("Explanation is reliable!")
      end

  ## Validation Metrics

  ### Faithfulness Score
  Correlation between feature importance and prediction change when features
  are removed. Range: -1 to 1 (higher is better).

  - **>0.9**: Excellent
  - **0.7-0.9**: Good
  - **0.5-0.7**: Fair
  - **<0.5**: Poor

  ### Infidelity Score
  Mean squared error between actual model changes and explanation-predicted
  changes under perturbations. Range: 0 to ∞ (lower is better).

  - **<0.02**: Excellent
  - **0.02-0.05**: Good
  - **0.05-0.10**: Acceptable
  - **>0.10**: Poor

  ### Stability Score
  Robustness to input perturbations. Range: 0 to 1 (higher is better).

  - **>0.95**: Excellent
  - **0.85-0.95**: Good
  - **0.70-0.85**: Acceptable
  - **<0.70**: Poor

  ## Usage Examples

  ### Example 1: Basic Validation

      explanation = CrucibleXai.explain(instance, predict_fn)

      faithfulness = CrucibleXAI.Validation.Faithfulness.measure_faithfulness(
        instance,
        explanation,
        predict_fn
      )

      IO.puts("Faithfulness: #{faithfulness.faithfulness_score}")

  ### Example 2: Compare Methods

      lime_exp = CrucibleXai.explain(instance, predict_fn)
      shap_vals = CrucibleXai.explain_shap(instance, background, predict_fn)

      result = CrucibleXAI.Validation.Infidelity.compare_methods(
        instance,
        [{:lime, lime_exp.feature_weights}, {:shap, shap_vals}],
        predict_fn
      )

      IO.puts("Best method: #{result.best_method}")

  ### Example 3: Production Monitoring

      defmodule MyApp.XAIMonitor do
        def validate_and_serve(instance, prediction) do
          explanation = generate_explanation(instance)

          quick_validation = CrucibleXAI.Validation.quick_validation(
            explanation,
            instance,
            &MyModel.predict/1
          )

          if not quick_validation.passes_quality_gate do
            Logger.warning("Low quality explanation detected")
            Metrics.increment("xai.quality_gate_failed")
          end

          explanation
        end
      end

  ## References

  Based on academic research:
  - Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations"
  - Hooker et al. (2019) "A Benchmark for Interpretability Methods"
  - Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
  """

  alias CrucibleXAI.Explanation
  alias CrucibleXAI.Validation.{Axioms, Faithfulness, Infidelity, Sensitivity}

  @type sensitivity_result ::
          Sensitivity.input_result() | %{skipped: true, reason: String.t()}

  @type comprehensive_report :: %{
          faithfulness: Faithfulness.faithfulness_result(),
          infidelity: Infidelity.result(),
          sensitivity: sensitivity_result(),
          axioms: Axioms.axioms_result(),
          quality_score: float(),
          summary: String.t()
        }

  @type quick_report :: %{
          faithfulness_score: float(),
          infidelity_score: float(),
          passes_quality_gate: boolean(),
          interpretation: String.t()
        }

  @doc """
  Comprehensive validation of an explanation.

  Runs all validation metrics and returns a complete quality report.
  This is the most thorough validation but takes longer to compute.

  ## Parameters

    * `explanation` - Explanation struct to validate
    * `instance` - Instance that was explained
    * `predict_fn` - Model prediction function
    * `opts` - Options:
      * `:include_sensitivity` - Run sensitivity analysis (default: false, adds ~2s)
      * `:baseline` - Baseline for axiom tests
      * `:method` - Explanation method (:lime, :shap, etc.)
      * `:num_perturbations` - Perturbations for infidelity (default: 100)

  ## Returns

  Map with:
    * `:faithfulness` - Faithfulness test results
    * `:infidelity` - Infidelity measurement results
    * `:sensitivity` - Sensitivity analysis (if enabled)
    * `:axioms` - Axiom verification results
    * `:quality_score` - Overall quality score (0-1)
    * `:summary` - Human-readable summary

  ## Examples

      validation = CrucibleXAI.Validation.comprehensive_validation(
        explanation,
        instance,
        predict_fn,
        include_sensitivity: true,
        baseline: background_data
      )

      IO.puts(validation.summary)
      # => Overall Quality Score: 0.87 / 1.0
      #    Faithfulness: 0.92 (Excellent)
      #    Infidelity: 0.03 (Good)
      #    ...
  """
  @spec comprehensive_validation(Explanation.t(), list(), (any() -> any()), keyword()) ::
          comprehensive_report()
  def comprehensive_validation(%Explanation{} = explanation, instance, predict_fn, opts \\ []) do
    # Run all validation metrics
    faithfulness = Faithfulness.measure_faithfulness(instance, explanation, predict_fn, opts)

    attributions = explanation.feature_weights
    infidelity = Infidelity.compute(instance, attributions, predict_fn, opts)

    # Sensitivity (if requested - slower)
    sensitivity =
      if Keyword.get(opts, :include_sensitivity, false) do
        explain_fn = fn inst ->
          CrucibleXai.explain(inst, predict_fn, Keyword.get(opts, :explain_opts, []))
        end

        Sensitivity.input_sensitivity(instance, explain_fn, predict_fn, opts)
      else
        %{skipped: true, reason: "Disabled (set include_sensitivity: true to enable)"}
      end

    # Axioms (method-specific)
    axioms =
      Axioms.validate_all_axioms(
        attributions,
        instance,
        predict_fn,
        Keyword.put(opts, :method, explanation.method)
      )

    # Generate overall quality score
    quality_score = calculate_quality_score(faithfulness, infidelity, axioms)

    # Generate summary
    summary = generate_validation_summary(faithfulness, infidelity, axioms, quality_score)

    %{
      faithfulness: faithfulness,
      infidelity: infidelity,
      sensitivity: sensitivity,
      axioms: axioms,
      quality_score: quality_score,
      summary: summary
    }
  end

  @doc """
  Quick validation with essential metrics only.

  Runs faithfulness and infidelity tests (fast metrics) for quick quality
  checks in production environments.

  ## Parameters

    * `explanation` - Explanation struct to validate
    * `instance` - Instance that was explained
    * `predict_fn` - Model prediction function
    * `opts` - Options (same as `comprehensive_validation/4`)

  ## Returns

  Map with:
    * `:faithfulness_score` - Faithfulness correlation
    * `:infidelity_score` - Infidelity error
    * `:passes_quality_gate` - Boolean (true if both metrics pass thresholds)

  ## Quality Gate Thresholds

  - Faithfulness: >= 0.7
  - Infidelity: <= 0.1

  ## Examples

      result = CrucibleXAI.Validation.quick_validation(
        explanation,
        instance,
        predict_fn
      )

      if result.passes_quality_gate do
        # Safe to use explanation
        serve_explanation_to_user(explanation)
      else
        # Quality too low
        Logger.warning("Explanation quality below threshold")
        fallback_explanation()
      end
  """
  @spec quick_validation(Explanation.t(), list(), (any() -> any()), keyword()) :: quick_report()
  def quick_validation(explanation, instance, predict_fn, opts \\ []) do
    faithfulness = Faithfulness.measure_faithfulness(instance, explanation, predict_fn, opts)
    infidelity = Infidelity.compute(instance, explanation.feature_weights, predict_fn, opts)

    %{
      faithfulness_score: faithfulness.faithfulness_score,
      infidelity_score: infidelity.infidelity_score,
      passes_quality_gate: passes_quality_gate?(faithfulness, infidelity),
      interpretation: generate_quick_interpretation(faithfulness, infidelity)
    }
  end

  @doc """
  Benchmark multiple explanation methods.

  Compares validation metrics across different explanation methods to help
  select the best method for your use case.

  ## Parameters

    * `instance` - Instance to explain
    * `predict_fn` - Model prediction function
    * `methods` - List of method configurations:
      * `{:lime, opts}` - LIME with options
      * `{:shap, background, opts}` - SHAP with background dataset
      * `{:gradient, model_fn, opts}` - Gradient methods
    * `opts` - Global validation options

  ## Returns

  Map with:
    * `:by_method` - Validation results for each method
    * `:ranking` - Methods ranked by quality score
    * `:best_method` - Method with highest quality
    * `:comparison_summary` - Summary table

  ## Examples

      result = CrucibleXAI.Validation.benchmark_methods(
        instance,
        predict_fn,
        [
          {:lime, num_samples: 2000},
          {:shap, background_data, num_samples: 1000}
        ]
      )

      IO.puts(result.comparison_summary)
      # Method  | Faithfulness | Infidelity | Quality | Time
      # --------|--------------|------------|---------|------
      # LIME    | 0.85         | 0.04       | 0.82    | 45ms
      # SHAP    | 0.91         | 0.02       | 0.89    | 950ms
  """
  @spec benchmark_methods(list(), (any() -> any()), list(), keyword()) :: map()
  def benchmark_methods(instance, predict_fn, methods, opts \\ []) do
    # Generate explanations and validate each
    results =
      methods
      |> Enum.map(fn method_config ->
        {method_name, explanation, time_us} =
          generate_timed_explanation(instance, predict_fn, method_config)

        validation = quick_validation(explanation, instance, predict_fn, opts)

        {method_name,
         %{
           validation: validation,
           time_ms: time_us / 1000.0,
           quality_score: calculate_quick_quality(validation)
         }}
      end)
      |> Map.new()

    # Rank methods by quality score
    ranking =
      results
      |> Enum.map(fn {method, result} -> {method, result.quality_score} end)
      |> Enum.sort_by(fn {_method, score} -> score end, :desc)

    best_method = ranking |> List.first() |> elem(0)

    # Generate comparison summary
    comparison_summary = generate_comparison_summary(results)

    %{
      by_method: results,
      ranking: ranking,
      best_method: best_method,
      comparison_summary: comparison_summary
    }
  end

  # Private helper functions

  defp calculate_quality_score(faithfulness, infidelity, axioms) do
    # Weighted combination of metrics
    # Faithfulness: 40%, Infidelity: 40%, Axioms: 20%

    faith_score = normalize_score(faithfulness.faithfulness_score, :correlation)
    infid_score = normalize_score(infidelity.infidelity_score, :error)
    axiom_score = Map.get(axioms, :overall_score, 0.8)

    0.4 * faith_score + 0.4 * infid_score + 0.2 * axiom_score
  end

  defp calculate_quick_quality(validation) do
    faith_score = normalize_score(validation.faithfulness_score, :correlation)
    infid_score = normalize_score(validation.infidelity_score, :error)

    0.5 * faith_score + 0.5 * infid_score
  end

  defp normalize_score(score, :correlation) do
    # Correlation: -1 to 1 → 0 to 1
    (score + 1.0) / 2.0
  end

  defp normalize_score(score, :error) do
    # Error: 0 to ∞ → 1 to 0 (exponential decay)
    :math.exp(-score * 10.0)
  end

  defp passes_quality_gate?(faithfulness, infidelity) do
    faithfulness.faithfulness_score >= 0.7 and infidelity.infidelity_score <= 0.1
  end

  defp generate_validation_summary(faithfulness, infidelity, axioms, quality_score) do
    """
    === Explanation Quality Report ===

    Overall Quality Score: #{Float.round(quality_score, 3)} / 1.0

    Faithfulness: #{Float.round(faithfulness.faithfulness_score, 3)}
      - #{faithfulness.interpretation}

    Infidelity: #{Float.round(infidelity.infidelity_score, 4)}
      - #{infidelity.interpretation}

    Axioms: #{if axioms.all_satisfied, do: "✓ All satisfied", else: "✗ Some violations"}

    Recommendation: #{recommend_action(quality_score)}
    """
  end

  defp generate_quick_interpretation(faithfulness, infidelity) do
    cond do
      faithfulness.faithfulness_score >= 0.9 and infidelity.infidelity_score <= 0.02 ->
        "Excellent - High quality explanation"

      faithfulness.faithfulness_score >= 0.7 and infidelity.infidelity_score <= 0.05 ->
        "Good - Reliable explanation"

      faithfulness.faithfulness_score >= 0.5 and infidelity.infidelity_score <= 0.10 ->
        "Acceptable - Use with caution"

      true ->
        "Poor - Explanation may be unreliable"
    end
  end

  defp recommend_action(score) when score >= 0.85, do: "Excellent - Safe for production"
  defp recommend_action(score) when score >= 0.70, do: "Good - Acceptable for most uses"
  defp recommend_action(score) when score >= 0.50, do: "Fair - Use with caution"
  defp recommend_action(_), do: "Poor - Explanation may be unreliable"

  defp generate_timed_explanation(instance, predict_fn, {:lime, opts}) do
    {time_us, explanation} =
      :timer.tc(fn ->
        CrucibleXai.explain(instance, predict_fn, opts)
      end)

    {:lime, explanation, time_us}
  end

  defp generate_timed_explanation(instance, predict_fn, {:shap, background, opts}) do
    {time_us, shap_values} =
      :timer.tc(fn ->
        CrucibleXai.explain_shap(instance, background, predict_fn, opts)
      end)

    explanation = %Explanation{
      instance: instance,
      feature_weights: shap_values,
      method: :shap
    }

    {:shap, explanation, time_us}
  end

  defp generate_comparison_summary(results) do
    header = "Method  | Faithfulness | Infidelity | Quality | Time"
    separator = "--------|--------------|------------|---------|------"

    rows =
      Enum.map_join(results, "\n", fn {method, result} ->
        faith = Float.round(result.validation.faithfulness_score, 2)
        infid = Float.round(result.validation.infidelity_score, 2)
        quality = Float.round(result.quality_score, 2)
        time = Float.round(result.time_ms, 0)

        "#{String.pad_trailing(to_string(method), 8)}| #{String.pad_trailing(to_string(faith), 13)}| #{String.pad_trailing(to_string(infid), 11)}| #{String.pad_trailing(to_string(quality), 8)}| #{time}ms"
      end)

    """
    #{header}
    #{separator}
    #{rows}
    """
  end
end

defmodule CrucibleXAI.Validation.Infidelity do
  @moduledoc ~S"""
  Infidelity metric for explanation quality assessment.

  Measures squared error between actual model changes and explanation-predicted
  changes under perturbations. Lower scores indicate more faithful explanations
  (0 = perfect fidelity).

  ## Mathematical Definition

  ```
  Infidelity = E[(f(x) - f(x̃) - φᵀ(x - x̃))²]
  ```

  Where:
  - `x` = original instance
  - `x̃` = perturbed instance
  - `f` = model prediction function
  - `φ` = attribution vector (feature importances)

  ## Interpretation

  - **0.00 - 0.02**: Excellent fidelity
  - **0.02 - 0.05**: Good fidelity
  - **0.05 - 0.10**: Acceptable fidelity
  - **0.10 - 0.20**: Poor fidelity
  - **> 0.20**: Very poor fidelity

  ## Usage

      attributions = explanation.feature_weights

      result = Infidelity.compute(
        instance,
        attributions,
        predict_fn,
        num_perturbations: 100
      )

      IO.puts("Infidelity: #{result.infidelity_score}")
      # => 0.03 (Good)

  ## References

  Based on:
  - Yeh et al. (2019) "On the (In)fidelity and Sensitivity of Explanations", NeurIPS
  """

  @doc """
  Compute infidelity score.

  ## Algorithm

  1. Generate N perturbations of the instance
  2. For each perturbation x̃:
     a. Compute actual model change: Δf = f(x) - f(x̃)
     b. Compute predicted change via attributions: Δφ = φᵀ(x - x̃)
     c. Compute squared error: (Δf - Δφ)²
  3. Return mean squared error across all perturbations

  ## Parameters

    * `instance` - Original instance (list of feature values)
    * `attributions` - Attribution map (feature_index => importance)
    * `predict_fn` - Model prediction function
    * `opts` - Options:
      * `:num_perturbations` - Number of perturbations (default: 100)
      * `:perturbation_std` - Std dev for Gaussian noise (default: 0.1)
      * `:perturbation_method` - `:gaussian`, `:uniform` (default: `:gaussian`)
      * `:normalize` - Normalize by prediction variance (default: false)

  ## Returns

  Map with:
    * `:infidelity_score` - Mean squared error (lower is better, 0 = perfect)
    * `:std_dev` - Standard deviation across perturbations
    * `:individual_errors` - Error for each perturbation
    * `:normalized_score` - Normalized by variance (if normalize: true)
    * `:interpretation` - Quality assessment string

  ## Examples

      # Perfect attribution (zero infidelity)
      attributions = %{0 => 2.0, 1 => 3.0}
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [5.0, 10.0]

      result = Infidelity.compute(instance, attributions, predict_fn)
  # => %{infidelity_score: ~0.0, interpretation: "Excellent", ...}
  """
  @type result :: %{
          infidelity_score: float(),
          std_dev: float(),
          individual_errors: [float()],
          normalized_score: float(),
          interpretation: String.t()
        }

  @spec compute(list(), map(), (any() -> any()), keyword()) :: result()
  def compute(instance, attributions, predict_fn, opts \\ []) do
    num_perturbations = Keyword.get(opts, :num_perturbations, 100)
    perturbation_std = Keyword.get(opts, :perturbation_std, 0.1)
    perturbation_method = Keyword.get(opts, :perturbation_method, :gaussian)
    normalize = Keyword.get(opts, :normalize, false)

    # Get original prediction
    original_pred = get_prediction(predict_fn, instance)

    # Generate perturbations and compute errors
    individual_errors =
      1..num_perturbations
      |> Enum.map(fn _ ->
        # Generate perturbation
        perturbed = generate_perturbation(instance, perturbation_std, perturbation_method)

        # Actual model change
        perturbed_pred = get_prediction(predict_fn, perturbed)
        actual_change = original_pred - perturbed_pred

        # Predicted change via attributions
        predicted_change = compute_predicted_change(instance, perturbed, attributions)

        # Squared error
        error = :math.pow(actual_change - predicted_change, 2)
        error
      end)

    # Statistics
    infidelity_score = Enum.sum(individual_errors) / length(individual_errors)
    variance = compute_variance(individual_errors)
    std_dev = :math.sqrt(variance)

    # Normalized score (if requested)
    normalized_score =
      if normalize do
        # Normalize by prediction variance across perturbations
        prediction_variance = compute_prediction_variance(instance, predict_fn, opts)

        if prediction_variance > 0.0 do
          infidelity_score / prediction_variance
        else
          infidelity_score
        end
      else
        infidelity_score
      end

    # Interpretation
    interpretation = interpret_infidelity(infidelity_score)

    %{
      infidelity_score: infidelity_score,
      std_dev: std_dev,
      individual_errors: individual_errors,
      normalized_score: normalized_score,
      interpretation: interpretation
    }
  end

  @doc """
  Sensitivity analysis across perturbation magnitudes.

  Tests how infidelity changes with perturbation size to ensure the metric
  is robust to the perturbation magnitude choice.

  ## Parameters

    * `instance` - Instance to test
    * `attributions` - Attribution map
    * `predict_fn` - Model prediction function
    * `opts` - Options:
      * `:std_range` - List of std devs to test (default: [0.05, 0.1, 0.2, 0.5])
      * `:num_perturbations` - Perturbations per std dev (default: 50)

  ## Returns

  Map with:
    * `:infidelity_by_std` - Map of std_dev => infidelity_score
    * `:is_stable` - Whether infidelity is stable across magnitudes
    * `:coefficient_of_variation` - Measure of stability

  ## Examples

      result = Infidelity.sensitivity_to_perturbation(
        instance,
        attributions,
        predict_fn
      )
      # => %{infidelity_by_std: %{0.05 => 0.03, 0.1 => 0.04, ...}, ...}
  """
  @spec sensitivity_to_perturbation(list(), map(), (any() -> any()), keyword()) :: map()
  def sensitivity_to_perturbation(instance, attributions, predict_fn, opts \\ []) do
    std_range = Keyword.get(opts, :std_range, [0.05, 0.1, 0.2, 0.5])
    num_perturbations = Keyword.get(opts, :num_perturbations, 50)

    # Compute infidelity for each std dev
    infidelity_by_std =
      std_range
      |> Enum.map(fn std ->
        result =
          compute(instance, attributions, predict_fn,
            perturbation_std: std,
            num_perturbations: num_perturbations
          )

        {std, result.infidelity_score}
      end)
      |> Map.new()

    # Compute stability metrics
    scores = Map.values(infidelity_by_std)
    mean = Enum.sum(scores) / length(scores)
    std_dev = :math.sqrt(compute_variance(scores))
    coefficient_of_variation = if mean > 0, do: std_dev / mean, else: 0.0

    # Stable if CV < 0.3
    is_stable = coefficient_of_variation < 0.3

    %{
      infidelity_by_std: infidelity_by_std,
      is_stable: is_stable,
      coefficient_of_variation: coefficient_of_variation,
      mean_infidelity: mean
    }
  end

  @doc """
  Compare infidelity across multiple explanation methods.

  Useful for selecting the most faithful explanation method for a given
  model and instance.

  ## Parameters

    * `instance` - Instance to test
    * `explanations` - List of explanation structs or attribution maps
    * `predict_fn` - Model prediction function
    * `opts` - Options passed to `compute/4`

  ## Returns

  Map with:
    * `:by_method` - Map of method_name => infidelity_result
    * `:best_method` - Method with lowest infidelity
    * `:worst_method` - Method with highest infidelity
    * `:ranking` - List of {method, score} sorted by quality

  ## Examples

      lime_attrs = %{0 => 2.1, 1 => 2.9}
      shap_attrs = %{0 => 2.0, 1 => 3.0}

      result = Infidelity.compare_methods(
        instance,
        [
          {:lime, lime_attrs},
          {:shap, shap_attrs}
        ],
        predict_fn
      )
      # => %{best_method: :shap, ...}
  """
  @spec compare_methods(list(), list(), function(), keyword()) :: map()
  def compare_methods(instance, explanations, predict_fn, opts \\ []) do
    # Compute infidelity for each method
    by_method =
      explanations
      |> Enum.map(fn
        {method_name, attributions} when is_map(attributions) ->
          result = compute(instance, attributions, predict_fn, opts)
          {method_name, result}

        %{method: method, feature_weights: attrs} ->
          result = compute(instance, attrs, predict_fn, opts)
          {method, result}
      end)
      |> Map.new()

    # Rank methods by infidelity (lower is better)
    ranking =
      by_method
      |> Enum.map(fn {method, result} -> {method, result.infidelity_score} end)
      |> Enum.sort_by(fn {_method, score} -> score end)

    best_method = ranking |> List.first() |> elem(0)
    worst_method = ranking |> List.last() |> elem(0)

    %{
      by_method: by_method,
      best_method: best_method,
      worst_method: worst_method,
      ranking: ranking
    }
  end

  # Private helper functions

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num * 1.0
      _ -> raise ArgumentError, "Prediction function must return number or Nx.Tensor"
    end
  end

  defp generate_perturbation(instance, std, :gaussian) do
    Enum.map(instance, fn val ->
      noise = :rand.normal(0.0, std)
      val + noise
    end)
  end

  defp generate_perturbation(instance, magnitude, :uniform) do
    Enum.map(instance, fn val ->
      noise = (:rand.uniform() - 0.5) * 2.0 * magnitude
      val + noise
    end)
  end

  defp compute_predicted_change(original, perturbed, attributions) do
    # Predicted change: φᵀ(x - x̃)
    original
    |> Enum.with_index()
    |> Enum.map(fn {orig_val, idx} ->
      perturbed_val = Enum.at(perturbed, idx)
      attribution = Map.get(attributions, idx, 0.0)
      attribution * (orig_val - perturbed_val)
    end)
    |> Enum.sum()
  end

  defp compute_variance(data) when length(data) < 2, do: 0.0

  defp compute_variance(data) do
    n = length(data)
    mean = Enum.sum(data) / n

    sum_squared_diffs =
      data
      |> Enum.map(fn x -> :math.pow(x - mean, 2) end)
      |> Enum.sum()

    sum_squared_diffs / n
  end

  defp compute_prediction_variance(instance, predict_fn, opts) do
    num_samples = Keyword.get(opts, :num_perturbations, 100)
    std = Keyword.get(opts, :perturbation_std, 0.1)
    method = Keyword.get(opts, :perturbation_method, :gaussian)

    predictions =
      1..num_samples
      |> Enum.map(fn _ ->
        perturbed = generate_perturbation(instance, std, method)
        get_prediction(predict_fn, perturbed)
      end)

    compute_variance(predictions)
  end

  defp interpret_infidelity(score) do
    cond do
      score < 0.02 -> "Excellent - Very high fidelity"
      score < 0.05 -> "Good - High fidelity"
      score < 0.10 -> "Acceptable - Moderate fidelity"
      score < 0.20 -> "Poor - Low fidelity"
      true -> "Very Poor - Unreliable attribution"
    end
  end
end

defmodule CrucibleXAI.SHAP.LinearSHAP do
  @moduledoc """
  LinearSHAP: Fast, exact SHAP computation for linear models.

  For linear models, SHAP values can be computed directly without sampling:
  φᵢ = wᵢ * (xᵢ - E[xᵢ])

  where:
  - φᵢ is the SHAP value for feature i
  - wᵢ is the coefficient (weight) for feature i
  - xᵢ is the value of feature i in the instance
  - E[xᵢ] is the expected value (mean) of feature i from background data

  This is much faster than KernelSHAP (~1ms vs ~1s) and provides exact values
  rather than approximations.

  ## References

  Lundberg, S. M., & Lee, S. I. (2017).
  A Unified Approach to Interpreting Model Predictions. NeurIPS.
  (Section on linear models and additive feature attribution)

  ## Examples

      # For a linear model: f(x) = 2*x₁ + 3*x₂ + 1
      coefficients = %{0 => 2.0, 1 => 3.0}
      intercept = 1.0
      instance = [5.0, 3.0]
      background = [[0.0, 0.0], [2.0, 2.0]]  # Mean: [1.0, 1.0]

      shap_values = LinearSHAP.explain(instance, background, coefficients, intercept)
      # => %{0 => 8.0, 1 => 6.0}
      # Because: φ₀ = 2.0 * (5.0 - 1.0) = 8.0
      #          φ₁ = 3.0 * (3.0 - 1.0) = 6.0
  """

  @doc """
  Compute exact SHAP values for a linear model.

  ## Parameters
    * `instance` - Instance to explain (list of feature values)
    * `background_data` - Background dataset for computing feature means
    * `coefficients` - Map of feature_index => coefficient
    * `intercept` - Model intercept (bias term)

  ## Returns
    Map of feature_index => shapley_value

  ## Examples

      iex> coefficients = %{0 => 2.0, 1 => 3.0}
      iex> intercept = 0.0
      iex> instance = [1.0, 1.0]
      iex> background = [[0.0, 0.0]]
      iex> shap = CrucibleXAI.SHAP.LinearSHAP.explain(instance, background, coefficients, intercept)
      iex> shap[0]
      2.0
      iex> shap[1]
      3.0
  """
  @spec explain(list(float()), list(list(float())), map(), float()) :: %{integer() => float()}
  def explain(instance, background_data, coefficients, _intercept) when is_list(instance) do
    # Calculate feature means from background data
    feature_means = calculate_feature_means(background_data)

    # For each feature: φᵢ = wᵢ * (xᵢ - E[xᵢ])
    instance
    |> Enum.with_index()
    |> Map.new(fn {x_i, i} ->
      coefficient = Map.get(coefficients, i, 0.0)
      mean = Map.get(feature_means, i, 0.0)
      shap_value = coefficient * (x_i - mean)

      {i, shap_value}
    end)
  end

  @doc """
  Calculate mean value for each feature from background data.

  ## Parameters
    * `background_data` - List of instances (each instance is a list of feature values)

  ## Returns
    Map of feature_index => mean_value

  ## Examples

      iex> background = [[1.0, 2.0], [3.0, 4.0]]
      iex> means = CrucibleXAI.SHAP.LinearSHAP.calculate_feature_means(background)
      iex> means[0]
      2.0
      iex> means[1]
      3.0
  """
  @spec calculate_feature_means(list(list(float()))) :: %{integer() => float()}
  def calculate_feature_means(background_data) when is_list(background_data) do
    num_instances = length(background_data)
    num_features = length(hd(background_data))

    # Transpose: convert list of instances to list of features
    features_transposed =
      for feature_idx <- 0..(num_features - 1) do
        feature_values =
          Enum.map(background_data, fn instance -> Enum.at(instance, feature_idx) end)

        {feature_idx, feature_values}
      end

    # Calculate mean for each feature
    Map.new(features_transposed, fn {feature_idx, values} ->
      mean = Enum.sum(values) / num_instances
      {feature_idx, mean}
    end)
  end
end

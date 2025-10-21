defmodule CrucibleXAI.LIME do
  @moduledoc """
  Local Interpretable Model-agnostic Explanations (LIME).

  LIME explains individual predictions by approximating the model locally
  with an interpretable linear model. It generates perturbed samples around
  the instance, gets predictions from the black-box model, and fits a
  weighted linear model to approximate local behavior.

  ## Algorithm Overview

  1. **Perturbation**: Generate samples around the instance to explain
  2. **Prediction**: Get predictions from the black-box model for samples
  3. **Weighting**: Weight samples by proximity to the instance (kernel)
  4. **Feature Selection**: Optionally select top K most important features
  5. **Approximation**: Fit interpretable linear model on weighted samples
  6. **Explanation**: Extract feature weights as the explanation

  ## References

  Ribeiro, M. T., Singh, S., & Guestrin, C. (2016).
  "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD.

  ## Examples

      # Explain a simple model
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y + 1.0 end
      instance = [1.0, 2.0]
      explanation = CrucibleXAI.LIME.explain(instance, predict_fn)

      # Customize parameters
      explanation = CrucibleXAI.LIME.explain(
        instance,
        predict_fn,
        num_samples: 5000,
        kernel_width: 0.75,
        num_features: 5,
        feature_selection: :forward_selection
      )
  """

  alias CrucibleXAI.LIME.{Sampling, Kernels, FeatureSelection, InterpretableModels}
  alias CrucibleXAI.Explanation

  require Logger

  @default_opts [
    num_samples: 5000,
    kernel_width: 0.75,
    kernel: :exponential,
    num_features: 10,
    feature_selection: :highest_weights,
    model_type: :linear_regression,
    sampling_method: :gaussian
  ]

  @type predict_fn :: (any() -> number() | Nx.Tensor.t())

  @doc """
  Explain a single prediction using LIME.

  ## Parameters
    * `instance` - The instance to explain (list or Nx.Tensor)
    * `predict_fn` - Function that takes input and returns prediction
    * `opts` - Options (see module attributes for defaults)

  ## Options
    * `:num_samples` - Number of perturbed samples (default: 5000)
    * `:kernel_width` - Width of proximity kernel (default: 0.75)
    * `:kernel` - Kernel function (default: :exponential)
    * `:num_features` - Number of features in explanation (default: 10)
    * `:feature_selection` - Method for selecting features (default: :highest_weights)
    * `:model_type` - Interpretable model type (default: :linear_regression)
    * `:sampling_method` - Sampling strategy (default: :gaussian)

  ## Returns
    `%Explanation{}` struct with feature weights and metadata

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instance = [5.0]
      iex> explanation = CrucibleXAI.LIME.explain(instance, predict_fn, num_samples: 100)
      iex> explanation.method
      :lime
      iex> is_map(explanation.feature_weights)
      true
  """
  @spec explain(any(), predict_fn(), keyword()) :: Explanation.t()
  def explain(instance, predict_fn, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    Logger.debug("Starting LIME explanation for instance: #{inspect(instance)}")
    start_time = System.monotonic_time(:millisecond)

    # Step 1: Generate perturbed samples around instance
    num_samples = Keyword.get(opts, :num_samples)
    sampling_method = Keyword.get(opts, :sampling_method)

    samples = generate_samples(instance, num_samples, sampling_method, opts)

    # Step 2: Get predictions from black-box model
    predictions = get_predictions(samples, predict_fn)

    # Step 3: Calculate sample weights using kernel
    kernel = Keyword.get(opts, :kernel)
    kernel_width = Keyword.get(opts, :kernel_width)
    sample_weights = calculate_sample_weights(samples, instance, kernel, kernel_width)

    # Step 4: Feature selection (optional)
    num_features = Keyword.get(opts, :num_features)
    feature_selection_method = Keyword.get(opts, :feature_selection)
    n_total_features = if is_list(instance), do: length(instance), else: Nx.size(instance)
    num_features = min(num_features, n_total_features)

    {selected_features, samples_subset} =
      select_and_subset_features(
        samples,
        predictions,
        sample_weights,
        num_features,
        n_total_features,
        feature_selection_method
      )

    # Step 5: Fit interpretable model
    model_type = Keyword.get(opts, :model_type)
    model = fit_interpretable_model(samples_subset, predictions, sample_weights, model_type)

    # Step 6: Build explanation
    feature_weights = build_feature_weights(model, selected_features)

    duration = System.monotonic_time(:millisecond) - start_time

    Logger.info(
      "LIME explanation completed in #{duration}ms, R²=#{Float.round(model.r_squared, 4)}"
    )

    %Explanation{
      instance: instance,
      feature_weights: feature_weights,
      intercept: model.intercept,
      score: model.r_squared,
      method: :lime,
      metadata: %{
        num_samples: num_samples,
        kernel: kernel,
        kernel_width: kernel_width,
        feature_selection: feature_selection_method,
        duration_ms: duration,
        num_features_selected: length(selected_features)
      }
    }
  end

  @doc """
  Explain multiple instances in parallel.

  ## Parameters
    * `instances` - List of instances to explain
    * `predict_fn` - Prediction function
    * `opts` - Options (same as `explain/3`)

  ## Returns
    List of `%Explanation{}` structs

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instances = [[1.0], [2.0], [3.0]]
      iex> explanations = CrucibleXAI.LIME.explain_batch(instances, predict_fn, num_samples: 50)
      iex> length(explanations)
      3
      iex> Enum.all?(explanations, fn exp -> exp.method == :lime end)
      true
  """
  @spec explain_batch(list(), predict_fn(), keyword()) :: list(Explanation.t())
  def explain_batch(instances, predict_fn, opts \\ []) do
    Enum.map(instances, fn instance ->
      explain(instance, predict_fn, opts)
    end)
  end

  # Private helper functions

  defp generate_samples(instance, num_samples, :gaussian, opts) do
    Sampling.gaussian(instance, num_samples, opts)
  end

  defp generate_samples(instance, num_samples, :uniform, opts) do
    Sampling.uniform(instance, num_samples, opts)
  end

  defp generate_samples(instance, num_samples, :combined, opts) do
    Sampling.combined(instance, num_samples, opts)
  end

  defp generate_samples(instance, num_samples, _method, opts) do
    # Default to Gaussian
    Sampling.gaussian(instance, num_samples, opts)
  end

  defp get_predictions(samples, predict_fn) do
    # Convert samples tensor to list of instances
    samples_list = Nx.to_list(samples)

    # Get predictions for each sample
    predictions =
      Enum.map(samples_list, fn sample ->
        result = predict_fn.(sample)
        # Handle both scalar and tensor returns
        case result do
          %Nx.Tensor{} -> Nx.to_number(result)
          num when is_number(num) -> num
        end
      end)

    Nx.tensor(predictions)
  end

  defp calculate_sample_weights(samples, instance, kernel, kernel_width) do
    # Calculate distances from instance
    distances = Kernels.euclidean_distance(samples, instance)

    # Apply kernel to get weights
    Kernels.apply_kernel(distances, kernel, kernel_width: kernel_width)
  end

  defp select_and_subset_features(
         samples,
         predictions,
         sample_weights,
         num_features,
         n_total_features,
         feature_selection_method
       ) do
    if num_features >= n_total_features do
      # Use all features
      {Enum.to_list(0..(n_total_features - 1)), samples}
    else
      # Select subset of features
      samples_list = Nx.to_list(samples)
      predictions_list = Nx.to_flat_list(predictions)
      weights_list = Nx.to_flat_list(sample_weights)

      selected_features =
        FeatureSelection.select_features(
          samples_list,
          predictions_list,
          weights_list,
          num_features,
          feature_selection_method
        )

      # Create subset of samples with only selected features
      samples_subset =
        Enum.map(samples_list, fn sample ->
          Enum.map(selected_features, fn idx -> Enum.at(sample, idx) end)
        end)

      {selected_features, Nx.tensor(samples_subset)}
    end
  end

  defp fit_interpretable_model(samples, predictions, sample_weights, :linear_regression) do
    InterpretableModels.LinearRegression.fit(samples, predictions, sample_weights)
  end

  defp fit_interpretable_model(samples, predictions, sample_weights, :ridge) do
    InterpretableModels.Ridge.fit(samples, predictions, sample_weights)
  end

  defp fit_interpretable_model(samples, predictions, sample_weights, _) do
    # Default to linear regression
    InterpretableModels.LinearRegression.fit(samples, predictions, sample_weights)
  end

  defp build_feature_weights(model, selected_features) do
    # Map coefficients to their feature indices
    model.coefficients
    |> Enum.zip(selected_features)
    |> Enum.map(fn {coef, feature_idx} -> {feature_idx, coef} end)
    |> Enum.into(%{})
  end
end

defmodule CrucibleXAI.SHAP do
  @moduledoc """
  SHAP (SHapley Additive exPlanations) for feature attribution.

  SHAP uses game theory (Shapley values) to assign each feature an
  importance value for a particular prediction. Unlike LIME which
  provides local linear approximations, SHAP provides theoretically
  grounded feature attributions based on cooperative game theory.

  ## Key Properties

  SHAP values satisfy important properties that LIME doesn't guarantee:

  1. **Additivity**: SHAP values sum to (prediction - baseline)
  2. **Consistency**: If a feature's contribution increases, its SHAP value shouldn't decrease
  3. **Symmetry**: Features with identical contributions get identical SHAP values
  4. **Dummy**: Features that don't affect output get zero SHAP value

  ## Methods

  - `KernelSHAP` - Model-agnostic approximation using weighted regression
  - `LinearSHAP` - Fast exact computation for linear models
  - `SamplingShap` - Fast Monte Carlo approximation via random sampling
  - `TreeSHAP` - Efficient exact computation for tree models (future)

  ## References

  Lundberg, S. M., & Lee, S. I. (2017).
  A Unified Approach to Interpreting Model Predictions. NeurIPS.

  ## Examples

      # Explain with SHAP
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background = [[0.0, 0.0]]

      shap_values = CrucibleXAI.SHAP.explain(instance, background, predict_fn)
      # => %{0 => 2.0, 1 => 3.0}

      # Verify Shapley property: sum = prediction - baseline
      sum = Enum.sum(Map.values(shap_values))
      # sum ≈ 5.0 (prediction) - 0.0 (baseline) = 5.0
  """

  alias CrucibleXAI.SHAP.KernelSHAP
  alias CrucibleXAI.SHAP.LinearSHAP
  alias CrucibleXAI.SHAP.SamplingShap

  require Logger

  @default_opts [
    method: :kernel_shap,
    num_samples: 2000,
    regularization: 0.01
  ]

  @doc """
  Explain instance using SHAP values.

  Computes Shapley values for each feature showing their contribution
  to the prediction relative to a baseline (background data).

  ## Parameters
    * `instance` - Instance to explain (list or Nx.Tensor)
    * `background_data` - Background dataset for baseline (list of instances)
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:method` - SHAP method (default: `:kernel_shap`). Available: `:kernel_shap`, `:linear_shap`, `:sampling_shap`
      * `:num_samples` - Number of samples: coalitions for KernelSHAP, permutations for SamplingShap (default: 2000)
      * `:regularization` - Regularization strength for KernelSHAP (default: 0.01)
      * `:coefficients` - Model coefficients (required for `:linear_shap`)
      * `:intercept` - Model intercept (required for `:linear_shap`)

  ## Returns
    Map of feature_index => shapley_value

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instance = [5.0]
      iex> background = [[0.0]]
      iex> shap = CrucibleXAI.SHAP.explain(instance, background, predict_fn, num_samples: 500)
      iex> is_map(shap)
      true
      iex> map_size(shap)
      1
  """
  @spec explain(list() | Nx.Tensor.t(), list(), function(), keyword()) :: %{integer() => float()}
  def explain(instance, background_data, predict_fn, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)
    method = Keyword.get(opts, :method)

    case method do
      :kernel_shap ->
        KernelSHAP.explain(instance, background_data, predict_fn, opts)

      :linear_shap ->
        # For linear models, extract coefficients and intercept from opts
        coefficients = Keyword.fetch!(opts, :coefficients)
        intercept = Keyword.get(opts, :intercept, 0.0)
        LinearSHAP.explain(instance, background_data, coefficients, intercept)

      :sampling_shap ->
        SamplingShap.explain(instance, background_data, predict_fn, opts)

      _ ->
        raise ArgumentError, "Unknown SHAP method: #{inspect(method)}"
    end
  end

  @doc """
  Explain multiple instances using SHAP.

  ## Parameters
    * `instances` - List of instances to explain
    * `background_data` - Background dataset
    * `predict_fn` - Prediction function
    * `opts` - Options (same as `explain/4` plus batch options):
      * `:parallel` - Enable parallel processing (default: false)
      * `:max_concurrency` - Max concurrent tasks (default: System.schedulers_online())
      * `:timeout` - Timeout per instance in ms (default: 30_000)
      * `:on_error` - Error handling: `:skip` or `:raise` (default: `:raise`)

  ## Returns
    List of maps with SHAP values for each instance

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instances = [[1.0], [2.0], [3.0]]
      iex> background = [[0.0]]
      iex> shap_list = CrucibleXAI.SHAP.explain_batch(instances, background, predict_fn, num_samples: 500)
      iex> length(shap_list)
      3

      # Parallel processing
      iex> shap_list = CrucibleXAI.SHAP.explain_batch(instances, background, predict_fn, num_samples: 500, parallel: true)
      iex> length(shap_list)
      3
  """
  @spec explain_batch(list(), list(), function(), keyword()) :: list(%{integer() => float()})
  def explain_batch(instances, background_data, predict_fn, opts \\ []) do
    parallel = Keyword.get(opts, :parallel, false)

    if parallel do
      explain_batch_parallel(instances, background_data, predict_fn, opts)
    else
      explain_batch_sequential(instances, background_data, predict_fn, opts)
    end
  end

  # Private batch processing functions

  defp explain_batch_sequential(instances, background_data, predict_fn, opts) do
    Enum.map(instances, fn instance ->
      explain(instance, background_data, predict_fn, opts)
    end)
  end

  defp explain_batch_parallel(instances, background_data, predict_fn, opts) do
    max_concurrency = Keyword.get(opts, :max_concurrency, System.schedulers_online())
    timeout = Keyword.get(opts, :timeout, 30_000)
    on_error = Keyword.get(opts, :on_error, :raise)

    instances
    |> Task.async_stream(
      fn instance ->
        try do
          {:ok, explain(instance, background_data, predict_fn, opts)}
        rescue
          e -> {:error, e}
        catch
          :exit, reason -> {:error, {:exit, reason}}
        end
      end,
      max_concurrency: max_concurrency,
      timeout: timeout,
      on_timeout: :kill_task
    )
    |> Enum.reduce([], fn
      {:ok, {:ok, shap_values}}, acc ->
        [shap_values | acc]

      {:ok, {:error, reason}}, acc ->
        case on_error do
          :skip ->
            Logger.warning("SHAP explanation failed: #{inspect(reason)}")
            acc

          :raise ->
            raise "SHAP explanation failed: #{inspect(reason)}"
        end

      {:exit, reason}, acc ->
        case on_error do
          :skip ->
            Logger.warning("SHAP explanation timed out: #{inspect(reason)}")
            acc

          :raise ->
            raise "SHAP explanation timed out: #{inspect(reason)}"
        end
    end)
    |> Enum.reverse()
  end

  @doc """
  Verify SHAP values satisfy the additivity property.

  SHAP values should sum to (prediction - baseline).

  ## Parameters
    * `shap_values` - Map of feature_index => shapley_value
    * `instance` - The instance that was explained
    * `background_data` - Background data used
    * `predict_fn` - Prediction function

  ## Returns
    Boolean indicating if property is satisfied (within tolerance)
  """
  @spec verify_additivity(map(), list(), list(), function(), float()) :: boolean()
  def verify_additivity(shap_values, instance, background_data, predict_fn, tolerance \\ 0.5) do
    prediction = get_prediction(predict_fn, instance)
    baseline = calculate_baseline(background_data, predict_fn)

    shap_sum = Enum.sum(Map.values(shap_values))
    expected = prediction - baseline

    abs(shap_sum - expected) <= tolerance
  end

  # Private helpers

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end

  defp calculate_baseline(background_data, predict_fn) do
    # Baseline is mean prediction on background data
    predictions =
      Enum.map(background_data, fn instance ->
        get_prediction(predict_fn, instance)
      end)

    Enum.sum(predictions) / length(predictions)
  end
end

defmodule CrucibleXAI.SHAP.KernelSHAP do
  @moduledoc """
  KernelSHAP implementation using weighted linear regression.

  KernelSHAP approximates Shapley values by formulating the problem as
  a weighted linear regression on feature coalitions. This provides an
  efficient way to estimate SHAP values without computing all possible
  feature combinations.

  ## Algorithm

  1. Generate random feature coalitions (subsets)
  2. For each coalition, create instance with only those features present
  3. Get predictions for coalition instances
  4. Calculate SHAP kernel weights for each coalition
  5. Solve weighted linear regression to get Shapley values

  ## SHAP Kernel

  The SHAP kernel weights coalitions by:

      weight(S) = (M - 1) / [C(M, |S|) × |S| × (M - |S|)]

  where M is the number of features and |S| is the coalition size.

  ## References

  Lundberg, S. M., & Lee, S. I. (2017).
  A Unified Approach to Interpreting Model Predictions. NeurIPS.

  ## Examples

      iex> predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      iex> instance = [1.0, 1.0]
      iex> background = [[0.0, 0.0]]
      iex> shap = CrucibleXAI.SHAP.KernelSHAP.explain(instance, background, predict_fn, num_samples: 1000)
      iex> is_map(shap)
      true
      iex> map_size(shap)
      2
  """

  require Logger

  @default_opts [
    num_samples: 2000,
    regularization: 0.01,
    max_coalition_size: nil
  ]

  @doc """
  Generate random feature coalitions for SHAP sampling.

  Each coalition is represented as a binary vector where 1 indicates
  the feature is present and 0 indicates it's replaced with background value.

  ## Parameters
    * `n_features` - Number of features
    * `n_samples` - Number of coalitions to generate

  ## Returns
    Nx.Tensor of shape {n_samples, n_features} with binary values

  ## Examples
      iex> coalitions = CrucibleXAI.SHAP.KernelSHAP.generate_coalitions(3, 10)
      iex> Nx.shape(coalitions)
      {10, 3}
  """
  @spec generate_coalitions(pos_integer(), pos_integer()) :: Nx.Tensor.t()
  def generate_coalitions(n_features, n_samples) do
    # Generate random binary coalitions
    key = Nx.Random.key(System.system_time())
    {random_vals, _key} = Nx.Random.uniform(key, shape: {n_samples - 2, n_features})

    # Convert to binary (0 or 1) using threshold 0.5
    random_coalitions = Nx.greater(random_vals, 0.5) |> Nx.as_type(:s64)

    # Always include empty coalition (all 0s) and full coalition (all 1s)
    empty_coalition = Nx.broadcast(0, {1, n_features})
    full_coalition = Nx.broadcast(1, {1, n_features})

    # Concatenate all coalitions
    Nx.concatenate([empty_coalition, full_coalition, random_coalitions], axis: 0)
  end

  @doc """
  Calculate SHAP kernel weights for coalitions.

  The SHAP kernel weight for a coalition S is:

      weight(S) = (M - 1) / [C(M, |S|) × |S| × (M - |S|)]

  where:
  - M is the number of features
  - |S| is the size of the coalition (number of present features)
  - C(M, |S|) is the binomial coefficient "M choose |S|"

  For empty and full coalitions (|S| = 0 or M), we use a large weight
  to approximate infinity.

  ## Parameters
    * `coalitions` - Binary tensor of shape {n_samples, n_features}
    * `n_features` - Total number of features

  ## Returns
    Nx.Tensor of shape {n_samples} with kernel weights

  ## Examples
      iex> coalitions = Nx.tensor([[0, 0], [1, 0], [1, 1]])
      iex> weights = CrucibleXAI.SHAP.KernelSHAP.calculate_shap_kernel_weights(coalitions, 2)
      iex> Nx.shape(weights)
      {3}
  """
  @spec calculate_shap_kernel_weights(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  def calculate_shap_kernel_weights(coalitions, n_features) do
    # Calculate coalition sizes (number of 1s in each row)
    coalition_sizes = Nx.sum(coalitions, axes: [1])

    # For each coalition, calculate the SHAP kernel weight
    weights =
      coalition_sizes
      |> Nx.to_flat_list()
      |> Enum.map(fn size ->
        calculate_single_coalition_weight(size, n_features)
      end)
      |> Nx.tensor()

    weights
  end

  @doc """
  Explain instance using KernelSHAP.

  Computes approximate Shapley values for each feature using weighted
  linear regression on feature coalitions.

  ## Parameters
    * `instance` - Instance to explain (list or Nx.Tensor)
    * `background_data` - Background dataset for baseline (list of instances)
    * `predict_fn` - Prediction function
    * `opts` - Options:
      * `:num_samples` - Number of coalitions to sample (default: 2000)
      * `:regularization` - L2 regularization strength (default: 0.01)

  ## Returns
    Map of feature indices to Shapley values

  ## Examples
      iex> predict_fn = fn [x] -> x * 2.0 end
      iex> instance = [5.0]
      iex> background = [[0.0]]
      iex> shap = CrucibleXAI.SHAP.KernelSHAP.explain(instance, background, predict_fn, num_samples: 500)
      iex> is_map(shap)
      true
  """
  @spec explain(list() | Nx.Tensor.t(), list(), function(), keyword()) :: %{integer() => float()}
  def explain(instance, background_data, predict_fn, opts \\ []) do
    opts = Keyword.merge(@default_opts, opts)

    instance_list = to_list(instance)
    n_features = length(instance_list)
    num_samples = Keyword.get(opts, :num_samples)
    regularization = Keyword.get(opts, :regularization)

    Logger.debug("Starting KernelSHAP explanation for instance: #{inspect(instance_list)}")
    start_time = System.monotonic_time(:millisecond)

    # Step 1: Generate coalitions
    coalitions = generate_coalitions(n_features, num_samples)

    # Step 2: Create instances from coalitions
    # For each coalition, features present use instance value,
    # features absent use background value
    background_mean = calculate_background_mean(background_data)
    coalition_instances = create_coalition_instances(coalitions, instance_list, background_mean)

    # Step 3: Get predictions
    predictions = get_predictions(coalition_instances, predict_fn)

    # Step 4: Calculate SHAP kernel weights
    sample_weights = calculate_shap_kernel_weights(coalitions, n_features)

    # Step 5: Solve weighted linear regression
    # Convert coalitions to float for regression
    coalitions_float = Nx.as_type(coalitions, :f32)

    # Fit weighted linear regression: predictions ~ coalitions
    shap_values = fit_shap_model(coalitions_float, predictions, sample_weights, regularization)

    duration = System.monotonic_time(:millisecond) - start_time
    Logger.info("KernelSHAP explanation completed in #{duration}ms")

    # Convert to map of feature_index => shap_value
    shap_values
    |> Nx.to_flat_list()
    |> Enum.with_index()
    |> Enum.map(fn {value, idx} -> {idx, value} end)
    |> Enum.into(%{})
  end

  # Private helper functions

  defp to_list(data) when is_list(data), do: data
  defp to_list(%Nx.Tensor{} = tensor), do: Nx.to_flat_list(tensor)

  defp calculate_background_mean(background_data) do
    # Calculate mean of background dataset
    background_tensor = Nx.tensor(background_data)
    Nx.mean(background_tensor, axes: [0]) |> Nx.to_flat_list()
  end

  defp create_coalition_instances(coalitions, instance, background_mean) do
    # For each coalition, create instance with selected features
    coalitions_list = Nx.to_list(coalitions)

    instances =
      Enum.map(coalitions_list, fn coalition ->
        Enum.zip([coalition, instance, background_mean])
        |> Enum.map(fn {present, inst_val, bg_val} ->
          if present == 1, do: inst_val, else: bg_val
        end)
      end)

    instances
  end

  defp get_predictions(instances, predict_fn) do
    predictions =
      Enum.map(instances, fn instance ->
        result = predict_fn.(instance)

        case result do
          %Nx.Tensor{} -> Nx.to_number(result)
          num when is_number(num) -> num
        end
      end)

    Nx.tensor(predictions)
  end

  defp fit_shap_model(coalitions, predictions, weights, regularization) do
    # Solve weighted linear regression: predictions = coalitions @ shap_values
    # With L2 regularization for stability

    # Convert to proper shapes
    x = coalitions
    y = predictions
    w = weights

    # Create diagonal weight matrix
    w_diag = Nx.make_diagonal(w)

    # X'W
    xt_w = Nx.dot(Nx.transpose(x), w_diag)

    # X'WX
    xt_w_x = Nx.dot(xt_w, x)

    # Add regularization: X'WX + λI
    {n_features, _} = Nx.shape(Nx.transpose(x))
    regularization_matrix = Nx.multiply(Nx.eye(n_features), regularization)
    xt_w_x_reg = Nx.add(xt_w_x, regularization_matrix)

    # X'Wy
    xt_w_y = Nx.dot(xt_w, y)

    # Solve for SHAP values
    try do
      Nx.LinAlg.solve(xt_w_x_reg, xt_w_y)
    rescue
      _ ->
        # Fallback to pseudoinverse
        Nx.dot(Nx.LinAlg.pinv(xt_w_x), xt_w_y)
    end
  end

  defp calculate_single_coalition_weight(size, n_features) do
    m = n_features
    s = trunc(size)

    cond do
      # Empty coalition or full coalition - use large weight
      s == 0 or s == m ->
        10_000.0

      # Normal coalitions
      true ->
        # weight = (M-1) / [C(M,|S|) × |S| × (M-|S|)]
        binomial_coef = binomial(m, s)
        denominator = binomial_coef * s * (m - s)

        if denominator > 0 do
          (m - 1) / denominator
        else
          10_000.0
        end
    end
  end

  # Calculate binomial coefficient C(n, k) = n! / (k! × (n-k)!)
  defp binomial(n, k) when k > n, do: 0
  defp binomial(_n, k) when k < 0, do: 0
  defp binomial(_n, 0), do: 1
  defp binomial(n, n), do: 1

  defp binomial(n, k) do
    # Use more efficient formula: C(n,k) = n×(n-1)×...×(n-k+1) / k!
    k = min(k, n - k)

    numerator = Enum.reduce((n - k + 1)..n, 1, &*/2)
    denominator = Enum.reduce(1..k, 1, &*/2)

    div(numerator, denominator)
  end
end

defmodule CrucibleXAI.GradientAttribution do
  @moduledoc """
  Gradient-based attribution methods for neural networks.

  These methods use gradients (derivatives) to attribute importance to input features.
  They work best with differentiable models like neural networks.

  ## Methods

  - **Gradient × Input**: Simple multiplication of gradients with input values
  - **Integrated Gradients**: Path integral of gradients from baseline to input
  - **SmoothGrad**: Average of gradients with added noise for smoother attributions

  ## Requirements

  - Model must be differentiable (implemented with Nx)
  - Model function should take Nx tensors as input
  - Model function should return a scalar output

  ## References

  - Shrikumar et al. (2016). "Not Just a Black Box: Learning Important Features Through Propagating Activation Differences"
  - Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks"
  - Smilkov et al. (2017). "SmoothGrad: removing noise by adding noise"

  ## Examples

      # Simple gradient × input
      model_fn = fn params -> Nx.sum(Nx.pow(params, 2)) end
      instance = Nx.tensor([3.0, 4.0])

      attributions = CrucibleXAI.GradientAttribution.gradient_x_input(model_fn, instance)
      # => Tensor with attribution scores for each feature
  """

  @doc """
  Compute Gradient × Input attribution.

  This is the simplest gradient-based attribution method:
  attribution_i = (∂f/∂x_i) * x_i

  ## Parameters
    * `model_fn` - Differentiable model function that takes Nx tensor and returns scalar
    * `instance` - Input instance as Nx tensor

  ## Returns
    Nx tensor with attribution scores (same shape as input)

  ## Examples

      iex> model_fn = fn params -> Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0]))) end
      iex> instance = Nx.tensor([5.0, 4.0])
      iex> attrs = CrucibleXAI.GradientAttribution.gradient_x_input(model_fn, instance)
      iex> Nx.shape(attrs)
      {2}
  """
  @spec gradient_x_input(function(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def gradient_x_input(model_fn, instance) do
    # Compute gradients
    gradients = compute_gradients(model_fn, instance)

    # Multiply gradients by input values
    Nx.multiply(gradients, instance)
  end

  @doc """
  Compute gradients of model output with respect to input.

  Uses Nx's automatic differentiation to compute ∂f/∂x_i for each input feature.

  ## Parameters
    * `model_fn` - Differentiable model function
    * `instance` - Input instance as Nx tensor

  ## Returns
    Nx tensor with gradient values (same shape as input)

  ## Examples

      iex> model_fn = fn params -> Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0]))) end
      iex> instance = Nx.tensor([1.0, 1.0])
      iex> grads = CrucibleXAI.GradientAttribution.compute_gradients(model_fn, instance)
      iex> Nx.shape(grads)
      {2}
  """
  @spec compute_gradients(function(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def compute_gradients(model_fn, instance) do
    # Use Nx.Defn.grad to compute gradients
    grad_fn = Nx.Defn.grad(instance, fn x -> model_fn.(x) end)
    grad_fn
  end

  @doc """
  Compute Integrated Gradients attribution.

  Integrated Gradients computes the attribution by integrating gradients
  along a straight path from a baseline to the input. This method satisfies
  important axioms like completeness and sensitivity.

  The formula:
  IG_i = (x_i - baseline_i) * ∫₀¹ (∂f/∂x_i)(baseline + α(x - baseline)) dα

  In practice, the integral is approximated using a Riemann sum with
  discrete steps.

  ## Parameters
    * `model_fn` - Differentiable model function
    * `instance` - Input instance as Nx tensor
    * `baseline` - Baseline instance as Nx tensor (typically zeros)
    * `opts` - Options:
      * `:steps` - Number of interpolation steps (default: 50)

  ## Returns
    Nx tensor with attribution scores (same shape as input)

  ## Examples

      iex> model_fn = fn params -> Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0]))) end
      iex> instance = Nx.tensor([5.0, 4.0])
      iex> baseline = Nx.tensor([0.0, 0.0])
      iex> attrs = CrucibleXAI.GradientAttribution.integrated_gradients(model_fn, instance, baseline, steps: 50)
      iex> Nx.shape(attrs)
      {2}
  """
  @spec integrated_gradients(function(), Nx.Tensor.t(), Nx.Tensor.t(), keyword()) ::
          Nx.Tensor.t()
  def integrated_gradients(model_fn, instance, baseline, opts \\ []) do
    steps = Keyword.get(opts, :steps, 50)

    # Generate interpolation alphas from 0 to 1
    alphas = Nx.linspace(0.0, 1.0, n: steps + 1)

    # Compute interpolated inputs: baseline + alpha * (instance - baseline)
    diff = Nx.subtract(instance, baseline)

    # Compute gradients at each interpolated point
    gradients_list =
      alphas
      |> Nx.to_flat_list()
      |> Enum.map(fn alpha ->
        # interpolated = baseline + alpha * (instance - baseline)
        interpolated = Nx.add(baseline, Nx.multiply(alpha, diff))
        compute_gradients(model_fn, interpolated)
      end)

    # Stack gradients and compute average
    gradients_stacked = Nx.stack(gradients_list)
    avg_gradients = Nx.mean(gradients_stacked, axes: [0])

    # Integrated Gradients: (instance - baseline) * avg_gradients
    Nx.multiply(diff, avg_gradients)
  end

  @doc """
  Compute SmoothGrad attribution.

  SmoothGrad computes attributions by averaging gradient × input over
  multiple noisy versions of the input. This reduces noise and creates
  smoother, more visually coherent attribution maps.

  The algorithm:
  1. Generate n_samples noisy versions of the input
  2. Compute gradient × input for each noisy sample
  3. Average the attributions

  ## Parameters
    * `model_fn` - Differentiable model function
    * `instance` - Input instance as Nx tensor
    * `opts` - Options:
      * `:noise_level` - Standard deviation of Gaussian noise (default: 0.15)
      * `:n_samples` - Number of noisy samples to average (default: 50)

  ## Returns
    Nx tensor with smoothed attribution scores (same shape as input)

  ## Examples

      iex> model_fn = fn params -> Nx.sum(Nx.pow(params, 2)) end
      iex> instance = Nx.tensor([3.0, 4.0])
      iex> attrs = CrucibleXAI.GradientAttribution.smooth_grad(model_fn, instance, noise_level: 0.1, n_samples: 30)
      iex> Nx.shape(attrs)
      {2}
  """
  @spec smooth_grad(function(), Nx.Tensor.t(), keyword()) :: Nx.Tensor.t()
  def smooth_grad(model_fn, instance, opts \\ []) do
    noise_level = Keyword.get(opts, :noise_level, 0.15)
    n_samples = Keyword.get(opts, :n_samples, 50)

    # Get random key for reproducible noise generation
    key = Nx.Random.key(System.system_time(:nanosecond))

    # Generate noisy samples and compute gradient × input for each
    {attributions_list, _final_key} =
      Enum.map_reduce(1..n_samples, key, fn _, current_key ->
        # Generate Gaussian noise
        {noise, next_key} =
          Nx.Random.normal(current_key, 0.0, noise_level, shape: Nx.shape(instance))

        # Add noise to instance
        noisy_instance = Nx.add(instance, noise)

        # Compute gradient × input for noisy sample
        attribution = gradient_x_input(model_fn, noisy_instance)

        {attribution, next_key}
      end)

    # Stack and average attributions
    attributions_stacked = Nx.stack(attributions_list)
    Nx.mean(attributions_stacked, axes: [0])
  end
end

defmodule CrucibleXAI.LIME.Kernels do
  @moduledoc """
  Kernel functions for sample weighting in LIME.

  Kernels measure proximity between samples and the instance being explained.
  Samples closer to the instance receive higher weights, ensuring that the
  local interpretable model focuses on the local region around the prediction.

  ## Available Kernels

  - **Exponential**: Decreases exponentially with distance (most commonly used)
  - **Cosine**: Based on cosine similarity, smooth transition
  - **RBF**: Radial basis function (alias for exponential)

  ## Examples

      iex> distances = Nx.tensor([0.0, 1.0, 2.0])
      iex> weights = CrucibleXAI.LIME.Kernels.exponential(distances, 1.0)
      iex> Nx.to_list(weights) |> Enum.map(&Float.round(&1, 6))
      [1.0, 0.367879, 0.018316]

      iex> distances = Nx.tensor([0.0, 0.5, 1.0])
      iex> weights = CrucibleXAI.LIME.Kernels.cosine(distances)
      iex> Nx.to_list(weights) |> Enum.map(&Float.round(&1, 2))
      [1.0, 0.5, 0.0]
  """

  @type kernel_fn :: (Nx.Tensor.t(), float() -> Nx.Tensor.t())

  @default_kernel_width 0.75

  @doc """
  Exponential kernel: exp(-d²/width²)

  Gives exponentially decreasing weight as distance increases.
  This is the default kernel used in LIME implementations.

  ## Parameters
    * `distances` - Nx.Tensor of distances from instance
    * `kernel_width` - Controls decay rate (larger = slower decay, default: 0.75)

  ## Returns
    Nx.Tensor of weights in range [0, 1]

  ## Examples
      iex> distances = Nx.tensor([0.0, 0.5, 1.0, 2.0])
      iex> weights = CrucibleXAI.LIME.Kernels.exponential(distances, 1.0)
      iex> [w0, w1, w2, w3] = Nx.to_flat_list(weights)
      iex> Float.round(w0, 2)
      1.0
      iex> w1 > w2 and w2 > w3
      true
  """
  @spec exponential(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def exponential(distances, kernel_width \\ @default_kernel_width) do
    distances
    |> Nx.pow(2)
    |> Nx.divide(kernel_width ** 2)
    |> Nx.negate()
    |> Nx.exp()
  end

  @doc """
  Cosine similarity kernel: (1 + cos(πd)) / 2

  Provides smooth transition from 1 to 0 as distance increases from 0 to 1.
  Assumes distances are normalized to [0, 1] range.

  ## Parameters
    * `distances` - Nx.Tensor of normalized distances in [0, 1]

  ## Returns
    Nx.Tensor of weights in range [0, 1]

  ## Examples
      iex> distances = Nx.tensor([0.0, 0.5, 1.0])
      iex> weights = CrucibleXAI.LIME.Kernels.cosine(distances)
      iex> [w0, w1, w2] = Nx.to_flat_list(weights)
      iex> Float.round(w0, 2)
      1.0
      iex> Float.round(w1, 2)
      0.5
      iex> Float.round(w2, 2)
      0.0
  """
  @spec cosine(Nx.Tensor.t()) :: Nx.Tensor.t()
  def cosine(distances) do
    distances
    |> Nx.multiply(:math.pi())
    |> Nx.cos()
    |> Nx.add(1)
    |> Nx.divide(2)
  end

  @doc """
  Radial Basis Function (RBF) kernel - alias for exponential kernel.

  ## Parameters
    * `distances` - Nx.Tensor of distances
    * `kernel_width` - Bandwidth parameter (default: 0.75)

  ## Returns
    Nx.Tensor of weights in range [0, 1]
  """
  @spec rbf(Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def rbf(distances, kernel_width \\ @default_kernel_width) do
    exponential(distances, kernel_width)
  end

  @doc """
  Calculate Euclidean distances between samples and instance.

  Computes L2 distance: sqrt(Σ(x_i - y_i)²)

  ## Parameters
    * `samples` - Nx.Tensor of shape {n_samples, n_features}
    * `instance` - List or Nx.Tensor of shape {n_features}

  ## Returns
    Nx.Tensor of shape {n_samples} with distances

  ## Examples
      iex> samples = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      iex> instance = [0, 0]
      iex> distances = CrucibleXAI.LIME.Kernels.euclidean_distance(samples, instance)
      iex> Nx.shape(distances)
      {3}
      iex> Nx.to_list(distances) |> Enum.map(&Float.round(&1, 2))
      [2.24, 5.0, 7.81]
  """
  @spec euclidean_distance(Nx.Tensor.t(), list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def euclidean_distance(samples, instance) do
    instance_tensor = to_tensor(instance)

    samples
    |> Nx.subtract(instance_tensor)
    |> Nx.pow(2)
    |> Nx.sum(axes: [1])
    |> Nx.sqrt()
  end

  @doc """
  Calculate Manhattan (L1) distances between samples and instance.

  Computes L1 distance: Σ|x_i - y_i|

  Also known as taxicab distance or city block distance.

  ## Parameters
    * `samples` - Nx.Tensor of shape {n_samples, n_features}
    * `instance` - List or Nx.Tensor of shape {n_features}

  ## Returns
    Nx.Tensor of shape {n_samples} with distances

  ## Examples
      iex> samples = Nx.tensor([[3.0, 4.0], [0.0, 0.0], [1.0, 1.0]])
      iex> instance = [0.0, 0.0]
      iex> distances = CrucibleXAI.LIME.Kernels.manhattan_distance(samples, instance)
      iex> Nx.to_list(distances)
      [7.0, 0.0, 2.0]
  """
  @spec manhattan_distance(Nx.Tensor.t(), list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def manhattan_distance(samples, instance) do
    instance_tensor = to_tensor(instance)

    samples
    |> Nx.subtract(instance_tensor)
    |> Nx.abs()
    |> Nx.sum(axes: [1])
  end

  @doc """
  Calculate cosine distance between samples and instance.

  Cosine distance = 1 - cosine similarity
  Cosine similarity = (A · B) / (||A|| × ||B||)

  ## Parameters
    * `samples` - Nx.Tensor of shape {n_samples, n_features}
    * `instance` - List or Nx.Tensor of shape {n_features}

  ## Returns
    Nx.Tensor of shape {n_samples} with distances in [0, 2]
  """
  @spec cosine_distance(Nx.Tensor.t(), list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def cosine_distance(samples, instance) do
    instance_tensor = to_tensor(instance)

    # Compute dot product
    dot_product = Nx.dot(samples, instance_tensor)

    # Compute norms
    sample_norms = Nx.sqrt(Nx.sum(Nx.pow(samples, 2), axes: [1]))
    instance_norm = Nx.sqrt(Nx.sum(Nx.pow(instance_tensor, 2)))

    # Cosine similarity
    cosine_sim = Nx.divide(dot_product, Nx.multiply(sample_norms, instance_norm))

    # Cosine distance
    Nx.subtract(1.0, cosine_sim)
  end

  @doc """
  Calculate pairwise distances between samples and instance with a specified metric.

  ## Parameters
    * `samples` - Nx.Tensor of shape {n_samples, n_features}
    * `instance` - List or Nx.Tensor of shape {n_features}
    * `metric` - Distance metric (:euclidean, :manhattan, :cosine)

  ## Returns
    Nx.Tensor of shape {n_samples} with distances
  """
  @spec calculate_distances(Nx.Tensor.t(), list() | Nx.Tensor.t(), atom()) :: Nx.Tensor.t()
  def calculate_distances(samples, instance, metric \\ :euclidean) do
    case metric do
      :euclidean -> euclidean_distance(samples, instance)
      :manhattan -> manhattan_distance(samples, instance)
      :cosine -> cosine_distance(samples, instance)
      _ -> raise ArgumentError, "Unknown metric: #{inspect(metric)}"
    end
  end

  @doc """
  Apply kernel function to distances.

  ## Parameters
    * `distances` - Nx.Tensor of distances
    * `kernel` - Kernel function (:exponential, :cosine, :rbf)
    * `opts` - Kernel-specific options

  ## Returns
    Nx.Tensor of weights
  """
  @spec apply_kernel(Nx.Tensor.t(), atom(), keyword()) :: Nx.Tensor.t()
  def apply_kernel(distances, kernel \\ :exponential, opts \\ []) do
    case kernel do
      :exponential ->
        kernel_width = Keyword.get(opts, :kernel_width, @default_kernel_width)
        exponential(distances, kernel_width)

      :rbf ->
        kernel_width = Keyword.get(opts, :kernel_width, @default_kernel_width)
        rbf(distances, kernel_width)

      :cosine ->
        cosine(distances)

      _ ->
        raise ArgumentError, "Unknown kernel: #{inspect(kernel)}"
    end
  end

  # Private helper functions

  @spec to_tensor(list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  defp to_tensor(data) when is_list(data), do: Nx.tensor(data)
  defp to_tensor(%Nx.Tensor{} = tensor), do: tensor
end

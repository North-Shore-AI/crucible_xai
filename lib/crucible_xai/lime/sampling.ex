defmodule CrucibleXAI.LIME.Sampling do
  @moduledoc """
  Sampling strategies for LIME.

  Generates perturbed samples around an instance for local approximation.
  Different sampling strategies are provided for different feature types:

  - `gaussian/3`: For continuous features using Gaussian perturbation
  - `uniform/3`: For continuous features using uniform perturbation
  - `categorical/3`: For categorical features
  - `combined/3`: For mixed feature types

  ## Examples

      iex> instance = [1.0, 2.0, 3.0]
      iex> samples = CrucibleXAI.LIME.Sampling.gaussian(instance, 100)
      iex> Nx.shape(samples)
      {100, 3}

      iex> instance = [5.0, 5.0]
      iex> samples = CrucibleXAI.LIME.Sampling.uniform(instance, 50, range: 1.0)
      iex> Nx.shape(samples)
      {50, 2}
  """

  @default_std 1.0
  @default_range 1.0

  @doc """
  Generate perturbed samples using Gaussian perturbation.

  Each feature is perturbed independently by adding Gaussian noise
  scaled by the feature's standard deviation in the training data.

  ## Parameters
    * `instance` - The instance to perturb (list or Nx.Tensor)
    * `n_samples` - Number of samples to generate
    * `opts` - Options:
      * `:feature_stats` - Map with `:std_devs` key containing list of standard deviations
      * `:scale` - Noise scale factor (default: 1.0)
      * `:key` - Random key for reproducibility (optional)

  ## Returns
    Nx.Tensor of shape {n_samples, n_features}

  ## Examples
      iex> instance = [1.0, 2.0, 3.0]
      iex> stats = %{std_devs: [0.5, 0.3, 0.8]}
      iex> samples = CrucibleXAI.LIME.Sampling.gaussian(instance, 100, feature_stats: stats)
      iex> Nx.shape(samples)
      {100, 3}
  """
  @spec gaussian(list() | Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def gaussian(instance, n_samples, opts \\ []) do
    instance_tensor = prepare_instance(instance)
    n_features = Nx.size(instance_tensor)

    # Generate base samples (broadcast instance to n_samples rows)
    base_samples = generate_base_samples(instance_tensor, n_samples)

    # Generate and add Gaussian noise
    add_gaussian_noise(base_samples, instance, n_features, opts)
  end

  @doc """
  Generate samples using uniform perturbation.

  Each feature is perturbed by adding uniform random noise within a specified range.

  ## Parameters
    * `instance` - The instance to perturb (list or Nx.Tensor)
    * `n_samples` - Number of samples to generate
    * `opts` - Options:
      * `:range` - Uniform range around instance (default: 1.0)
      * `:feature_ranges` - List of ranges per feature
      * `:key` - Random key for reproducibility (optional)

  ## Returns
    Nx.Tensor of shape {n_samples, n_features}

  ## Examples
      iex> instance = [5.0, 5.0]
      iex> samples = CrucibleXAI.LIME.Sampling.uniform(instance, 100, range: 2.0)
      iex> Nx.shape(samples)
      {100, 2}
  """
  @spec uniform(list() | Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def uniform(instance, n_samples, opts \\ []) do
    instance_tensor = prepare_instance(instance)
    n_features = Nx.size(instance_tensor)

    # Get range parameters
    range = Keyword.get(opts, :range, @default_range)

    # Generate uniform noise in [-range, +range]
    noise_shape = {n_samples, n_features}
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))
    {uniform_noise, _new_key} = Nx.Random.uniform(key, 0.0, 1.0, shape: noise_shape)

    # Scale to [-range, +range]
    scaled_noise = Nx.subtract(Nx.multiply(uniform_noise, 2.0 * range), range)

    # Add to instance
    base_samples = generate_base_samples(instance_tensor, n_samples)
    Nx.add(base_samples, scaled_noise)
  end

  @doc """
  Generate samples for categorical features.

  Samples from observed feature values in the training data.

  ## Parameters
    * `instance` - The instance to perturb (list of categorical values)
    * `n_samples` - Number of samples to generate
    * `opts` - Options:
      * `:possible_values` - Map of feature_index => list of possible values
      * `:key` - Random key for reproducibility (optional)

  ## Returns
    Nx.Tensor of shape {n_samples, n_features}

  ## Examples
      iex> instance = [0, 1, 2]
      iex> possible = %{0 => [0, 1, 2], 1 => [0, 1], 2 => [0, 1, 2, 3]}
      iex> samples = CrucibleXAI.LIME.Sampling.categorical(instance, 100, possible_values: possible)
      iex> Nx.shape(samples)
      {100, 3}
  """
  @spec categorical(list(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def categorical(instance, n_samples, opts \\ []) do
    possible_values = Keyword.get(opts, :possible_values, %{})
    n_features = length(instance)

    # Generate samples by randomly selecting from possible values for each feature
    samples =
      for _ <- 1..n_samples do
        for feature_idx <- 0..(n_features - 1) do
          values = Map.get(possible_values, feature_idx, [Enum.at(instance, feature_idx)])
          Enum.random(values)
        end
      end

    Nx.tensor(samples)
  end

  @doc """
  Combined sampling for mixed feature types.

  Handles both continuous (Gaussian) and categorical (sampling) features.

  ## Parameters
    * `instance` - The instance to perturb
    * `n_samples` - Number of samples to generate
    * `opts` - Options:
      * `:continuous_features` - List of continuous feature indices
      * `:categorical_features` - List of categorical feature indices
      * `:possible_values` - Map of categorical feature values
      * `:feature_stats` - Statistics for continuous features

  ## Returns
    Nx.Tensor of shape {n_samples, n_features}
  """
  @spec combined(list(), pos_integer(), keyword()) :: Nx.Tensor.t()
  def combined(instance, n_samples, opts \\ []) do
    continuous_indices = Keyword.get(opts, :continuous_features, [])
    categorical_indices = Keyword.get(opts, :categorical_features, [])

    # Generate continuous features
    continuous_instance = Enum.map(continuous_indices, &Enum.at(instance, &1))

    continuous_samples =
      if length(continuous_instance) > 0 do
        gaussian(continuous_instance, n_samples, opts)
      else
        Nx.tensor([])
      end

    # Generate categorical features
    categorical_instance = Enum.map(categorical_indices, &Enum.at(instance, &1))

    categorical_samples =
      if length(categorical_instance) > 0 do
        categorical(categorical_instance, n_samples, opts)
      else
        Nx.tensor([])
      end

    # Combine samples
    if Nx.size(continuous_samples) > 0 and Nx.size(categorical_samples) > 0 do
      Nx.concatenate([continuous_samples, categorical_samples], axis: 1)
    else
      if Nx.size(continuous_samples) > 0, do: continuous_samples, else: categorical_samples
    end
  end

  # Private helper functions

  @spec prepare_instance(list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  defp prepare_instance(instance) when is_list(instance), do: Nx.tensor(instance)
  defp prepare_instance(%Nx.Tensor{} = instance), do: instance

  @spec generate_base_samples(Nx.Tensor.t(), pos_integer()) :: Nx.Tensor.t()
  defp generate_base_samples(instance_tensor, n_samples) do
    n_features = Nx.size(instance_tensor)
    Nx.broadcast(instance_tensor, {n_samples, n_features})
  end

  @spec add_gaussian_noise(Nx.Tensor.t(), list() | Nx.Tensor.t(), pos_integer(), keyword()) ::
          Nx.Tensor.t()
  defp add_gaussian_noise(base_samples, instance, n_features, opts) do
    {n_samples, _} = Nx.shape(base_samples)

    # Build standard deviation tensor
    std_tensor = build_std_tensor(instance, n_features, opts)

    # Generate Gaussian noise
    key = Keyword.get(opts, :key, Nx.Random.key(System.system_time()))
    {noise, _new_key} = Nx.Random.normal(key, 0.0, 1.0, shape: {n_samples, n_features})

    # Scale noise by standard deviations
    scaled_noise = Nx.multiply(noise, std_tensor)

    # Add to base samples
    Nx.add(base_samples, scaled_noise)
  end

  @spec build_std_tensor(list() | Nx.Tensor.t(), pos_integer(), keyword()) :: Nx.Tensor.t()
  defp build_std_tensor(_instance, n_features, opts) do
    scale = Keyword.get(opts, :scale, 1.0)

    std_devs =
      case Keyword.get(opts, :feature_stats) do
        %{std_devs: stds} -> stds
        _ -> List.duplicate(@default_std, n_features)
      end

    std_devs
    |> Nx.tensor()
    |> Nx.multiply(scale)
  end
end

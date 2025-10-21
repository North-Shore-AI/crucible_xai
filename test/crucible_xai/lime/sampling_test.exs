defmodule CrucibleXAI.LIME.SamplingTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.LIME.Sampling

  describe "gaussian/3" do
    test "generates correct number of samples" do
      instance = [1.0, 2.0, 3.0]
      n_samples = 100

      samples = Sampling.gaussian(instance, n_samples)

      assert Nx.shape(samples) == {100, 3}
    end

    test "perturbed samples are normally distributed around instance" do
      instance = [1.0, 2.0, 3.0]
      n_samples = 1000
      stats = %{std_devs: [0.5, 0.5, 0.5]}

      samples = Sampling.gaussian(instance, n_samples, feature_stats: stats)

      # Check mean is close to instance
      means = Nx.mean(samples, axes: [0]) |> Nx.to_flat_list()

      Enum.zip(instance, means)
      |> Enum.each(fn {expected, actual} ->
        assert_in_delta expected, actual, 0.1
      end)
    end

    test "respects scale parameter" do
      instance = [1.0, 2.0, 3.0]
      stats = %{std_devs: [1.0, 1.0, 1.0]}

      samples_scale_1 = Sampling.gaussian(instance, 1000, feature_stats: stats, scale: 1.0)
      samples_scale_2 = Sampling.gaussian(instance, 1000, feature_stats: stats, scale: 2.0)

      std_1 = Nx.standard_deviation(samples_scale_1, axes: [0]) |> Nx.to_flat_list()
      std_2 = Nx.standard_deviation(samples_scale_2, axes: [0]) |> Nx.to_flat_list()

      # Scale 2 should have ~2x the standard deviation
      Enum.zip(std_1, std_2)
      |> Enum.each(fn {s1, s2} ->
        assert_in_delta s2 / s1, 2.0, 0.3
      end)
    end

    test "handles list input" do
      instance = [1.0, 2.0]
      samples = Sampling.gaussian(instance, 10)

      assert Nx.shape(samples) == {10, 2}
    end

    test "handles Nx.Tensor input" do
      instance = Nx.tensor([1.0, 2.0])
      samples = Sampling.gaussian(instance, 10)

      assert Nx.shape(samples) == {10, 2}
    end

    test "uses default std_devs when not provided" do
      instance = [1.0, 2.0, 3.0]
      samples = Sampling.gaussian(instance, 100)

      assert Nx.shape(samples) == {100, 3}
    end
  end

  describe "uniform/3" do
    test "generates correct number of samples" do
      instance = [1.0, 2.0, 3.0]
      n_samples = 100

      samples = Sampling.uniform(instance, n_samples)

      assert Nx.shape(samples) == {100, 3}
    end

    test "samples are uniformly distributed around instance" do
      instance = [5.0, 5.0, 5.0]
      n_samples = 1000
      range = 2.0

      samples = Sampling.uniform(instance, n_samples, range: range)

      # Check samples are within expected range
      mins = Nx.reduce_min(samples, axes: [0]) |> Nx.to_flat_list()
      maxs = Nx.reduce_max(samples, axes: [0]) |> Nx.to_flat_list()

      Enum.zip([mins, maxs])
      |> Enum.each(fn {min_val, max_val} ->
        assert min_val >= 5.0 - range
        assert max_val <= 5.0 + range
      end)
    end
  end

  describe "categorical/3" do
    test "generates correct number of samples" do
      instance = [0, 1, 2]
      n_samples = 100
      possible_values = %{0 => [0, 1, 2], 1 => [0, 1], 2 => [0, 1, 2, 3]}

      samples = Sampling.categorical(instance, n_samples, possible_values: possible_values)

      assert Nx.shape(samples) == {100, 3}
    end

    test "samples only contain valid categorical values" do
      instance = [0, 1]
      n_samples = 100
      possible_values = %{0 => [0, 1, 2], 1 => [0, 1]}

      samples = Sampling.categorical(instance, n_samples, possible_values: possible_values)

      # Convert to list and check all values are valid
      samples_list = Nx.to_list(samples)

      Enum.each(samples_list, fn row ->
        [val0, val1] = row
        assert val0 in [0, 1, 2]
        assert val1 in [0, 1]
      end)
    end
  end

  # Property-based tests
  property "gaussian sampling produces correct shape" do
    check all(
            n_features <- integer(1..20),
            n_samples <- integer(1..100)
          ) do
      instance = List.duplicate(0.0, n_features)
      samples = Sampling.gaussian(instance, n_samples)

      assert Nx.shape(samples) == {n_samples, n_features}
    end
  end

  property "gaussian samples have finite values" do
    check all(
            instance <- list_of(float(), min_length: 1, max_length: 10),
            n_samples <- integer(1..50)
          ) do
      samples = Sampling.gaussian(instance, n_samples)
      samples_list = Nx.to_flat_list(samples)

      assert Enum.all?(samples_list, fn x ->
               is_number(x) and not is_nan(x) and not is_infinity(x)
             end)
    end
  end

  defp is_nan(x), do: x != x
  defp is_infinity(x), do: x == :infinity or x == :neg_infinity or abs(x) > 1.0e308
end

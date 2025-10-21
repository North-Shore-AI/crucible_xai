defmodule CrucibleXAI.LIME.KernelsTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.LIME.Kernels

  doctest Kernels

  describe "exponential/2" do
    test "returns 1.0 for zero distance" do
      distances = Nx.tensor([0.0])
      weights = Kernels.exponential(distances, 1.0)

      assert_in_delta Nx.to_number(weights[0]), 1.0, 1.0e-6
    end

    test "decreases with increasing distance" do
      distances = Nx.tensor([0.0, 1.0, 2.0, 3.0])
      weights = Kernels.exponential(distances, 1.0)
      weights_list = Nx.to_flat_list(weights)

      # Weights should be monotonically decreasing
      assert weights_list == Enum.sort(weights_list, :desc)
    end

    test "kernel_width affects decay rate" do
      distances = Nx.tensor([1.0])

      w1 = Kernels.exponential(distances, 0.5)[0] |> Nx.to_number()
      w2 = Kernels.exponential(distances, 1.0)[0] |> Nx.to_number()

      # Smaller width = faster decay = smaller weight
      assert w1 < w2
    end

    test "handles default kernel width" do
      distances = Nx.tensor([0.0, 1.0])
      weights = Kernels.exponential(distances)

      assert Nx.shape(weights) == {2}
    end

    test "handles vector of distances" do
      distances = Nx.tensor([0.0, 0.5, 1.0, 1.5, 2.0])
      weights = Kernels.exponential(distances, 1.0)

      assert Nx.shape(weights) == {5}
      # First weight should be 1.0
      assert_in_delta Nx.to_number(weights[0]), 1.0, 1.0e-6
    end
  end

  describe "cosine/1" do
    test "returns 1.0 for zero distance" do
      distances = Nx.tensor([0.0])
      weights = Kernels.cosine(distances)

      assert_in_delta Nx.to_number(weights[0]), 1.0, 1.0e-6
    end

    test "returns 0.0 for distance of 1.0" do
      distances = Nx.tensor([1.0])
      weights = Kernels.cosine(distances)

      assert_in_delta Nx.to_number(weights[0]), 0.0, 1.0e-6
    end

    test "returns 0.5 for distance of 0.5" do
      distances = Nx.tensor([0.5])
      weights = Kernels.cosine(distances)

      assert_in_delta Nx.to_number(weights[0]), 0.5, 1.0e-6
    end

    test "decreases monotonically" do
      distances = Nx.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
      weights = Kernels.cosine(distances)
      weights_list = Nx.to_flat_list(weights)

      assert weights_list == Enum.sort(weights_list, :desc)
    end
  end

  describe "euclidean_distance/2" do
    test "calculates correct distances" do
      samples = Nx.tensor([[3, 4], [0, 0], [5, 12]])
      instance = [0, 0]

      distances = Kernels.euclidean_distance(samples, instance)
      expected = [5.0, 0.0, 13.0]

      Enum.zip(Nx.to_flat_list(distances), expected)
      |> Enum.each(fn {actual, exp} ->
        assert_in_delta actual, exp, 1.0e-6
      end)
    end

    test "handles list and tensor inputs" do
      samples = Nx.tensor([[1, 2]])

      dist1 = Kernels.euclidean_distance(samples, [0, 0])
      dist2 = Kernels.euclidean_distance(samples, Nx.tensor([0, 0]))

      assert_in_delta Nx.to_number(dist1[0]), Nx.to_number(dist2[0]), 1.0e-6
    end

    test "distance from instance to itself is zero" do
      samples = Nx.tensor([[1.0, 2.0, 3.0]])
      instance = [1.0, 2.0, 3.0]

      distances = Kernels.euclidean_distance(samples, instance)

      assert_in_delta Nx.to_number(distances[0]), 0.0, 1.0e-6
    end

    test "returns correct shape" do
      samples = Nx.tensor([[1, 2], [3, 4], [5, 6]])
      instance = [0, 0]

      distances = Kernels.euclidean_distance(samples, instance)

      assert Nx.shape(distances) == {3}
    end
  end

  describe "manhattan_distance/2" do
    test "calculates correct L1 distances" do
      samples = Nx.tensor([[3, 4], [0, 0], [1, 1]])
      instance = [0, 0]

      distances = Kernels.manhattan_distance(samples, instance)
      expected = [7.0, 0.0, 2.0]

      Enum.zip(Nx.to_flat_list(distances), expected)
      |> Enum.each(fn {actual, exp} ->
        assert_in_delta actual, exp, 1.0e-6
      end)
    end

    test "handles negative coordinates" do
      samples = Nx.tensor([[-1, -1], [1, 1]])
      instance = [0, 0]

      distances = Kernels.manhattan_distance(samples, instance)
      expected = [2.0, 2.0]

      Enum.zip(Nx.to_flat_list(distances), expected)
      |> Enum.each(fn {actual, exp} ->
        assert_in_delta actual, exp, 1.0e-6
      end)
    end
  end

  # Property-based tests
  property "exponential kernel weights are in [0, 1]" do
    check all(
            distances <- list_of(float(min: 0.0, max: 100.0), min_length: 1, max_length: 100),
            kernel_width <- float(min: 0.01, max: 10.0)
          ) do
      distances_tensor = Nx.tensor(distances)
      weights = Kernels.exponential(distances_tensor, kernel_width)
      weights_list = Nx.to_flat_list(weights)

      assert Enum.all?(weights_list, fn w -> w >= 0.0 and w <= 1.0 end)
    end
  end

  property "kernel weight decreases with distance" do
    check all(kernel_width <- float(min: 0.1, max: 5.0)) do
      distances = Nx.tensor([0.0, 1.0, 2.0, 3.0])
      weights = Kernels.exponential(distances, kernel_width)
      weights_list = Nx.to_flat_list(weights)

      # Weights should be monotonically decreasing
      assert weights_list == Enum.sort(weights_list, :desc)
    end
  end

  property "zero distance gives weight of 1.0" do
    check all(kernel_width <- float(min: 0.01, max: 10.0)) do
      distances = Nx.tensor([0.0])
      weights = Kernels.exponential(distances, kernel_width)

      assert_in_delta Nx.to_number(weights[0]), 1.0, 1.0e-6
    end
  end

  property "euclidean distance is non-negative" do
    check all(
            samples <- list_of(list_of(float(), length: 3), min_length: 1, max_length: 50),
            instance <- list_of(float(), length: 3)
          ) do
      samples_tensor = Nx.tensor(samples)
      distances = Kernels.euclidean_distance(samples_tensor, instance)
      distances_list = Nx.to_flat_list(distances)

      assert Enum.all?(distances_list, fn d -> d >= 0.0 end)
    end
  end

  property "cosine kernel weights are in [0, 1]" do
    check all(distances <- list_of(float(min: 0.0, max: 1.0), min_length: 1, max_length: 100)) do
      distances_tensor = Nx.tensor(distances)
      weights = Kernels.cosine(distances_tensor)
      weights_list = Nx.to_flat_list(weights)

      assert Enum.all?(weights_list, fn w -> w >= -0.001 and w <= 1.001 end)
    end
  end
end

defmodule CrucibleXAI.Global.InteractionTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.Global.Interaction

  describe "h_statistic/4" do
    test "detects no interaction in additive model" do
      # Model: f(x, y) = 2*x + 3*y (purely additive, no interaction)
      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      data = for i <- 1..20, do: [i * 1.0, i * 0.5]

      h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 10)

      # H should be close to 0 (no interaction)
      assert is_float(h_stat)
      assert h_stat >= 0.0
      assert h_stat <= 1.0
      # Very small interaction
      assert h_stat < 0.2
    end

    test "detects strong interaction in multiplicative model" do
      # Model: f(x, y) = x * y (pure interaction)
      predict_fn = fn [x, y] -> x * y end

      data = for i <- 1..20, j <- 1..3, do: [i * 1.0, j * 1.0]

      h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 10)

      # H should be measurable (interaction exists)
      # Note: H-statistic magnitude depends on data distribution
      assert is_float(h_stat)
      # Detectable interaction
      assert h_stat > 0.2
    end

    test "handles partial interaction" do
      # Model: f(x, y) = x + y + 0.5*x*y (additive + interaction)
      predict_fn = fn [x, y] -> x + y + 0.5 * x * y end

      data = for i <- 1..15, j <- 1..3, do: [i * 1.0, j * 1.0]

      h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 8)

      # H should show some interaction
      assert is_float(h_stat)
      assert h_stat > 0.05
      assert h_stat <= 1.0
    end

    test "handles three-way interaction check" do
      # Model with three features
      predict_fn = fn [x, y, z] -> x + y + z + 0.5 * x * y end

      data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 1.0, 1.0],
        [5.0, 2.0, 2.0],
        [6.0, 3.0, 3.0],
        [7.0, 1.0, 2.0],
        [8.0, 2.0, 1.0]
      ]

      # Check interaction between features 0 and 1
      h_stat_01 = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 5)

      # Check interaction between features 0 and 2 (should be lower)
      h_stat_02 = Interaction.h_statistic(predict_fn, data, {0, 2}, num_grid_points: 5)

      # Features 0 and 1 have interaction, 0 and 2 don't
      assert h_stat_01 > h_stat_02
    end

    test "returns value between 0 and 1" do
      predict_fn = fn [x, y] -> x + y * x end
      data = for i <- 1..10, do: [i * 1.0, i * 0.3]

      h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 5)

      assert h_stat >= 0.0
      assert h_stat <= 1.0
    end

    test "handles single feature pair" do
      predict_fn = fn [x, y] -> x * y end
      data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

      h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 3)

      assert is_float(h_stat)
      assert h_stat >= 0.0
    end
  end

  describe "interaction_strength/4" do
    test "computes interaction strength between feature pair" do
      # Simple wrapper around h_statistic with different return format
      predict_fn = fn [x, y, z] -> x + y + 0.3 * x * z end

      data = for i <- 1..15, j <- 1..3, k <- 1..2, do: [i * 1.0, j * 1.0, k * 1.0]

      strength = Interaction.interaction_strength(predict_fn, data, 0, 2)

      assert is_map(strength)
      assert Map.has_key?(strength, :h_statistic)
      assert Map.has_key?(strength, :feature_pair)

      assert strength.feature_pair == {0, 2}
      assert strength.h_statistic >= 0.0
      assert strength.h_statistic <= 1.0
    end

    test "includes interpretation in result" do
      predict_fn = fn [x, y] -> x + y end
      data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]

      strength = Interaction.interaction_strength(predict_fn, data, 0, 1)

      assert Map.has_key?(strength, :interpretation)
      assert is_binary(strength.interpretation)
    end
  end

  describe "find_all_interactions/3" do
    test "finds all pairwise interactions" do
      predict_fn = fn [x, y, z] -> x + y + z + x * y end

      # Simple clear data with 3 features
      data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 1.0, 2.0],
        [5.0, 2.0, 1.0],
        [6.0, 3.0, 3.0],
        [7.0, 1.0, 2.0],
        [8.0, 2.0, 1.0]
      ]

      interactions = Interaction.find_all_interactions(predict_fn, data, num_grid_points: 5)

      # Should find all pairwise interactions
      assert length(interactions) >= 1
      assert Enum.all?(interactions, &is_map/1)

      # All feature pairs should be valid (i < j)
      assert Enum.all?(interactions, fn int ->
               {i, j} = int.feature_pair
               is_integer(i) and is_integer(j) and i < j
             end)

      # Each interaction should have h_statistic between 0 and 1
      assert Enum.all?(interactions, fn int ->
               int.h_statistic >= 0.0 and int.h_statistic <= 1.0
             end)
    end

    test "sorts interactions by strength" do
      predict_fn = fn [x, y, z] -> x + y + z + 0.8 * x * y + 0.1 * y * z end

      data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 1.0, 2.0],
        [5.0, 2.0, 1.0],
        [6.0, 3.0, 3.0],
        [7.0, 1.0, 2.0],
        [8.0, 2.0, 1.0],
        [9.0, 3.0, 2.0],
        [10.0, 1.0, 3.0]
      ]

      interactions =
        Interaction.find_all_interactions(
          predict_fn,
          data,
          num_grid_points: 5,
          sort: true
        )

      # Should find pairwise interactions
      assert length(interactions) >= 1

      # Should be sorted by h_statistic (descending)
      if length(interactions) > 1 do
        h_values = Enum.map(interactions, fn int -> int.h_statistic end)
        assert h_values == Enum.sort(h_values, :desc)
      end

      # Check that all feature pairs are valid (i < j)
      assert Enum.all?(interactions, fn int ->
               {i, j} = int.feature_pair
               is_integer(i) and is_integer(j) and i < j
             end)
    end

    test "filters weak interactions" do
      predict_fn = fn [x, y, z] -> x + y + z + 0.01 * x * y end

      data = [
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 1.0, 2.0],
        [5.0, 2.0, 1.0],
        [6.0, 3.0, 3.0]
      ]

      interactions =
        Interaction.find_all_interactions(
          predict_fn,
          data,
          num_grid_points: 5,
          threshold: 0.3
        )

      # Only interactions with H >= 0.3 should be included
      assert Enum.all?(interactions, fn int -> int.h_statistic >= 0.3 end)
      # May have 0-3 interactions depending on threshold
      assert length(interactions) <= 3
    end
  end

  describe "property-based tests" do
    property "H-statistic is always between 0 and 1" do
      check all(
              n_instances <- integer(5..12),
              data <- list_of(list_of(float(min: 0.0, max: 10.0), length: 3), length: n_instances)
            ) do
        predict_fn = fn [x, y, _z] -> x + y + 0.3 * x * y end

        h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 5)

        assert h_stat >= 0.0
        assert h_stat <= 1.0
      end
    end

    property "purely additive models have low H-statistic" do
      check all(
              n_instances <- integer(8..15),
              data <- list_of(list_of(float(min: 0.0, max: 10.0), length: 2), length: n_instances)
            ) do
        # Purely additive model
        predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

        h_stat = Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 6)

        # Should be very low for additive model
        assert h_stat < 0.3
      end
    end
  end
end

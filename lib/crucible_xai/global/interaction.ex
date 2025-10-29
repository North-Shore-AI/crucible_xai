defmodule CrucibleXAI.Global.Interaction do
  @moduledoc """
  Feature interaction detection using Friedman's H-statistic.

  The H-statistic measures the strength of interaction between features
  by comparing the variance in 2D partial dependence to the sum of
  variances in 1D partial dependences.

  ## H-Statistic Interpretation

  - **H = 0**: No interaction (features are additive)
  - **0 < H < 0.3**: Weak interaction
  - **0.3 ≤ H < 0.7**: Moderate interaction
  - **H ≥ 0.7**: Strong interaction
  - **H = 1**: Pure interaction (effect entirely from interaction)

  ## Formula

  H² = Var(PD_ij - PD_i - PD_j) / Var(PD_ij)

  Where:
  - PD_ij is the 2D partial dependence for features i and j
  - PD_i is the 1D partial dependence for feature i
  - PD_j is the 1D partial dependence for feature j

  ## Use Cases

  - Identify which features interact
  - Guide feature engineering
  - Detect non-additive effects
  - Validate model assumptions

  ## References

  - Friedman, J. H., & Popescu, B. E. (2008). "Predictive Learning via Rule Ensembles"
  - Molnar, C. (2022). "Interpretable Machine Learning" (Chapter on Feature Interactions)

  ## Examples

      predict_fn = fn [x, y, z] -> x + y + 0.5 * x * y + z end
      data = [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], ...]

      # Check interaction between x and y
      h = CrucibleXAI.Global.Interaction.h_statistic(predict_fn, data, {0, 1})
      # => 0.45 (moderate interaction)
  """

  alias CrucibleXAI.Global.PDP

  @doc """
  Compute H-statistic for a pair of features.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset (list of instances)
    * `feature_pair` - Tuple {feature_i, feature_j}
    * `opts` - Options:
      * `:num_grid_points` - Grid resolution (default: 10)

  ## Returns
    Float between 0 and 1 representing interaction strength

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]
      iex> h = CrucibleXAI.Global.Interaction.h_statistic(predict_fn, data, {0, 1}, num_grid_points: 3)
      iex> is_float(h)
      true
      iex> h >= 0.0 and h <= 1.0
      true
  """
  @spec h_statistic(function(), list(list(float())), {integer(), integer()}, keyword()) :: float()
  def h_statistic(predict_fn, data, {feature_i, feature_j}, opts \\ []) do
    num_grid_points = Keyword.get(opts, :num_grid_points, 10)

    # Compute 2D PDP
    pdp_2d =
      PDP.partial_dependence_2d(
        predict_fn,
        data,
        {feature_i, feature_j},
        num_grid_points: num_grid_points
      )

    # Compute 1D PDPs
    pdp_i = PDP.partial_dependence(predict_fn, data, feature_i, num_grid_points: num_grid_points)
    pdp_j = PDP.partial_dependence(predict_fn, data, feature_j, num_grid_points: num_grid_points)

    # Flatten 2D predictions
    pd_ij_flat = List.flatten(pdp_2d.predictions)

    # Compute interaction term: PD_ij - PD_i - PD_j
    # Need to broadcast 1D PDPs to 2D grid
    # PD_i varies by row (x), PD_j varies by column (y)
    interaction_term =
      for {row, row_idx} <- Enum.with_index(pdp_2d.predictions) do
        for {cell, col_idx} <- Enum.with_index(row) do
          pd_i_val = Enum.at(pdp_i.predictions, row_idx)
          pd_j_val = Enum.at(pdp_j.predictions, col_idx)

          # Check for nil values
          if is_nil(cell) or is_nil(pd_i_val) or is_nil(pd_j_val) do
            0.0
          else
            # Interaction = 2D PDP - 1D PDP_i - 1D PDP_j
            cell - pd_i_val - pd_j_val
          end
        end
      end
      |> List.flatten()

    # Compute variances
    var_interaction = compute_variance(interaction_term)
    var_pd_ij = compute_variance(pd_ij_flat)

    # H² = Var(interaction) / Var(PD_ij)
    # Avoid division by zero
    if var_pd_ij < 1.0e-10 do
      0.0
    else
      h_squared = var_interaction / var_pd_ij
      # Return H (not H²)
      :math.sqrt(max(0.0, min(1.0, h_squared)))
    end
  end

  @doc """
  Compute interaction strength with metadata.

  Returns a map with the H-statistic and interpretation.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset
    * `feature_i` - First feature index
    * `feature_j` - Second feature index
    * `opts` - Options (same as h_statistic/4)

  ## Returns
    Map with:
    * `:h_statistic` - The H value
    * `:feature_pair` - The analyzed feature pair
    * `:interpretation` - Text interpretation of strength

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [2.0, 3.0]]
      iex> result = CrucibleXAI.Global.Interaction.interaction_strength(predict_fn, data, 0, 1)
      iex> is_map(result)
      true
      iex> Map.has_key?(result, :interpretation)
      true
  """
  @spec interaction_strength(
          (any() -> any()),
          list(list(float())),
          integer(),
          integer(),
          Keyword.t()
        ) :: %{
          feature_pair: {integer(), integer()},
          h_statistic: float(),
          interpretation: String.t()
        }
  def interaction_strength(predict_fn, data, feature_i, feature_j, opts \\ []) do
    h_stat = h_statistic(predict_fn, data, {feature_i, feature_j}, opts)

    interpretation =
      cond do
        h_stat < 0.1 -> "No interaction"
        h_stat < 0.3 -> "Weak interaction"
        h_stat < 0.7 -> "Moderate interaction"
        true -> "Strong interaction"
      end

    %{
      h_statistic: h_stat,
      feature_pair: {feature_i, feature_j},
      interpretation: interpretation
    }
  end

  @doc """
  Find all pairwise feature interactions.

  Computes H-statistic for all feature pairs and optionally sorts/filters.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset
    * `opts` - Options:
      * `:num_grid_points` - Grid resolution (default: 10)
      * `:sort` - Sort by interaction strength (default: false)
      * `:threshold` - Only return interactions above threshold (default: 0.0)

  ## Returns
    List of interaction strength maps

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [2.0, 3.0]]
      iex> ints = CrucibleXAI.Global.Interaction.find_all_interactions(predict_fn, data)
      iex> is_list(ints)
      true
  """
  @spec find_all_interactions(function(), list(list(float())), keyword()) :: list(map())
  def find_all_interactions(predict_fn, data, opts \\ []) do
    sort = Keyword.get(opts, :sort, false)
    threshold = Keyword.get(opts, :threshold, 0.0)

    # Get number of features - ensure we have valid data structure
    first_instance = hd(data)

    n_features =
      cond do
        is_list(first_instance) -> length(first_instance)
        true -> raise "Invalid data structure: expected list of lists"
      end

    # Generate all feature pairs (only valid pairs where i < j)
    pairs =
      for i <- 0..(n_features - 1),
          j <- (i + 1)..(n_features - 1)//1,
          i < j,
          do: {i, j}

    # Compute interaction strength for each pair
    interactions =
      Enum.map(pairs, fn {i, j} ->
        interaction_strength(predict_fn, data, i, j, opts)
      end)

    # Filter by threshold
    interactions = Enum.filter(interactions, fn int -> int.h_statistic >= threshold end)

    # Sort if requested
    if sort do
      Enum.sort_by(interactions, fn int -> int.h_statistic end, :desc)
    else
      interactions
    end
  end

  # Private helpers

  defp compute_variance(values) do
    if length(values) == 0 do
      0.0
    else
      mean = Enum.sum(values) / length(values)

      sum_squared_diff =
        values
        |> Enum.map(fn v -> (v - mean) * (v - mean) end)
        |> Enum.sum()

      sum_squared_diff / length(values)
    end
  end
end

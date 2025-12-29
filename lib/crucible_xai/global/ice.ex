defmodule CrucibleXAI.Global.ICE do
  @moduledoc """
  Individual Conditional Expectation (ICE) plots for global interpretation.

  ICE plots show how predictions change for individual instances as a
  feature varies. Unlike PDP which averages across instances, ICE shows
  one curve per instance, revealing heterogeneity in the model's behavior.

  ## How It Works

  1. Choose a feature to analyze
  2. Create a grid of values for that feature
  3. For EACH instance in the dataset:
     - Replace the feature with each grid value
     - Get predictions for the instance at each grid value
     - Store the curve (grid_values → predictions)
  4. Return all individual curves

  ## ICE vs PDP

  - **PDP**: Average effect (single curve)
  - **ICE**: Individual effects (one curve per instance)
  - **Relationship**: PDP = Average of ICE curves

  ## Centered ICE

  Subtract the first prediction from each curve to show relative changes.
  This makes it easier to see if curves are parallel (additive effect) or
  have different slopes (interactions).

  ## References

  - Goldstein et al. (2015). "Peeking Inside the Black Box: Visualizing
    Statistical Learning with Plots of Individual Conditional Expectation"

  ## Examples

      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

      ice = CrucibleXAI.Global.ICE.ice_curves(predict_fn, data, 0, num_grid_points: 10)
      # => %{grid_values: [...], curves: [[...], [...], [...]]}

      # Center the curves
      centered = CrucibleXAI.Global.ICE.centered_ice(ice)
  """

  alias CrucibleXAI.Global.PDP

  @doc """
  Compute ICE curves for a feature.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset (list of instances)
    * `feature_index` - Index of feature to analyze
    * `opts` - Options:
      * `:num_grid_points` - Number of grid points (default: 20)
      * `:grid_range` - Custom range as {min, max}

  ## Returns
    Map with:
    * `:grid_values` - List of feature values
    * `:curves` - List of curves (one per instance), each curve is a list of predictions
    * `:feature_index` - The feature that was analyzed

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [3.0, 4.0]]
      iex> ice = CrucibleXAI.Global.ICE.ice_curves(predict_fn, data, 0, num_grid_points: 3)
      iex> is_map(ice)
      true
      iex> length(ice.curves)
      2
  """
  @spec ice_curves(function(), list(list(float())), integer(), keyword()) :: map()
  def ice_curves(predict_fn, data, feature_index, opts \\ []) do
    num_grid_points = Keyword.get(opts, :num_grid_points, 20)

    # Create grid for the feature
    grid_values = PDP.create_grid(data, feature_index, num_grid_points, opts)

    # For each instance, create a curve
    curves =
      Enum.map(data, fn instance ->
        # For each grid value, get prediction for this instance
        Enum.map(grid_values, fn grid_value ->
          # Replace feature with grid value
          modified = List.replace_at(instance, feature_index, grid_value)
          # Get prediction
          get_prediction(predict_fn, modified)
        end)
      end)

    %{
      grid_values: grid_values,
      curves: curves,
      feature_index: feature_index
    }
  end

  @doc """
  Center ICE curves by subtracting the first prediction.

  This shows relative changes from the baseline (first grid point),
  making it easier to see if curves are parallel.

  ## Parameters
    * `ice_result` - Result from ice_curves/4

  ## Returns
    ICE result with centered curves

  ## Examples

      iex> ice = %{grid_values: [1.0, 2.0], curves: [[10.0, 12.0], [5.0, 7.0]], feature_index: 0}
      iex> centered = CrucibleXAI.Global.ICE.centered_ice(ice)
      iex> hd(hd(centered.curves))
      0.0
  """
  @spec centered_ice(%{
          curves: list(),
          grid_values: any(),
          feature_index: any()
        }) :: %{
          curves: list(),
          grid_values: any(),
          feature_index: any()
        }
  def centered_ice(ice_result) do
    centered_curves =
      Enum.map(ice_result.curves, &center_curve/1)

    Map.put(ice_result, :curves, centered_curves)
  end

  defp center_curve(curve) do
    if length(curve) > 0 do
      first = hd(curve)
      Enum.map(curve, fn pred -> pred - first end)
    else
      curve
    end
  end

  @doc """
  Average ICE curves to produce PDP.

  The average of all ICE curves equals the PDP.

  ## Parameters
    * `ice_result` - Result from ice_curves/4

  ## Returns
    List of average predictions (same as PDP predictions)

  ## Examples

      iex> ice = %{curves: [[1.0, 2.0], [3.0, 4.0]]}
      iex> avg = CrucibleXAI.Global.ICE.average_ice_curves(ice)
      iex> avg
      [2.0, 3.0]
  """
  @spec average_ice_curves(map()) :: list(float())
  def average_ice_curves(ice_result) do
    curves = ice_result.curves

    if curves == [] do
      []
    else
      # Number of points in each curve
      num_points = length(hd(curves))

      # For each grid point, average across all curves
      for point_idx <- 0..(num_points - 1) do
        point_values = Enum.map(curves, fn curve -> Enum.at(curve, point_idx) end)
        Enum.sum(point_values) / length(point_values)
      end
    end
  end

  # Private helpers

  defp get_prediction(predict_fn, instance) do
    result = predict_fn.(instance)

    case result do
      %Nx.Tensor{} -> Nx.to_number(result)
      num when is_number(num) -> num
    end
  end
end

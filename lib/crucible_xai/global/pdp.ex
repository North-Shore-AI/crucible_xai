defmodule CrucibleXAI.Global.PDP do
  @moduledoc """
  Partial Dependence Plots (PDP) for global model interpretation.

  PDP shows the marginal effect of one or two features on the predicted
  outcome by averaging predictions over the dataset while varying the
  feature(s) of interest.

  ## How It Works (1D PDP)

  1. Choose a feature to analyze
  2. Create a grid of values for that feature
  3. For each grid value:
     - Replace the feature with that value in ALL data instances
     - Get predictions for all modified instances
     - Average the predictions
  4. Plot grid values vs. average predictions

  ## Interpretation

  - Shows the **average** effect of a feature on predictions
  - Reveals functional relationship (linear, quadratic, etc.)
  - Can identify thresholds and non-linearities
  - Assumes features are independent (can be misleading with correlations)

  ## References

  - Friedman, J. H. (2001). "Greedy Function Approximation: A Gradient Boosting Machine"
  - Molnar, C. (2022). "Interpretable Machine Learning" (Chapter on PDP)

  ## Examples

      predict_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      data = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

      # 1D PDP for feature 0
      pdp = CrucibleXAI.Global.PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 10)
      # => %{grid_values: [...], predictions: [...], feature_index: 0}
  """

  @doc """
  Compute 1D partial dependence for a single feature.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset (list of instances)
    * `feature_index` - Index of feature to analyze
    * `opts` - Options:
      * `:num_grid_points` - Number of grid points (default: 20)
      * `:grid_range` - Custom range as {min, max} (default: auto-detect from data)

  ## Returns
    Map with:
    * `:grid_values` - List of feature values
    * `:predictions` - List of average predictions at each grid value
    * `:feature_index` - The feature that was analyzed

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [3.0, 4.0]]
      iex> pdp = CrucibleXAI.Global.PDP.partial_dependence(predict_fn, data, 0, num_grid_points: 3)
      iex> is_map(pdp)
      true
      iex> Map.has_key?(pdp, :grid_values)
      true
  """
  @spec partial_dependence(function(), list(list(float())), integer(), keyword()) :: map()
  def partial_dependence(predict_fn, data, feature_index, opts \\ []) do
    num_grid_points = Keyword.get(opts, :num_grid_points, 20)

    # Create grid for the feature
    grid_values = create_grid(data, feature_index, num_grid_points, opts)

    # For each grid value, compute average prediction
    predictions =
      Enum.map(grid_values, fn grid_value ->
        # Replace feature with grid value in all instances
        modified_instances =
          Enum.map(data, fn instance ->
            List.replace_at(instance, feature_index, grid_value)
          end)

        # Get predictions for all modified instances
        preds = Enum.map(modified_instances, fn inst -> get_prediction(predict_fn, inst) end)

        # Return average prediction
        Enum.sum(preds) / length(preds)
      end)

    %{
      grid_values: grid_values,
      predictions: predictions,
      feature_index: feature_index
    }
  end

  @doc """
  Compute 2D partial dependence for a pair of features.

  Shows interaction effects between two features.

  ## Parameters
    * `predict_fn` - Prediction function
    * `data` - Dataset (list of instances)
    * `feature_pair` - Tuple of {feature_index_1, feature_index_2}
    * `opts` - Options:
      * `:num_grid_points` - Grid points per feature (default: 10)
      * `:grid_range_x` - Range for first feature
      * `:grid_range_y` - Range for second feature

  ## Returns
    Map with:
    * `:grid_values_x` - Grid values for first feature
    * `:grid_values_y` - Grid values for second feature
    * `:predictions` - 2D matrix of average predictions
    * `:feature_indices` - The feature pair that was analyzed

  ## Examples

      iex> predict_fn = fn [x, y] -> x + y end
      iex> data = [[1.0, 2.0], [3.0, 4.0]]
      iex> pdp = CrucibleXAI.Global.PDP.partial_dependence_2d(predict_fn, data, {0, 1}, num_grid_points: 3)
      iex> is_map(pdp)
      true
  """
  @spec partial_dependence_2d(function(), list(list(float())), {integer(), integer()}, keyword()) ::
          map()
  def partial_dependence_2d(predict_fn, data, {feature_x, feature_y}, opts \\ []) do
    num_grid_points = Keyword.get(opts, :num_grid_points, 10)

    # Create grids for both features
    grid_x_opts =
      if Keyword.has_key?(opts, :grid_range_x) do
        [grid_range: Keyword.get(opts, :grid_range_x)]
      else
        []
      end

    grid_y_opts =
      if Keyword.has_key?(opts, :grid_range_y) do
        [grid_range: Keyword.get(opts, :grid_range_y)]
      else
        []
      end

    grid_values_x = create_grid(data, feature_x, num_grid_points, grid_x_opts)
    grid_values_y = create_grid(data, feature_y, num_grid_points, grid_y_opts)

    # For each combination of grid values, compute average prediction
    predictions =
      for grid_x <- grid_values_x do
        for grid_y <- grid_values_y do
          # Replace both features in all instances
          modified_instances =
            Enum.map(data, fn instance ->
              instance
              |> List.replace_at(feature_x, grid_x)
              |> List.replace_at(feature_y, grid_y)
            end)

          # Get predictions and average
          preds = Enum.map(modified_instances, fn inst -> get_prediction(predict_fn, inst) end)
          Enum.sum(preds) / length(preds)
        end
      end

    %{
      grid_values_x: grid_values_x,
      grid_values_y: grid_values_y,
      predictions: predictions,
      feature_indices: {feature_x, feature_y}
    }
  end

  @doc """
  Create evenly spaced grid for a feature.

  ## Parameters
    * `data` - Dataset
    * `feature_index` - Feature to create grid for
    * `num_points` - Number of grid points
    * `opts` - Options:
      * `:grid_range` - Custom range as {min, max}

  ## Returns
    List of evenly spaced values

  ## Examples

      iex> data = [[1.0, 5.0], [3.0, 6.0], [5.0, 7.0]]
      iex> grid = CrucibleXAI.Global.PDP.create_grid(data, 0, 5)
      iex> length(grid)
      5
  """
  @spec create_grid(list(list(float())), integer(), integer(), keyword()) :: list(float())
  def create_grid(data, feature_index, num_points, opts \\ []) do
    {min_val, max_val} =
      if Keyword.has_key?(opts, :grid_range) do
        Keyword.get(opts, :grid_range)
      else
        # Auto-detect range from data
        feature_values =
          Enum.map(data, fn instance -> Enum.at(instance, feature_index) end)
          |> Enum.reject(&is_nil/1)

        if feature_values == [] do
          # Default range if no data
          {0.0, 1.0}
        else
          {Enum.min(feature_values), Enum.max(feature_values)}
        end
      end

    # Handle case where min == max
    {min_val, max_val} =
      if abs(max_val - min_val) < 1.0e-10 do
        {min_val - 0.5, max_val + 0.5}
      else
        {min_val, max_val}
      end

    # Create evenly spaced grid
    if num_points == 1 do
      [(min_val + max_val) / 2]
    else
      step = (max_val - min_val) / (num_points - 1)

      for i <- 0..(num_points - 1) do
        min_val + i * step
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

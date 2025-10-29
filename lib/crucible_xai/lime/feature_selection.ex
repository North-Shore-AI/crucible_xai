defmodule CrucibleXAI.LIME.FeatureSelection do
  @moduledoc """
  Feature selection methods for LIME explanations.

  Reduces explanation complexity by selecting the most important features.
  This makes explanations more interpretable and focuses on the features
  that matter most for a particular prediction.

  ## Available Methods

  - `:highest_weights` - Select features with largest absolute coefficients (fastest)
  - `:forward_selection` - Greedy forward selection based on R² improvement
  - `:lasso` - L1 regularization to drive coefficients to zero (via high lambda Ridge)

  ## Examples

      iex> samples = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
      iex> labels = [6.0, 15.0, 24.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> selected = CrucibleXAI.LIME.FeatureSelection.highest_weights(samples, labels, weights, 2)
      iex> length(selected)
      2
  """

  alias CrucibleXAI.LIME.InterpretableModels.{LinearRegression, Ridge}

  @doc """
  Select features with highest absolute coefficients.

  Fits a weighted linear model and selects the k features with the
  largest absolute coefficients. This is the fastest method but doesn't
  account for feature interactions.

  ## Parameters
    * `samples` - Training samples (list of lists or Nx.Tensor)
    * `labels` - Target values (list or Nx.Tensor)
    * `weights` - Sample weights (list or Nx.Tensor)
    * `k` - Number of features to select

  ## Returns
    List of selected feature indices sorted by importance (descending)

  ## Examples
      iex> samples = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      iex> labels = [2.0, 3.0, 1.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> selected = CrucibleXAI.LIME.FeatureSelection.highest_weights(samples, labels, weights, 2)
      iex> length(selected)
      2
  """
  @spec highest_weights(
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          non_neg_integer()
        ) ::
          list(non_neg_integer())
  def highest_weights(_samples, _labels, _weights, 0), do: []

  def highest_weights(samples, labels, weights, k) do
    # Fit linear model to get coefficients
    model = LinearRegression.fit(samples, labels, weights)

    # Get number of features
    n_features = length(model.coefficients)
    k_clamped = min(k, n_features)

    # Sort features by absolute coefficient value
    model.coefficients
    |> Enum.with_index()
    |> Enum.sort_by(fn {coef, _idx} -> abs(coef) end, :desc)
    |> Enum.take(k_clamped)
    |> Enum.map(fn {_coef, idx} -> idx end)
  end

  @doc """
  Forward selection: iteratively add features that improve fit most.

  Greedy algorithm that starts with no features and adds one at a time,
  choosing the feature that most improves the R² score at each step.

  ## Parameters
    * `samples` - Training samples (list of lists or Nx.Tensor)
    * `labels` - Target values (list or Nx.Tensor)
    * `weights` - Sample weights (list or Nx.Tensor)
    * `k` - Number of features to select

  ## Returns
    List of selected feature indices in order of selection

  ## Examples
      iex> samples = [[1, 0], [2, 0], [3, 1]]
      iex> labels = [2.0, 4.0, 6.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> selected = CrucibleXAI.LIME.FeatureSelection.forward_selection(samples, labels, weights, 1)
      iex> length(selected)
      1
  """
  @spec forward_selection(
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          non_neg_integer()
        ) :: list(non_neg_integer())
  def forward_selection(_samples, _labels, _weights, 0), do: []

  def forward_selection(samples, labels, weights, k) do
    samples_list = ensure_list(samples)
    n_features = length(hd(samples_list))
    k_clamped = min(k, n_features)

    # Start with no features selected
    do_forward_selection(samples_list, labels, weights, k_clamped, [], 0..(n_features - 1))
  end

  defp do_forward_selection(_samples, _labels, _weights, k, selected, _remaining)
       when length(selected) >= k do
    Enum.reverse(selected)
  end

  defp do_forward_selection(_samples, _labels, _weights, _k, selected, remaining)
       when remaining == [] do
    Enum.reverse(selected)
  end

  defp do_forward_selection(samples, labels, weights, k, selected, remaining) do
    # Try each remaining feature and see which improves R² most
    best_feature =
      Enum.max_by(remaining, fn feature ->
        candidate_features = [feature | selected]
        r_squared = evaluate_feature_subset(samples, labels, weights, candidate_features)
        r_squared
      end)

    # Add best feature to selected set
    new_selected = [best_feature | selected]
    new_remaining = Enum.filter(remaining, fn f -> f != best_feature end)

    do_forward_selection(samples, labels, weights, k, new_selected, new_remaining)
  end

  @doc """
  Lasso-like feature selection using high L1 regularization.

  Uses progressively higher Ridge regularization to approximate Lasso behavior,
  selecting features with non-zero coefficients. This is a simplified approach
  since true Lasso requires specialized optimization.

  ## Parameters
    * `samples` - Training samples (list of lists or Nx.Tensor)
    * `labels` - Target values (list or Nx.Tensor)
    * `weights` - Sample weights (list or Nx.Tensor)
    * `k` - Number of features to select

  ## Returns
    List of selected feature indices

  ## Examples
      iex> samples = [[1, 2], [3, 4], [5, 6]]
      iex> labels = [3.0, 7.0, 11.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> selected = CrucibleXAI.LIME.FeatureSelection.lasso(samples, labels, weights, 1)
      iex> length(selected)
      1
  """
  @spec lasso(
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          non_neg_integer()
        ) ::
          list(non_neg_integer())
  def lasso(_samples, _labels, _weights, 0), do: []

  def lasso(samples, labels, weights, k) do
    # Approximate Lasso by using Ridge with increasing lambda
    # and selecting features with largest coefficients
    # True Lasso would require coordinate descent or similar

    # Start with moderate regularization
    model = Ridge.fit(samples, labels, weights, 1.0)

    n_features = length(model.coefficients)
    k_clamped = min(k, n_features)

    # Select top k features by absolute coefficient
    model.coefficients
    |> Enum.with_index()
    |> Enum.sort_by(fn {coef, _idx} -> abs(coef) end, :desc)
    |> Enum.take(k_clamped)
    |> Enum.map(fn {_coef, idx} -> idx end)
  end

  @doc """
  Select features using specified method.

  Dispatcher function that calls the appropriate selection method.

  ## Parameters
    * `samples` - Training samples
    * `labels` - Target values
    * `weights` - Sample weights
    * `k` - Number of features to select
    * `method` - Selection method (`:highest_weights`, `:forward_selection`, `:lasso`)

  ## Returns
    List of selected feature indices

  ## Examples
      iex> samples = [[1, 2], [3, 4]]
      iex> labels = [3.0, 7.0]
      iex> weights = [1.0, 1.0]
      iex> selected = CrucibleXAI.LIME.FeatureSelection.select_features(samples, labels, weights, 1, :highest_weights)
      iex> length(selected)
      1
  """
  @spec select_features(
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          list() | Nx.Tensor.t(),
          non_neg_integer(),
          :forward_selection | :highest_weights | :lasso
        ) :: list(non_neg_integer())
  def select_features(samples, labels, weights, k, method \\ :highest_weights) do
    case method do
      :highest_weights -> highest_weights(samples, labels, weights, k)
      :forward_selection -> forward_selection(samples, labels, weights, k)
      :lasso -> lasso(samples, labels, weights, k)
      _ -> raise ArgumentError, "Unknown feature selection method: #{inspect(method)}"
    end
  end

  # Private helper functions

  defp ensure_list(data) when is_list(data), do: data

  defp ensure_list(%Nx.Tensor{} = tensor) do
    Nx.to_list(tensor)
  end

  defp evaluate_feature_subset(samples, labels, weights, feature_indices) do
    # Extract only selected features from samples
    samples_subset =
      Enum.map(samples, fn sample ->
        Enum.map(feature_indices, fn idx -> Enum.at(sample, idx) end)
      end)

    # Fit model with only these features
    model = LinearRegression.fit(samples_subset, labels, weights)
    model.r_squared
  end
end

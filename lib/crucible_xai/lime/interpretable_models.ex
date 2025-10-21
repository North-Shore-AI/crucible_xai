defmodule CrucibleXAI.LIME.InterpretableModels do
  @moduledoc """
  Interpretable models for LIME explanations.

  This module provides simple, interpretable models that can be used to
  approximate complex black-box models locally. These models are designed
  to be easily understood by humans while faithfully representing the
  local behavior of the complex model.

  ## Available Models

  - `LinearRegression` - Weighted ordinary least squares
  - `Ridge` - Linear regression with L2 regularization

  ## Model Structure

  All models return a map with:
  - `:intercept` - The intercept/bias term
  - `:coefficients` - List of feature coefficients
  - `:r_squared` - R² score measuring goodness of fit
  """
end

defmodule CrucibleXAI.LIME.InterpretableModels.LinearRegression do
  @moduledoc """
  Weighted linear regression for LIME.

  Solves the weighted least squares problem:

      β = (X'WX)^(-1) X'Wy

  where:
  - X is the design matrix (samples)
  - W is a diagonal matrix of sample weights
  - y is the target vector (labels)
  - β contains [coefficients; intercept]

  ## Examples

      iex> samples = [[1, 2], [3, 4], [5, 6]]
      iex> labels = [5.0, 11.0, 17.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> model = CrucibleXAI.LIME.InterpretableModels.LinearRegression.fit(samples, labels, weights)
      iex> is_float(model.intercept)
      true
      iex> length(model.coefficients)
      2
      iex> model.r_squared > 0.9
      true
  """

  @type model :: %{
          intercept: float(),
          coefficients: list(float()),
          r_squared: float()
        }

  @doc """
  Fit weighted linear regression model.

  ## Parameters
    * `samples` - Training samples (list of lists or Nx.Tensor) of shape {n_samples, n_features}
    * `labels` - Target values (list or Nx.Tensor) of shape {n_samples}
    * `weights` - Sample weights (list or Nx.Tensor) of shape {n_samples}

  ## Returns
    Model map with `:intercept`, `:coefficients`, and `:r_squared`

  ## Examples
      iex> samples = [[1, 2], [3, 4], [5, 6]]
      iex> labels = [2.0, 3.0, 4.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> model = CrucibleXAI.LIME.InterpretableModels.LinearRegression.fit(samples, labels, weights)
      iex> is_map(model)
      true
  """
  @spec fit(list() | Nx.Tensor.t(), list() | Nx.Tensor.t(), list() | Nx.Tensor.t()) :: model()
  def fit(samples, labels, weights) do
    # Convert inputs to tensors
    x = prepare_samples(samples)
    y = prepare_vector(labels)
    w = prepare_vector(weights)

    # Add intercept column (column of ones)
    {n_samples, _n_features} = Nx.shape(x)
    ones = Nx.broadcast(1.0, {n_samples, 1})
    x_with_intercept = Nx.concatenate([ones, x], axis: 1)

    # Compute weighted least squares: β = (X'WX)^(-1) X'Wy
    # Create diagonal weight matrix
    w_diag = Nx.make_diagonal(w)

    # X'W
    xt_w = Nx.dot(Nx.transpose(x_with_intercept), w_diag)

    # X'WX
    xt_w_x = Nx.dot(xt_w, x_with_intercept)

    # X'Wy
    xt_w_y = Nx.dot(xt_w, y)

    # Solve (X'WX)β = X'Wy
    # Beta will be [intercept, coef1, coef2, ...]
    beta = solve_linear_system(xt_w_x, xt_w_y)

    # Extract intercept and coefficients
    beta_list = Nx.to_flat_list(beta)
    intercept = hd(beta_list)
    coefficients = tl(beta_list)

    # Calculate R²
    predictions = predict_internal(x_with_intercept, beta)
    r_squared = calculate_r_squared(y, predictions, w)

    %{
      intercept: intercept,
      coefficients: coefficients,
      r_squared: r_squared
    }
  end

  @doc """
  Predict using fitted linear regression model.

  ## Parameters
    * `model` - Fitted model from `fit/3`
    * `samples` - Samples to predict (list of lists or Nx.Tensor)

  ## Returns
    Nx.Tensor of predictions

  ## Examples
      iex> model = %{intercept: 1.0, coefficients: [2.0, 3.0], r_squared: 0.95}
      iex> predictions = CrucibleXAI.LIME.InterpretableModels.LinearRegression.predict(model, [[1, 1], [2, 2]])
      iex> Nx.shape(predictions)
      {2}
  """
  @spec predict(model(), list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict(model, samples) do
    x = prepare_samples(samples)
    coefficients_tensor = Nx.tensor(model.coefficients)

    # y = X @ coef + intercept
    x
    |> Nx.dot(coefficients_tensor)
    |> Nx.add(model.intercept)
  end

  # Private helper functions

  defp prepare_samples(samples) when is_list(samples) do
    Nx.tensor(samples)
  end

  defp prepare_samples(%Nx.Tensor{} = samples), do: samples

  defp prepare_vector(vector) when is_list(vector) do
    Nx.tensor(vector)
  end

  defp prepare_vector(%Nx.Tensor{} = vector), do: vector

  defp predict_internal(x_with_intercept, beta) do
    Nx.dot(x_with_intercept, beta)
  end

  defp solve_linear_system(a, b) do
    # Use Cholesky decomposition for positive definite matrices
    # Add small regularization for numerical stability
    eps = 1.0e-8
    n = Nx.axis_size(a, 0)
    a_reg = Nx.add(a, Nx.multiply(Nx.eye(n), eps))

    # Solve using Nx linear solve
    try do
      Nx.LinAlg.solve(a_reg, b)
    rescue
      _ ->
        # Fallback: use pseudoinverse
        Nx.dot(Nx.LinAlg.pinv(a), b)
    end
  end

  defp calculate_r_squared(y_true, y_pred, weights) do
    # Weighted R² = 1 - (SS_res / SS_tot)
    # SS_res = Σw_i(y_i - ŷ_i)²
    # SS_tot = Σw_i(y_i - ȳ)²

    # Weighted mean
    weighted_sum = Nx.dot(weights, y_true)
    weight_sum = Nx.sum(weights)
    y_mean = Nx.divide(weighted_sum, weight_sum)

    # Residual sum of squares
    residuals = Nx.subtract(y_true, y_pred)
    ss_res = Nx.dot(weights, Nx.pow(residuals, 2))

    # Total sum of squares
    deviations = Nx.subtract(y_true, y_mean)
    ss_tot = Nx.dot(weights, Nx.pow(deviations, 2))

    # R² = 1 - SS_res/SS_tot
    r_squared =
      Nx.subtract(1.0, Nx.divide(ss_res, Nx.add(ss_tot, 1.0e-10)))
      |> Nx.to_number()

    # Clamp to reasonable range
    max(min(r_squared, 1.0), -10.0)
  end
end

defmodule CrucibleXAI.LIME.InterpretableModels.Ridge do
  @moduledoc """
  Ridge regression with L2 regularization.

  Solves the regularized weighted least squares problem:

      β = (X'WX + λI)^(-1) X'Wy

  where λ is the regularization strength. L2 regularization helps prevent
  overfitting and improves numerical stability.

  ## Examples

      iex> samples = [[1, 2], [3, 4], [5, 6]]
      iex> labels = [5.0, 11.0, 17.0]
      iex> weights = [1.0, 1.0, 1.0]
      iex> model = CrucibleXAI.LIME.InterpretableModels.Ridge.fit(samples, labels, weights, 0.1)
      iex> is_float(model.intercept)
      true
  """

  alias CrucibleXAI.LIME.InterpretableModels.LinearRegression

  @type model :: LinearRegression.model()

  @default_lambda 1.0

  @doc """
  Fit ridge regression model with L2 regularization.

  ## Parameters
    * `samples` - Training samples (list of lists or Nx.Tensor)
    * `labels` - Target values (list or Nx.Tensor)
    * `weights` - Sample weights (list or Nx.Tensor)
    * `lambda` - Regularization strength (default: 1.0). Higher values = more regularization

  ## Returns
    Model map with `:intercept`, `:coefficients`, and `:r_squared`

  ## Examples
      iex> samples = [[1, 2], [3, 4]]
      iex> labels = [5.0, 11.0]
      iex> weights = [1.0, 1.0]
      iex> model = CrucibleXAI.LIME.InterpretableModels.Ridge.fit(samples, labels, weights, 0.5)
      iex> length(model.coefficients)
      2
  """
  @spec fit(list() | Nx.Tensor.t(), list() | Nx.Tensor.t(), list() | Nx.Tensor.t(), float()) ::
          model()
  def fit(samples, labels, weights, lambda \\ @default_lambda) do
    # Convert inputs to tensors
    x = prepare_samples(samples)
    y = prepare_vector(labels)
    w = prepare_vector(weights)

    # Add intercept column
    {n_samples, n_features} = Nx.shape(x)
    ones = Nx.broadcast(1.0, {n_samples, 1})
    x_with_intercept = Nx.concatenate([ones, x], axis: 1)

    # Compute ridge regression: β = (X'WX + λI)^(-1) X'Wy
    w_diag = Nx.make_diagonal(w)

    # X'W
    xt_w = Nx.dot(Nx.transpose(x_with_intercept), w_diag)

    # X'WX
    xt_w_x = Nx.dot(xt_w, x_with_intercept)

    # Add L2 regularization: X'WX + λI
    # Don't regularize intercept (first element)
    n_params = n_features + 1
    regularization = Nx.eye(n_params)
    # Set first diagonal element to 0 (don't regularize intercept)
    regularization = Nx.indexed_put(regularization, Nx.tensor([[0, 0]]), Nx.tensor([0.0]))
    regularization = Nx.multiply(regularization, lambda)

    xt_w_x_reg = Nx.add(xt_w_x, regularization)

    # X'Wy
    xt_w_y = Nx.dot(xt_w, y)

    # Solve (X'WX + λI)β = X'Wy
    beta = solve_linear_system(xt_w_x_reg, xt_w_y)

    # Extract intercept and coefficients
    beta_list = Nx.to_flat_list(beta)
    intercept = hd(beta_list)
    coefficients = tl(beta_list)

    # Calculate R²
    predictions = predict_internal(x_with_intercept, beta)
    r_squared = calculate_r_squared(y, predictions, w)

    %{
      intercept: intercept,
      coefficients: coefficients,
      r_squared: r_squared
    }
  end

  @doc """
  Predict using fitted ridge regression model.

  ## Parameters
    * `model` - Fitted model from `fit/4`
    * `samples` - Samples to predict (list of lists or Nx.Tensor)

  ## Returns
    Nx.Tensor of predictions
  """
  @spec predict(model(), list() | Nx.Tensor.t()) :: Nx.Tensor.t()
  def predict(model, samples) do
    LinearRegression.predict(model, samples)
  end

  # Private helper functions (same as LinearRegression)

  defp prepare_samples(samples) when is_list(samples), do: Nx.tensor(samples)
  defp prepare_samples(%Nx.Tensor{} = samples), do: samples

  defp prepare_vector(vector) when is_list(vector), do: Nx.tensor(vector)
  defp prepare_vector(%Nx.Tensor{} = vector), do: vector

  defp predict_internal(x_with_intercept, beta) do
    Nx.dot(x_with_intercept, beta)
  end

  defp solve_linear_system(a, b) do
    try do
      Nx.LinAlg.solve(a, b)
    rescue
      _ ->
        Nx.dot(Nx.LinAlg.pinv(a), b)
    end
  end

  defp calculate_r_squared(y_true, y_pred, weights) do
    weighted_sum = Nx.dot(weights, y_true)
    weight_sum = Nx.sum(weights)
    y_mean = Nx.divide(weighted_sum, weight_sum)

    residuals = Nx.subtract(y_true, y_pred)
    ss_res = Nx.dot(weights, Nx.pow(residuals, 2))

    deviations = Nx.subtract(y_true, y_mean)
    ss_tot = Nx.dot(weights, Nx.pow(deviations, 2))

    r_squared =
      Nx.subtract(1.0, Nx.divide(ss_res, Nx.add(ss_tot, 1.0e-10)))
      |> Nx.to_number()

    max(min(r_squared, 1.0), -10.0)
  end
end

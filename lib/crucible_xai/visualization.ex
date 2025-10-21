defmodule CrucibleXAI.Visualization do
  @moduledoc """
  Visualization utilities for XAI explanations.

  Provides HTML generation for interactive visualizations of LIME, SHAP,
  and feature attribution results.

  ## Examples

      explanation = CrucibleXai.explain(instance, predict_fn)
      html = CrucibleXAI.Visualization.to_html(explanation)
      CrucibleXAI.Visualization.save_html(explanation, "explanation.html")
  """

  alias CrucibleXAI.Explanation

  @doc """
  Generate HTML visualization for an explanation.

  ## Parameters
    * `explanation` - Explanation struct
    * `opts` - Options:
      * `:feature_names` - Map of feature_index => name
      * `:style` - Style theme (`:light` or `:dark`)
      * `:num_features` - Number of features to display

  ## Returns
    HTML string

  ## Examples
      iex> exp = %CrucibleXAI.Explanation{
      ...>   instance: [1.0, 2.0],
      ...>   feature_weights: %{0 => 0.5, 1 => 0.3},
      ...>   method: :lime
      ...> }
      iex> html = CrucibleXAI.Visualization.to_html(exp)
      iex> String.contains?(html, "<!DOCTYPE html>")
      true
  """
  @spec to_html(Explanation.t(), keyword()) :: String.t()
  def to_html(%Explanation{} = explanation, opts \\ []) do
    feature_names = Keyword.get(opts, :feature_names, %{})
    style = Keyword.get(opts, :style, :light)
    num_features = Keyword.get(opts, :num_features, 10)

    method_name = explanation.method |> to_string() |> String.upcase()

    """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>#{method_name} Explanation</title>
      <style>#{css_styles(style)}</style>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    </head>
    <body>
      <div class="container">
        <h1>#{method_name} Explanation</h1>
        #{summary_section(explanation)}
        <div class="chart-container">
          <canvas id="featureChart"></canvas>
        </div>
        #{feature_table(explanation, num_features, feature_names)}
      </div>
      <script>#{javascript(explanation, num_features, feature_names)}</script>
    </body>
    </html>
    """
  end

  @doc """
  Generate HTML for SHAP values.

  ## Parameters
    * `shap_values` - Map of feature_index => shapley_value
    * `instance` - The instance that was explained
    * `opts` - Options (same as `to_html/2`)

  ## Returns
    HTML string
  """
  @spec shap_to_html(map(), list(), keyword()) :: String.t()
  def shap_to_html(shap_values, instance, opts \\ []) do
    feature_names = Keyword.get(opts, :feature_names, %{})
    style = Keyword.get(opts, :style, :light)

    """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>SHAP Explanation</title>
      <style>#{css_styles(style)}</style>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    </head>
    <body>
      <div class="container">
        <h1>SHAP Values</h1>
        #{shap_summary_section(shap_values, instance)}
        <div class="chart-container">
          <canvas id="shapChart"></canvas>
        </div>
        #{shap_feature_table(shap_values, feature_names)}
      </div>
      <script>#{shap_javascript(shap_values, feature_names)}</script>
    </body>
    </html>
    """
  end

  @doc """
  Save visualization to HTML file.

  ## Parameters
    * `explanation` - Explanation struct
    * `path` - File path to save to
    * `opts` - Visualization options

  ## Returns
    `{:ok, path}` on success
  """
  @spec save_html(Explanation.t(), String.t(), keyword()) :: {:ok, String.t()}
  def save_html(explanation, path, opts \\ []) do
    html = to_html(explanation, opts)
    File.write!(path, html)
    {:ok, path}
  end

  @doc """
  Generate comparison HTML for LIME vs SHAP.

  ## Parameters
    * `lime_explanation` - LIME explanation
    * `shap_values` - SHAP values map
    * `instance` - Instance that was explained
    * `opts` - Options

  ## Returns
    HTML string comparing both methods
  """
  @spec comparison_html(Explanation.t(), map(), list(), keyword()) :: String.t()
  def comparison_html(lime_explanation, shap_values, instance, opts \\ []) do
    feature_names = Keyword.get(opts, :feature_names, %{})
    style = Keyword.get(opts, :style, :light)

    """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>LIME vs SHAP Comparison</title>
      <style>#{css_styles(style)}</style>
      <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    </head>
    <body>
      <div class="container">
        <h1>Explanation Comparison: LIME vs SHAP</h1>
        <div class="comparison-grid">
          <div class="method-section">
            <h2>LIME</h2>
            #{summary_section(lime_explanation)}
          </div>
          <div class="method-section">
            <h2>SHAP</h2>
            #{shap_summary_section(shap_values, instance)}
          </div>
        </div>
        <div class="chart-container">
          <canvas id="comparisonChart"></canvas>
        </div>
      </div>
      <script>#{comparison_javascript(lime_explanation, shap_values, feature_names)}</script>
    </body>
    </html>
    """
  end

  # Private helper functions

  defp css_styles(:light) do
    """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
    .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
    h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
    h2 { color: #555; }
    .summary { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0; }
    .summary-item { margin: 8px 0; }
    .chart-container { margin: 30px 0; position: relative; height: 400px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
    th { background: #4CAF50; color: white; }
    tr:hover { background: #f5f5f5; }
    .positive { color: #4CAF50; font-weight: bold; }
    .negative { color: #f44336; font-weight: bold; }
    .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .method-section { padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
    """
  end

  defp css_styles(:dark) do
    """
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #1a1a1a; color: #e0e0e0; }
    .container { max-width: 1200px; margin: 0 auto; background: #2d2d2d; padding: 30px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }
    h1 { color: #e0e0e0; border-bottom: 3px solid #66BB6A; padding-bottom: 10px; }
    h2 { color: #b0b0b0; }
    .summary { background: #3a3a3a; padding: 15px; border-radius: 5px; margin: 20px 0; }
    .summary-item { margin: 8px 0; }
    .chart-container { margin: 30px 0; position: relative; height: 400px; }
    table { width: 100%; border-collapse: collapse; margin: 20px 0; }
    th, td { padding: 12px; text-align: left; border-bottom: 1px solid #444; }
    th { background: #66BB6A; color: #1a1a1a; }
    tr:hover { background: #3a3a3a; }
    .positive { color: #66BB6A; font-weight: bold; }
    .negative { color: #EF5350; font-weight: bold; }
    .comparison-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    .method-section { padding: 15px; border: 1px solid #444; border-radius: 5px; }
    """
  end

  defp summary_section(explanation) do
    """
    <div class="summary">
      <div class="summary-item"><strong>Method:</strong> #{explanation.method}</div>
      #{if explanation.score, do: "<div class=\"summary-item\"><strong>R² Score:</strong> #{Float.round(explanation.score, 4)}</div>", else: ""}
      #{if explanation.intercept, do: "<div class=\"summary-item\"><strong>Intercept:</strong> #{Float.round(explanation.intercept, 4)}</div>", else: ""}
      <div class="summary-item"><strong>Number of Features:</strong> #{map_size(explanation.feature_weights)}</div>
    </div>
    """
  end

  defp shap_summary_section(shap_values, instance) do
    shap_sum = Enum.sum(Map.values(shap_values))

    """
    <div class="summary">
      <div class="summary-item"><strong>Method:</strong> SHAP (Shapley Values)</div>
      <div class="summary-item"><strong>Sum of SHAP values:</strong> #{Float.round(shap_sum, 4)}</div>
      <div class="summary-item"><strong>Instance:</strong> #{inspect(instance)}</div>
      <div class="summary-item"><strong>Number of Features:</strong> #{map_size(shap_values)}</div>
    </div>
    """
  end

  defp feature_table(explanation, num_features, feature_names) do
    top_features =
      Explanation.top_features(explanation, num_features)

    rows =
      Enum.map(top_features, fn {idx, weight} ->
        name = Map.get(feature_names, idx, "Feature #{idx}")
        class = if weight > 0, do: "positive", else: "negative"
        sign = if weight > 0, do: "+", else: ""

        """
        <tr>
          <td>#{name}</td>
          <td class="#{class}">#{sign}#{Float.round(weight, 4)}</td>
        </tr>
        """
      end)
      |> Enum.join()

    """
    <h2>Top Features</h2>
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Weight</th>
        </tr>
      </thead>
      <tbody>
        #{rows}
      </tbody>
    </table>
    """
  end

  defp shap_feature_table(shap_values, feature_names) do
    sorted_features =
      shap_values
      |> Enum.sort_by(fn {_idx, val} -> abs(val) end, :desc)

    rows =
      Enum.map(sorted_features, fn {idx, value} ->
        name = Map.get(feature_names, idx, "Feature #{idx}")
        class = if value > 0, do: "positive", else: "negative"
        sign = if value > 0, do: "+", else: ""

        """
        <tr>
          <td>#{name}</td>
          <td class="#{class}">#{sign}#{Float.round(value, 4)}</td>
        </tr>
        """
      end)
      |> Enum.join()

    """
    <h2>SHAP Values</h2>
    <table>
      <thead>
        <tr>
          <th>Feature</th>
          <th>Shapley Value</th>
        </tr>
      </thead>
      <tbody>
        #{rows}
      </tbody>
    </table>
    """
  end

  defp javascript(explanation, num_features, feature_names) do
    top_features = Explanation.top_features(explanation, num_features)

    labels =
      Enum.map(top_features, fn {idx, _} ->
        Map.get(feature_names, idx, "Feature #{idx}")
      end)

    data = Enum.map(top_features, fn {_, weight} -> weight end)
    colors = Enum.map(data, fn val -> if val > 0, do: "#4CAF50", else: "#f44336" end)

    """
    const ctx = document.getElementById('featureChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: #{Jason.encode!(labels)},
        datasets: [{
          label: 'Feature Weight',
          data: #{Jason.encode!(data)},
          backgroundColor: #{Jason.encode!(colors)}
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'Feature Importance'
          },
          legend: {
            display: false
          }
        }
      }
    });
    """
  end

  defp shap_javascript(shap_values, feature_names) do
    sorted_features =
      shap_values
      |> Enum.sort_by(fn {_idx, val} -> abs(val) end, :desc)

    labels =
      Enum.map(sorted_features, fn {idx, _} ->
        Map.get(feature_names, idx, "Feature #{idx}")
      end)

    data = Enum.map(sorted_features, fn {_, value} -> value end)
    colors = Enum.map(data, fn val -> if val > 0, do: "#4CAF50", else: "#f44336" end)

    """
    const ctx = document.getElementById('shapChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: #{Jason.encode!(labels)},
        datasets: [{
          label: 'Shapley Value',
          data: #{Jason.encode!(data)},
          backgroundColor: #{Jason.encode!(colors)}
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'SHAP Feature Attribution'
          }
        }
      }
    });
    """
  end

  defp comparison_javascript(lime_explanation, shap_values, feature_names) do
    # Get all feature indices from both
    lime_features = Map.keys(lime_explanation.feature_weights)
    shap_features = Map.keys(shap_values)
    all_features = Enum.uniq(lime_features ++ shap_features) |> Enum.sort()

    labels =
      Enum.map(all_features, fn idx ->
        Map.get(feature_names, idx, "Feature #{idx}")
      end)

    lime_data =
      Enum.map(all_features, fn idx ->
        Map.get(lime_explanation.feature_weights, idx, 0.0)
      end)

    shap_data =
      Enum.map(all_features, fn idx ->
        Map.get(shap_values, idx, 0.0)
      end)

    """
    const ctx = document.getElementById('comparisonChart').getContext('2d');
    new Chart(ctx, {
      type: 'bar',
      data: {
        labels: #{Jason.encode!(labels)},
        datasets: [
          {
            label: 'LIME',
            data: #{Jason.encode!(lime_data)},
            backgroundColor: 'rgba(76, 175, 80, 0.7)'
          },
          {
            label: 'SHAP',
            data: #{Jason.encode!(shap_data)},
            backgroundColor: 'rgba(33, 150, 243, 0.7)'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          title: {
            display: true,
            text: 'LIME vs SHAP Feature Importance Comparison'
          }
        }
      }
    });
    """
  end
end

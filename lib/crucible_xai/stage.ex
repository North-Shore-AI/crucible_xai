defmodule CrucibleXAI.Stage do
  @moduledoc """
  Pipeline stage for explainable AI analysis.

  Implements a Crucible pipeline stage that provides LIME, SHAP, and feature
  attribution explanations for model outputs. Uses CrucibleIR configuration
  for flexible XAI method selection.

  ## Context Requirements

  The context map must contain:
  - `model_fn` - Prediction function that takes an instance and returns a prediction
  - `instances` - List of instances to explain, or single instance
  - `experiment.reliability.xai` (optional) - XAI configuration

  For SHAP methods, also requires:
  - `background_data` - Background dataset for baseline computation

  ## Returns

  Updated context with `:xai` key containing explanation results.

  ## Example

      # Context from pipeline
      context = %{
        model_fn: fn [x, y] -> 2.0 * x + 3.0 * y end,
        instances: [[1.0, 2.0], [3.0, 4.0]],
        background_data: [[0.0, 0.0], [1.0, 1.0]],
        experiment: %{
          reliability: %{
            xai: %{
              methods: [:lime, :shap],
              lime_opts: %{num_samples: 1000},
              shap_opts: %{num_samples: 500}
            }
          }
        }
      }

      {:ok, updated_context} = CrucibleXAI.Stage.run(context)
      # updated_context.xai contains explanation results
  """

  alias CrucibleXAI.{FeatureAttribution, LIME, SHAP}

  # Note: We define the callback functions but don't use @behaviour since
  # crucible_framework may not be a dependency. The framework will call these
  # functions dynamically.

  @doc """
  Runs XAI analysis on model outputs.

  Accepts a context map with model function, instances, and optional XAI
  configuration. Generates explanations using LIME, SHAP, and/or feature
  attribution methods.

  ## Options

  Options can be provided to override context config:
  - `:methods` - List of methods to run (default: [:lime])
    - Supported: `:lime`, `:shap`, `:kernel_shap`, `:linear_shap`, `:sampling_shap`, `:feature_importance`
  - `:lime_opts` - Options for LIME (num_samples, kernel, etc.)
  - `:shap_opts` - Options for SHAP (num_samples, method, etc.)
  - `:feature_importance_opts` - Options for permutation importance
  - `:instance_key` - Key to extract instances from context (default: `:instances`)
  - `:parallel` - Run batch explanations in parallel (default: false)

  ## Returns

  - `{:ok, context}` - Updated context with XAI results
  - `{:error, reason}` - If configuration or data is missing/invalid
  """
  @spec run(map(), map()) :: {:ok, map()} | {:error, String.t()}
  def run(context, opts \\ %{}) when is_map(context) do
    with {:ok, model_fn} <- extract_model_fn(context),
         {:ok, instances} <- extract_instances(context, opts),
         {:ok, config} <- extract_config(context, opts),
         {:ok, results} <- run_xai_methods(instances, model_fn, context, config) do
      {:ok, Map.put(context, :xai, results)}
    end
  end

  @doc """
  Describes this stage for introspection.

  Returns metadata about the stage including its purpose, requirements, and
  configuration options.

  ## Options

  - `:verbose` - Include detailed information (default: false)
  """
  @dialyzer {:nowarn_function, describe: 1}
  @spec describe(map()) :: %{atom() => any()}
  def describe(opts \\ %{}) do
    verbose = Map.get(opts, :verbose, false)

    base = %{
      name: "CrucibleXAI.Stage",
      type: :analysis,
      purpose: "Explainable AI analysis with LIME, SHAP, and feature attribution",
      inputs: [:model_fn, :instances, :background_data],
      outputs: [:xai],
      config_source: "experiment.reliability.xai or opts"
    }

    if verbose do
      Map.merge(base, %{
        available_methods: [
          :lime,
          :shap,
          :kernel_shap,
          :linear_shap,
          :sampling_shap,
          :feature_importance
        ],
        lime_capabilities: [
          "Local interpretable model-agnostic explanations",
          "Multiple sampling strategies (Gaussian, Uniform, Categorical)",
          "Flexible kernel functions",
          "Feature selection methods"
        ],
        shap_capabilities: [
          "KernelSHAP: Model-agnostic approximation",
          "LinearSHAP: Exact for linear models",
          "SamplingShap: Monte Carlo approximation",
          "Shapley value guarantees (additivity, symmetry)"
        ],
        requirements: [
          "model_fn: Prediction function",
          "instances: List of instances to explain",
          "background_data: Required for SHAP methods"
        ]
      })
    else
      base
    end
  end

  # Private Functions

  defp extract_model_fn(context) do
    case Map.get(context, :model_fn) do
      nil -> {:error, "Missing model_fn in context"}
      fn_val when is_function(fn_val) -> {:ok, fn_val}
      other -> {:error, "model_fn must be a function, got: #{inspect(other)}"}
    end
  end

  defp extract_instances(context, opts) do
    instance_key = Map.get(opts, :instance_key, :instances)

    case Map.get(context, instance_key) do
      nil ->
        # Try singular :instance
        case Map.get(context, :instance) do
          nil ->
            {:error, "No instances found in context (tried #{instance_key} and :instance)"}

          instance when is_list(instance) ->
            {:ok, [instance]}

          other ->
            {:error, "Instance must be a list, got: #{inspect(other)}"}
        end

      instances when is_list(instances) ->
        # Check if it's a single instance (list of numbers) or multiple instances
        if is_list(hd(instances)) or not is_number(hd(instances)) do
          {:ok, instances}
        else
          # Single instance as a list of numbers
          {:ok, [instances]}
        end

      other ->
        {:error, "Instances must be a list, got: #{inspect(other)}"}
    end
  end

  defp extract_config(context, opts) do
    # Get config from context or use defaults
    xai_config = get_in(context, [:experiment, :reliability, :xai]) || %{}

    config = %{
      methods: get_config_value(opts, xai_config, :methods, [:lime]),
      lime_opts: get_config_value(opts, xai_config, :lime_opts, %{}),
      shap_opts: get_config_value(opts, xai_config, :shap_opts, %{}),
      feature_importance_opts: get_config_value(opts, xai_config, :feature_importance_opts, %{}),
      parallel: get_config_value(opts, xai_config, :parallel, false)
    }

    {:ok, config}
  end

  defp get_config_value(opts, xai_config, key, default) do
    Map.get(opts, key) || Map.get(xai_config, key) || default
  end

  defp run_xai_methods(instances, model_fn, context, config) do
    methods = List.wrap(config.methods)

    results = %{
      methods_run: methods,
      explanations: %{},
      metadata: %{
        num_instances: length(instances),
        timestamp: DateTime.utc_now()
      }
    }

    # Run each method
    results =
      Enum.reduce(methods, results, fn method, acc ->
        case execute_method(method, instances, model_fn, context, config) do
          {:ok, method_result} ->
            put_in(acc, [:explanations, method], method_result)

          {:error, reason} ->
            put_in(acc, [:explanations, method], %{error: reason})
        end
      end)

    {:ok, results}
  end

  defp execute_method(:lime, instances, model_fn, _context, config) do
    lime_opts = Map.to_list(config.lime_opts)

    if config.parallel do
      explanations = LIME.explain_batch(instances, model_fn, lime_opts ++ [parallel: true])
      {:ok, %{method: :lime, explanations: explanations, count: length(explanations)}}
    else
      explanations = LIME.explain_batch(instances, model_fn, lime_opts)
      {:ok, %{method: :lime, explanations: explanations, count: length(explanations)}}
    end
  end

  defp execute_method(method, instances, model_fn, context, config)
       when method in [:shap, :kernel_shap, :linear_shap, :sampling_shap] do
    case Map.get(context, :background_data) do
      nil ->
        {:error, "SHAP methods require background_data in context"}

      background_data ->
        execute_shap_method(method, instances, background_data, model_fn, config)
    end
  end

  defp execute_method(:feature_importance, instances, model_fn, _context, config) do
    # Feature importance requires labeled data
    # For now, we'll create mock labels based on predictions
    # In a real pipeline, labels would come from context
    validation_data =
      Enum.map(instances, fn instance ->
        label = model_fn.(instance)
        {instance, label}
      end)

    fi_opts = Map.to_list(config.feature_importance_opts)

    importance = FeatureAttribution.permutation_importance(model_fn, validation_data, fi_opts)
    {:ok, %{method: :feature_importance, importance: importance}}
  end

  defp execute_method(method, _instances, _model_fn, _context, _config) do
    {:error, "Unknown XAI method: #{method}"}
  end

  defp execute_shap_method(method, instances, background_data, model_fn, config) do
    shap_opts = Map.to_list(config.shap_opts)
    # Add method-specific option
    shap_opts =
      case method do
        :linear_shap -> shap_opts ++ [method: :linear_shap]
        :sampling_shap -> shap_opts ++ [method: :sampling_shap]
        _ -> shap_opts
      end

    if method == :linear_shap and not Keyword.has_key?(shap_opts, :coefficients) do
      {:error, "linear_shap requires :coefficients in shap_opts (for known linear models only)"}
    else
      # Run SHAP for each instance
      shap_values =
        Enum.map(instances, fn instance ->
          SHAP.explain(instance, background_data, model_fn, shap_opts)
        end)

      {:ok, %{method: method, shap_values: shap_values, count: length(shap_values)}}
    end
  end
end

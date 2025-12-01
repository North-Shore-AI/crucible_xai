defmodule CrucibleXAI.StageTest do
  use ExUnit.Case, async: true

  alias CrucibleXAI.Stage

  describe "describe/1" do
    test "returns basic stage metadata" do
      meta = Stage.describe()

      assert meta.name == "CrucibleXAI.Stage"
      assert meta.type == :analysis
      assert meta.purpose =~ "Explainable AI"
      assert :model_fn in meta.inputs
      assert :instances in meta.inputs
      assert :xai in meta.outputs
    end

    test "returns verbose metadata when requested" do
      meta = Stage.describe(%{verbose: true})

      assert is_list(meta.available_methods)
      assert :lime in meta.available_methods
      assert :shap in meta.available_methods
      assert is_list(meta.lime_capabilities)
      assert is_list(meta.shap_capabilities)
      assert is_list(meta.requirements)
    end
  end

  describe "run/2 with LIME" do
    test "runs LIME on a single instance" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 2.0]

      context = %{
        model_fn: model_fn,
        instance: instance
      }

      # Use fewer samples for faster tests
      opts = %{methods: [:lime], lime_opts: %{num_samples: 100}}
      assert {:ok, updated_context} = Stage.run(context, opts)
      assert Map.has_key?(updated_context, :xai)
      assert updated_context.xai.methods_run == [:lime]
      assert Map.has_key?(updated_context.xai.explanations, :lime)

      lime_result = updated_context.xai.explanations.lime
      assert lime_result.method == :lime
      assert lime_result.count == 1
      assert is_list(lime_result.explanations)
      assert length(lime_result.explanations) == 1

      explanation = hd(lime_result.explanations)
      assert explanation.method == :lime
      assert is_map(explanation.feature_weights)
    end

    test "runs LIME on multiple instances" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

      context = %{
        model_fn: model_fn,
        instances: instances
      }

      # Use fewer samples for faster tests
      opts = %{methods: [:lime], lime_opts: %{num_samples: 100}}
      assert {:ok, updated_context} = Stage.run(context, opts)

      lime_result = updated_context.xai.explanations.lime
      assert lime_result.count == 3
      assert length(lime_result.explanations) == 3

      # All should be valid explanations
      for explanation <- lime_result.explanations do
        assert explanation.method == :lime
        assert is_map(explanation.feature_weights)
        assert is_number(explanation.score)
      end
    end

    test "passes LIME options through to explainer" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 2.0]

      context = %{
        model_fn: model_fn,
        instance: instance
      }

      opts = %{
        methods: [:lime],
        lime_opts: %{num_samples: 500, kernel: :cosine}
      }

      assert {:ok, updated_context} = Stage.run(context, opts)

      lime_result = updated_context.xai.explanations.lime
      explanation = hd(lime_result.explanations)

      # Check metadata contains the options
      assert explanation.metadata.num_samples == 500
      assert explanation.metadata.kernel == :cosine
    end

    test "supports parallel batch processing" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]

      context = %{
        model_fn: model_fn,
        instances: instances
      }

      # Use fewer samples for faster tests
      opts = %{
        methods: [:lime],
        lime_opts: %{num_samples: 100},
        parallel: true
      }

      assert {:ok, updated_context} = Stage.run(context, opts)

      lime_result = updated_context.xai.explanations.lime
      assert lime_result.count == 4
      assert length(lime_result.explanations) == 4
    end
  end

  describe "run/2 with SHAP" do
    test "runs SHAP on a single instance" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background_data = [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]

      context = %{
        model_fn: model_fn,
        instance: instance,
        background_data: background_data
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:shap]})

      shap_result = updated_context.xai.explanations.shap
      assert shap_result.method == :shap
      assert shap_result.count == 1
      assert is_list(shap_result.shap_values)
      assert length(shap_result.shap_values) == 1

      shap_vals = hd(shap_result.shap_values)
      assert is_map(shap_vals)
      assert Map.has_key?(shap_vals, 0)
      assert Map.has_key?(shap_vals, 1)
    end

    test "runs SHAP on multiple instances" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 1.0], [2.0, 2.0]]
      background_data = [[0.0, 0.0], [1.0, 1.0]]

      context = %{
        model_fn: model_fn,
        instances: instances,
        background_data: background_data
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:shap]})

      shap_result = updated_context.xai.explanations.shap
      assert shap_result.count == 2
      assert length(shap_result.shap_values) == 2
    end

    test "returns error when background_data is missing for SHAP" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]

      context = %{
        model_fn: model_fn,
        instance: instance
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:shap]})

      shap_result = updated_context.xai.explanations.shap
      assert Map.has_key?(shap_result, :error)
      assert shap_result.error =~ "background_data"
    end

    test "supports linear_shap method variant with coefficients" do
      # linear_shap is for known linear models - must provide coefficients as a map
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background_data = [[0.0, 0.0], [1.0, 1.0]]

      context = %{
        model_fn: model_fn,
        instance: instance,
        background_data: background_data
      }

      # Provide the known coefficients for the linear model (as a map: index => coefficient)
      opts = %{
        methods: [:linear_shap],
        shap_opts: %{coefficients: %{0 => 2.0, 1 => 3.0}, intercept: 0.0}
      }

      assert {:ok, updated_context} = Stage.run(context, opts)

      shap_result = updated_context.xai.explanations.linear_shap
      assert shap_result.method == :linear_shap
      assert is_list(shap_result.shap_values)
    end

    test "returns error when linear_shap is used without coefficients" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background_data = [[0.0, 0.0], [1.0, 1.0]]

      context = %{
        model_fn: model_fn,
        instance: instance,
        background_data: background_data
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:linear_shap]})

      shap_result = updated_context.xai.explanations.linear_shap
      assert Map.has_key?(shap_result, :error)
      assert shap_result.error =~ "coefficients"
    end

    test "supports sampling_shap method variant" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 1.0]
      background_data = [[0.0, 0.0], [1.0, 1.0]]

      context = %{
        model_fn: model_fn,
        instance: instance,
        background_data: background_data
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:sampling_shap]})

      shap_result = updated_context.xai.explanations.sampling_shap
      assert shap_result.method == :sampling_shap
      assert is_list(shap_result.shap_values)
    end
  end

  describe "run/2 with feature importance" do
    test "computes permutation importance" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

      context = %{
        model_fn: model_fn,
        instances: instances
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:feature_importance]})

      fi_result = updated_context.xai.explanations.feature_importance
      assert fi_result.method == :feature_importance
      assert is_map(fi_result.importance)
      assert Map.has_key?(fi_result.importance, 0)
      assert Map.has_key?(fi_result.importance, 1)
    end

    test "passes feature importance options" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

      context = %{
        model_fn: model_fn,
        instances: instances
      }

      opts = %{
        methods: [:feature_importance],
        feature_importance_opts: %{num_repeats: 5, metric: :mae}
      }

      assert {:ok, updated_context} = Stage.run(context, opts)

      fi_result = updated_context.xai.explanations.feature_importance
      assert is_map(fi_result.importance)
    end
  end

  describe "run/2 with multiple methods" do
    test "runs multiple XAI methods together" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0]]
      background_data = [[0.0, 0.0], [1.0, 1.0]]

      context = %{
        model_fn: model_fn,
        instances: instances,
        background_data: background_data
      }

      # Use fewer samples for faster tests
      opts = %{
        methods: [:lime, :shap, :feature_importance],
        lime_opts: %{num_samples: 100},
        shap_opts: %{num_samples: 100}
      }

      assert {:ok, updated_context} = Stage.run(context, opts)

      assert updated_context.xai.methods_run == [:lime, :shap, :feature_importance]
      assert Map.has_key?(updated_context.xai.explanations, :lime)
      assert Map.has_key?(updated_context.xai.explanations, :shap)
      assert Map.has_key?(updated_context.xai.explanations, :feature_importance)
    end
  end

  describe "run/2 with configuration from context" do
    test "reads XAI config from experiment.reliability.xai" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 2.0]

      context = %{
        model_fn: model_fn,
        instance: instance,
        experiment: %{
          reliability: %{
            xai: %{
              methods: [:lime],
              lime_opts: %{num_samples: 200}
            }
          }
        }
      }

      assert {:ok, updated_context} = Stage.run(context)

      lime_result = updated_context.xai.explanations.lime
      explanation = hd(lime_result.explanations)
      assert explanation.metadata.num_samples == 200
    end
  end

  describe "run/2 error cases" do
    test "returns error when model_fn is missing" do
      context = %{
        instance: [1.0, 2.0]
      }

      assert {:error, reason} = Stage.run(context)
      assert reason =~ "model_fn"
    end

    test "returns error when model_fn is not a function" do
      context = %{
        model_fn: "not a function",
        instance: [1.0, 2.0]
      }

      assert {:error, reason} = Stage.run(context)
      assert reason =~ "must be a function"
    end

    test "returns error when instances are missing" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end

      context = %{
        model_fn: model_fn
      }

      assert {:error, reason} = Stage.run(context)
      assert reason =~ "No instances found"
    end

    test "handles unknown XAI method gracefully" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 2.0]

      context = %{
        model_fn: model_fn,
        instance: instance
      }

      assert {:ok, updated_context} = Stage.run(context, %{methods: [:unknown_method]})

      unknown_result = updated_context.xai.explanations.unknown_method
      assert Map.has_key?(unknown_result, :error)
      assert unknown_result.error =~ "Unknown XAI method"
    end
  end

  describe "run/2 metadata" do
    test "includes metadata in results" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instances = [[1.0, 2.0], [3.0, 4.0]]

      context = %{
        model_fn: model_fn,
        instances: instances
      }

      # Use fewer samples for faster tests
      opts = %{methods: [:lime], lime_opts: %{num_samples: 100}}
      assert {:ok, updated_context} = Stage.run(context, opts)

      meta = updated_context.xai.metadata
      assert meta.num_instances == 2
      assert %DateTime{} = meta.timestamp
    end

    test "includes methods_run in results" do
      model_fn = fn [x, y] -> 2.0 * x + 3.0 * y end
      instance = [1.0, 2.0]

      context = %{
        model_fn: model_fn,
        instance: instance
      }

      # Use fewer samples for faster tests
      opts = %{methods: [:lime], lime_opts: %{num_samples: 100}}
      assert {:ok, updated_context} = Stage.run(context, opts)
      assert updated_context.xai.methods_run == [:lime]
    end
  end
end

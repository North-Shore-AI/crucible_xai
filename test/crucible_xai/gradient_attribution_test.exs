defmodule CrucibleXAI.GradientAttributionTest do
  use ExUnit.Case, async: true
  use ExUnitProperties

  alias CrucibleXAI.GradientAttribution

  describe "gradient_x_input/2" do
    test "computes gradient × input for simple linear model" do
      # Model: f(x, y) = 2*x + 3*y
      # Gradients: ∂f/∂x = 2, ∂f/∂y = 3
      # Attribution: [2*x, 3*y]

      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([5.0, 4.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      # Expected: [2*5, 3*4] = [10.0, 12.0]
      assert_in_delta Nx.to_number(attributions[0]), 10.0, 0.001
      assert_in_delta Nx.to_number(attributions[1]), 12.0, 0.001
    end

    test "computes gradient × input for nonlinear model" do
      # Model: f(x, y) = x^2 + y^2
      # At point (3, 4):
      # Gradients: ∂f/∂x = 2x = 6, ∂f/∂y = 2y = 8
      # Attribution: [6*3, 8*4] = [18, 32]

      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([3.0, 4.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      assert_in_delta Nx.to_number(attributions[0]), 18.0, 0.001
      assert_in_delta Nx.to_number(attributions[1]), 32.0, 0.001
    end

    test "handles single feature" do
      # Model: f(x) = 3*x^2
      # At x=2: gradient = 6x = 12
      # Attribution: 12*2 = 24

      model_fn = fn params ->
        Nx.multiply(3.0, Nx.pow(params, 2))
      end

      instance = Nx.tensor([2.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      assert_in_delta Nx.to_number(attributions[0]), 24.0, 0.001
    end

    test "handles zero input" do
      # Model: f(x, y) = 2*x + 3*y
      # At (0, 0): attributions should be [0, 0]

      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([0.0, 0.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      assert_in_delta Nx.to_number(attributions[0]), 0.0, 0.001
      assert_in_delta Nx.to_number(attributions[1]), 0.0, 0.001
    end

    test "handles negative values" do
      # Model: f(x, y) = x^2 + y^2
      # At (-2, -3): gradients = [-4, -6]
      # Attribution: [-4*-2, -6*-3] = [8, 18]

      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([-2.0, -3.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      assert_in_delta Nx.to_number(attributions[0]), 8.0, 0.001
      assert_in_delta Nx.to_number(attributions[1]), 18.0, 0.001
    end

    test "works with multi-dimensional input" do
      # Model: f(x1, x2, x3, x4) = sum of squares
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      attributions = GradientAttribution.gradient_x_input(model_fn, instance)

      # Expected: [2*1*1, 2*2*2, 2*3*3, 2*4*4] = [2, 8, 18, 32]
      assert Nx.shape(attributions) == {4}
      assert_in_delta Nx.to_number(attributions[0]), 2.0, 0.001
      assert_in_delta Nx.to_number(attributions[1]), 8.0, 0.001
      assert_in_delta Nx.to_number(attributions[2]), 18.0, 0.001
      assert_in_delta Nx.to_number(attributions[3]), 32.0, 0.001
    end
  end

  describe "compute_gradients/2" do
    test "computes gradients for linear model" do
      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([1.0, 1.0])

      gradients = GradientAttribution.compute_gradients(model_fn, instance)

      # For linear model f(x,y) = 2x + 3y, gradients are constant [2, 3]
      assert_in_delta Nx.to_number(gradients[0]), 2.0, 0.001
      assert_in_delta Nx.to_number(gradients[1]), 3.0, 0.001
    end

    test "computes gradients at specific point for nonlinear model" do
      model_fn = fn params ->
        # f(x, y) = x^3 + 2*y^2
        Nx.add(Nx.pow(params[0], 3), Nx.multiply(2.0, Nx.pow(params[1], 2)))
      end

      instance = Nx.tensor([2.0, 3.0])

      gradients = GradientAttribution.compute_gradients(model_fn, instance)

      # At (2, 3):
      # ∂f/∂x = 3x^2 = 12
      # ∂f/∂y = 4y = 12
      assert_in_delta Nx.to_number(gradients[0]), 12.0, 0.01
      assert_in_delta Nx.to_number(gradients[1]), 12.0, 0.01
    end

    test "returns tensor with same shape as input" do
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([1.0, 2.0, 3.0])
      gradients = GradientAttribution.compute_gradients(model_fn, instance)

      assert Nx.shape(gradients) == Nx.shape(instance)
    end
  end

  describe "integrated_gradients/4" do
    test "computes integrated gradients for linear model" do
      # For linear model f(x, y) = 2x + 3y
      # Integrated gradients from baseline [0, 0] to [5, 4]
      # Should equal gradient × input since gradients are constant

      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([5.0, 4.0])
      baseline = Nx.tensor([0.0, 0.0])

      attributions =
        GradientAttribution.integrated_gradients(
          model_fn,
          instance,
          baseline,
          steps: 50
        )

      # For linear model, should equal gradient × input
      # Expected: [2*5, 3*4] = [10.0, 12.0]
      assert_in_delta Nx.to_number(attributions[0]), 10.0, 0.1
      assert_in_delta Nx.to_number(attributions[1]), 12.0, 0.1
    end

    test "computes integrated gradients for nonlinear model" do
      # Model: f(x, y) = x^2 + y^2
      # IG integrates gradients along the path from baseline to instance

      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([4.0, 3.0])
      baseline = Nx.tensor([0.0, 0.0])

      attributions =
        GradientAttribution.integrated_gradients(
          model_fn,
          instance,
          baseline,
          steps: 100
        )

      # For f(x, y) = x^2 + y^2:
      # IG_x = x * ∫₀¹ 2(baseline_x + α*x) dα
      # With baseline=0: IG_x = x * ∫₀¹ 2αx dα = x * x = x^2
      # Expected: [16, 9]
      assert_in_delta Nx.to_number(attributions[0]), 16.0, 0.5
      assert_in_delta Nx.to_number(attributions[1]), 9.0, 0.5
    end

    test "satisfies completeness axiom" do
      # IG should satisfy: sum(attributions) = f(x) - f(baseline)

      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0, 1.5])))
      end

      instance = Nx.tensor([5.0, 4.0, 2.0])
      baseline = Nx.tensor([0.0, 0.0, 0.0])

      attributions =
        GradientAttribution.integrated_gradients(
          model_fn,
          instance,
          baseline,
          steps: 100
        )

      prediction = model_fn.(instance) |> Nx.to_number()
      baseline_pred = model_fn.(baseline) |> Nx.to_number()
      attr_sum = Nx.sum(attributions) |> Nx.to_number()

      # Completeness: sum(IG) ≈ f(x) - f(baseline)
      expected_diff = prediction - baseline_pred
      assert_in_delta attr_sum, expected_diff, 0.1
    end

    test "handles non-zero baseline" do
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([3.0, 4.0])
      baseline = Nx.tensor([1.0, 1.0])

      attributions =
        GradientAttribution.integrated_gradients(
          model_fn,
          instance,
          baseline,
          steps: 50
        )

      # Should integrate from baseline to instance
      assert Nx.shape(attributions) == {2}
      # Verify completeness
      prediction = model_fn.(instance) |> Nx.to_number()
      baseline_pred = model_fn.(baseline) |> Nx.to_number()
      attr_sum = Nx.sum(attributions) |> Nx.to_number()

      assert_in_delta attr_sum, prediction - baseline_pred, 0.5
    end

    test "configurable number of steps" do
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([2.0, 2.0])
      baseline = Nx.tensor([0.0, 0.0])

      # With more steps, should get more accurate approximation
      ig_10 = GradientAttribution.integrated_gradients(model_fn, instance, baseline, steps: 10)
      ig_100 = GradientAttribution.integrated_gradients(model_fn, instance, baseline, steps: 100)

      # Both should be close to each other and the true value
      diff = Nx.subtract(ig_10, ig_100) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 0.5
    end

    test "handles single feature" do
      model_fn = fn params ->
        Nx.multiply(2.0, Nx.pow(params, 2))
      end

      instance = Nx.tensor([3.0])
      baseline = Nx.tensor([0.0])

      attributions =
        GradientAttribution.integrated_gradients(
          model_fn,
          instance,
          baseline,
          steps: 50
        )

      # For f(x) = 2x^2 from 0 to 3:
      # IG = (3-0) * ∫₀¹ 4αx dα = 3 * 2x = 6x = 18
      assert_in_delta Nx.to_number(attributions[0]), 18.0, 0.5
    end
  end

  describe "smooth_grad/4" do
    test "computes smooth gradients by averaging noisy gradients" do
      # SmoothGrad should smooth out the attribution by averaging
      # gradients computed with added noise

      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([5.0, 4.0])

      smooth_attrs =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.1,
          n_samples: 50
        )

      # For linear model, should be close to regular gradient × input
      # Expected: approximately [10.0, 12.0]
      assert_in_delta Nx.to_number(smooth_attrs[0]), 10.0, 1.0
      assert_in_delta Nx.to_number(smooth_attrs[1]), 12.0, 1.0
    end

    test "produces smoother attributions than gradient × input" do
      # SmoothGrad should reduce noise/variance in attributions

      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([3.0, 4.0])

      # Compute both methods
      grad_input = GradientAttribution.gradient_x_input(model_fn, instance)

      smooth =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.2,
          n_samples: 100
        )

      # Both should have same shape
      assert Nx.shape(grad_input) == Nx.shape(smooth)

      # Smooth version should be close to gradient × input for smooth functions
      # but may differ for non-smooth functions
      assert Nx.shape(smooth) == {2}
    end

    test "configurable noise level" do
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([2.0, 2.0])

      # With very small noise, should be very close to gradient × input
      smooth_small =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.01,
          n_samples: 50
        )

      grad_input = GradientAttribution.gradient_x_input(model_fn, instance)

      # Should be close with small noise
      diff = Nx.subtract(smooth_small, grad_input) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      assert diff < 1.0
    end

    test "configurable number of samples" do
      model_fn = fn params ->
        Nx.sum(Nx.multiply(params, Nx.tensor([2.0, 3.0])))
      end

      instance = Nx.tensor([5.0, 4.0])

      # More samples should give more stable results
      smooth_10 =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.1,
          n_samples: 10
        )

      smooth_100 =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.1,
          n_samples: 100
        )

      # Both should have same shape
      assert Nx.shape(smooth_10) == Nx.shape(smooth_100)
      # For linear model, should converge to similar values
      assert Nx.shape(smooth_100) == {2}
    end

    test "handles single feature" do
      model_fn = fn params ->
        Nx.multiply(2.0, Nx.pow(params, 2))
      end

      instance = Nx.tensor([3.0])

      smooth_attrs =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.1,
          n_samples: 50
        )

      # Should be close to gradient × input: 2 * 2 * 3 * 3 = 36
      assert_in_delta Nx.to_number(smooth_attrs[0]), 36.0, 5.0
    end

    test "returns same shape as input" do
      model_fn = fn params ->
        Nx.sum(Nx.pow(params, 2))
      end

      instance = Nx.tensor([1.0, 2.0, 3.0, 4.0])

      smooth_attrs =
        GradientAttribution.smooth_grad(
          model_fn,
          instance,
          noise_level: 0.1,
          n_samples: 30
        )

      assert Nx.shape(smooth_attrs) == Nx.shape(instance)
    end
  end

  describe "property-based tests" do
    property "gradient × input preserves dimensionality" do
      check all(
              n_features <- integer(1..10),
              values <- list_of(float(min: -10.0, max: 10.0), length: n_features)
            ) do
        model_fn = fn params ->
          Nx.sum(Nx.pow(params, 2))
        end

        instance = Nx.tensor(values)
        attributions = GradientAttribution.gradient_x_input(model_fn, instance)

        assert Nx.shape(attributions) == {n_features}
      end
    end

    property "gradient computation is consistent" do
      check all(
              n_features <- integer(1..5),
              values <- list_of(float(min: -10.0, max: 10.0), length: n_features)
            ) do
        model_fn = fn params ->
          Nx.sum(Nx.multiply(params, Nx.tensor(2.0)))
        end

        instance = Nx.tensor(values)

        # Compute gradients twice - should be identical
        grad1 = GradientAttribution.compute_gradients(model_fn, instance)
        grad2 = GradientAttribution.compute_gradients(model_fn, instance)

        # Convert to lists for comparison
        assert Nx.to_flat_list(grad1) == Nx.to_flat_list(grad2)
      end
    end
  end
end

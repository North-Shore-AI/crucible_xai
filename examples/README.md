# CrucibleXAI Examples

This directory contains comprehensive, runnable examples demonstrating the capabilities of CrucibleXAI.

## Running the Examples

All examples can be run directly with `mix run`:

```bash
# From the crucible_xai root directory
mix run examples/01_basic_lime.exs
mix run examples/02_customized_lime.exs
# ... etc
```

Or run all examples:

```bash
for file in examples/*.exs; do mix run "$file"; done
```

## Example List

### 1. Basic LIME (`01_basic_lime.exs`)
**What it demonstrates:**
- Basic LIME explanation workflow
- Understanding feature weights
- Interpreting explanation results
- Text output formatting

**Key concepts:**
- LIME basics
- Feature importance
- Local linear approximation
- R² score interpretation

**Run time:** ~1 second

---

### 2. Customized LIME (`02_customized_lime.exs`)
**What it demonstrates:**
- Fine-tuning LIME parameters
- Different sampling methods
- Various kernel functions
- Feature selection strategies

**Key concepts:**
- Parameter optimization
- Kernel width tuning
- Feature selection methods (lasso, forward selection, highest weights)
- Trade-offs between accuracy and computation time

**Run time:** ~3-5 seconds

---

### 3. Batch Explanations (`03_batch_explanations.exs`)
**What it demonstrates:**
- Efficient batch processing
- Consistency analysis across instances
- Performance metrics
- Statistical summaries

**Key concepts:**
- Batch processing efficiency
- Feature importance consistency
- Quality metrics across multiple instances
- Average R² scores

**Run time:** ~2-3 seconds

---

### 4. SHAP Explanations (`04_shap_explanations.exs`)
**What it demonstrates:**
- SHAP (Shapley values) computation
- Additivity property verification
- LIME vs SHAP comparison
- When to use each method

**Key concepts:**
- Shapley values
- Game theory-based attribution
- Additivity property
- SHAP vs LIME trade-offs

**Run time:** ~3-4 seconds

---

### 5. Feature Importance (`05_feature_importance.exs`)
**What it demonstrates:**
- Permutation importance calculation
- Global feature ranking
- Statistical validation
- Top-k feature selection

**Key concepts:**
- Global vs local importance
- Permutation importance
- Feature ranking
- Confidence intervals

**Run time:** ~2 seconds

---

### 6. Visualization (`06_visualization.exs`)
**What it demonstrates:**
- HTML visualization generation
- LIME charts
- SHAP charts
- Comparison visualizations

**Key concepts:**
- Interactive visualizations
- Chart.js integration
- HTML export
- Feature name customization

**Run time:** ~2 seconds
**Output:** Creates HTML files in `examples/output/`

---

### 7. Model Debugging (`07_model_debugging.exs`)
**What it demonstrates:**
- Using XAI for debugging
- Detecting unexpected feature importance
- Identifying data leakage
- Bias detection

**Key concepts:**
- Model debugging techniques
- Data leakage detection
- Feature importance analysis
- Bias identification

**Run time:** ~2-3 seconds

---

### 8. Model Comparison (`08_model_comparison.exs`)
**What it demonstrates:**
- Comparing different models
- Feature importance differences
- Model strategy analysis
- Selection criteria

**Key concepts:**
- Model comparison methodology
- Feature weight analysis
- Strategy differences
- Model selection

**Run time:** ~2 seconds

---

### 9. Nonlinear Model (`09_nonlinear_model.exs`)
**What it demonstrates:**
- Explaining complex nonlinear models
- Local approximation quality
- Gradient interpretation
- R² score variation

**Key concepts:**
- Nonlinear model challenges
- Local linearity
- Tangent plane approximation
- Context-dependent explanations

**Run time:** ~3-4 seconds

---

### 10. Complete Workflow (`10_complete_workflow.exs`)
**What it demonstrates:**
- End-to-end XAI workflow
- Multiple analysis techniques
- Quality metrics
- Best practices

**Key concepts:**
- Complete analysis pipeline
- Multi-method approach
- Quality assurance
- Comprehensive reporting

**Run time:** ~5-7 seconds
**Output:** Creates visualizations in `examples/output/workflow/`

---

## Example Output

Each example produces clear, formatted output to the console. Examples 6 and 10 also generate HTML visualizations in the `examples/output/` directory.

### Viewing Visualizations

After running examples that generate HTML:

```bash
# Open in your browser
firefox examples/output/lime_explanation.html
# or
xdg-open examples/output/lime_explanation.html
```

## Learning Path

We recommend following this order:

1. **Beginners:** Start with examples 1-3 to understand LIME basics
2. **Intermediate:** Move to examples 4-6 for SHAP and visualizations
3. **Advanced:** Explore examples 7-9 for debugging and analysis
4. **Comprehensive:** Run example 10 to see everything together

## Common Use Cases

### Model Development
- Example 1: Quick explanation during development
- Example 2: Parameter tuning
- Example 7: Debugging unexpected behavior

### Model Evaluation
- Example 3: Batch analysis
- Example 5: Global feature importance
- Example 8: Comparing model versions

### Production Deployment
- Example 4: SHAP for compliance requirements
- Example 6: Generating reports
- Example 10: Complete workflow automation

### Research & Analysis
- Example 9: Understanding complex models
- Example 7: Bias detection
- Example 8: Model comparison studies

## Prerequisites

All examples work with the base CrucibleXAI installation:

```elixir
# mix.exs
defp deps do
  [
    {:crucible_xai, github: "North-Shore-AI/crucible_xai"}
  ]
end
```

No additional dependencies required!

## Performance Notes

- Examples use reduced sample counts for speed
- Production usage should increase `num_samples` for better accuracy
- LIME: ~50ms per explanation with 5000 samples
- SHAP: ~1s per explanation with 2000 samples
- Batch processing is parallelizable

## Customization

All examples can be easily customized:

1. **Change the model:** Replace `predict_fn` with your model
2. **Change the data:** Modify `instance` or `instances`
3. **Tune parameters:** Adjust `num_samples`, `kernel_width`, etc.
4. **Add features:** Extend `feature_names` mapping

## Troubleshooting

### Example doesn't run
```bash
# Ensure you're in the project root
cd /path/to/crucible_xai

# Compile the project first
mix compile

# Then run
mix run examples/01_basic_lime.exs
```

### Low R² scores
- Increase `num_samples` parameter
- Adjust `kernel_width` for better locality
- Check if model is highly nonlinear at that point

### Slow performance
- Reduce `num_samples` for faster (less accurate) results
- Use batch processing for multiple instances
- Consider using LIME instead of SHAP for speed

## Contributing

To add a new example:

1. Create `NN_example_name.exs` in this directory
2. Follow the existing format with clear sections
3. Add documentation to this README
4. Test thoroughly with `mix run examples/NN_example_name.exs`

## Support

For questions or issues:
- GitHub Issues: https://github.com/North-Shore-AI/crucible_xai/issues
- Documentation: See main README.md and docs/ directory

## License

These examples are part of CrucibleXAI and are released under the MIT License.

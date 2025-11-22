"""
Quick example showing how TSU visualizations work.
Creates a simple uncertainty plot showing Bayesian predictions.
"""

import numpy as np
from tsu import plot_predictions_with_uncertainty

# Generate some fake data - sine wave with noise
x_train = np.array([0, 1, 2, 3, 7, 8, 9, 10])
y_train = np.sin(x_train) + np.random.randn(len(x_train)) * 0.1

# Test points - dense sampling
x_test = np.linspace(-1, 11, 100)
y_true = np.sin(x_test)

# Simulate predictions with uncertainty
# Note: High uncertainty far from training data (between 3-7)
y_pred = np.sin(x_test)
distance_to_train = np.min([np.abs(x_test - xt)[:, None] for xt in x_train], axis=0).flatten()
y_std = 0.1 + 0.4 * np.tanh(distance_to_train / 2)  # Uncertainty increases with distance

# Create the visualization
fig = plot_predictions_with_uncertainty(
    x=x_test,
    y_pred=y_pred,
    y_std=y_std,
    y_true=y_true,
    x_train=x_train,
    y_train=y_train,
    title="Bayesian Prediction with Epistemic Uncertainty",
    xlabel="x",
    ylabel="sin(x)",
    save_path="visual_output/uncertainty_example.png",  # Saves to visual_output folder
    show=False  # Set to True to display interactively
)

print("✓ Visualization saved to: visual_output/uncertainty_example.png")
print("  - Blue shaded regions: confidence intervals (1σ and 2σ)")
print("  - Blue line: mean prediction")
print("  - Green line: true function")
print("  - Red dots: training data")
print("\nNotice: Uncertainty grows in the gap between x=3 and x=7 where there's no training data!")

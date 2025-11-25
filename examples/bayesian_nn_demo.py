"""
Bayesian neural network demonstration.

Demonstrates uncertainty quantification and active learning
with Bayesian neural networks on a regression task.
"""

import numpy as np
from tsu import BayesianRegressor
import matplotlib.pyplot as plt


def demonstrate_uncertainty_quantification():
    """Demonstrate Bayesian NN with uncertainty quantification on sparse data."""
    print("=" * 80)
    print("BAYESIAN NEURAL NETWORK DEMONSTRATION")
    print("=" * 80)

    # Generate training data (sparse sampling)
    np.random.seed(42)
    x_train = np.array([-3, -2, -1, 0, 1, 2, 3]).reshape(-1, 1)
    y_train = np.sin(x_train) + 0.1 * np.random.randn(7, 1)

    print("\n[1] Training on 7 data points")
    print("-" * 80)

    # Train Bayesian regressor
    model = BayesianRegressor(input_dim=1, hidden_dims=[20, 20], prior_std=1.0, temperature=1.0)

    history = model.fit(
        x_train, y_train, n_epochs=100, batch_size=7, learning_rate=0.01, verbose=False
    )

    print(f"Training complete. Final loss: {history['loss_history'][-1]:.4f}")

    # Make predictions across entire range
    print("\n[2] Predictions with Uncertainty")
    print("-" * 80)

    x_test = np.linspace(-5, 5, 50).reshape(-1, 1)
    result = model.predict_with_interval(x_test, n_samples=100, confidence=0.95)

    # Show uncertainty at different locations
    test_points = [-4.0, -2.0, 0.0, 2.0, 4.0]
    for x_val in test_points:
        idx = np.argmin(np.abs(x_test.flatten() - x_val))
        mean = result["mean"][idx, 0]
        std = result["std"][idx, 0]
        lower = result["lower"][idx, 0]
        upper = result["upper"][idx, 0]

        # Check if in training data region
        in_data = any(np.abs(x_train.flatten() - x_val) < 0.5)
        region = "TRAIN" if in_data else "EXTRAP"

        print(
            f"x={x_val:+.1f} ({region:6s}): "
            f"μ={mean:+.3f}, σ={std:.3f}, "
            f"95% CI=[{lower:+.3f}, {upper:+.3f}]"
        )

    # Active learning demo
    print("\n[3] Active Learning: Select Most Uncertain Points")
    print("-" * 80)

    # Pool of candidate points
    x_pool = np.linspace(-5, 5, 100).reshape(-1, 1)
    selected_idx = model.select_informative_samples(x_pool, n_select=5, n_samples=100)

    print("Most informative points to label next:")
    for i, idx in enumerate(selected_idx):
        x_val = x_pool[idx, 0]
        print(f"  {i+1}. x={x_val:+.3f}")

    print("\n" + "=" * 80)
    print("Demonstration complete")
    print("=" * 80)

    # Try to create visualization
    try:
        plt.figure(figsize=(12, 5))

        # Plot 1: Predictions with uncertainty
        plt.subplot(1, 2, 1)
        plt.plot(x_train, y_train, "ko", label="Training Data", markersize=8)
        plt.plot(x_test, result["mean"], "b-", label="Prediction", linewidth=2)
        plt.fill_between(
            x_test.flatten(),
            result["lower"].flatten(),
            result["upper"].flatten(),
            alpha=0.3,
            label="95% Confidence",
        )
        plt.plot(x_test, np.sin(x_test), "g--", label="True Function", alpha=0.5)
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.title("Bayesian NN: Predictions with Uncertainty")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 2: Uncertainty vs position
        plt.subplot(1, 2, 2)
        plt.plot(x_test, result["std"], "r-", linewidth=2)
        plt.axvspan(-3, 3, alpha=0.1, color="green", label="Training Region")
        plt.xlabel("Input")
        plt.ylabel("Predictive Uncertainty (σ)")
        plt.title("Uncertainty Increases Away from Data")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("bayesian_nn_demo.png", dpi=150, bbox_inches="tight")
        print(f"\n[OK] Saved visualization to bayesian_nn_demo.png")

    except Exception as e:
        print(f"\n[INFO] Could not create visualization: {e}")


if __name__ == "__main__":
    demonstrate_uncertainty_quantification()

"""
Test suite for visualization utilities.

Validates visualization functions create proper figures without errors.
Does not test visual appearance (difficult to automate), only that
functions execute correctly with various inputs.
"""

import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tsu.visualization import (
    plot_predictions_with_uncertainty,
    plot_uncertainty_vs_error,
    plot_energy_landscape_2d,
    plot_ising_state,
    plot_phase_transition,
    plot_sampling_diagnostics,
    plot_active_learning_curve,
)


class TestPredictionsWithUncertainty:
    """Test uncertainty visualization plots."""
    
    def test_basic_plot(self):
        """Test basic uncertainty plot creation."""
        x = np.linspace(0, 10, 50)
        y_pred = np.sin(x)
        y_std = 0.1 * np.ones_like(x)
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_true_values(self):
        """Test plot with true values overlay."""
        x = np.linspace(0, 10, 50)
        y_pred = np.sin(x)
        y_std = 0.1 * np.ones_like(x)
        y_true = np.sin(x) + 0.05 * np.random.randn(50)
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, y_true=y_true, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_training_data(self):
        """Test plot with training data overlay."""
        x = np.linspace(0, 10, 50)
        y_pred = np.sin(x)
        y_std = 0.1 * np.ones_like(x)
        x_train = np.array([1, 3, 5, 7, 9])
        y_train = np.sin(x_train)
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, 
            x_train=x_train, y_train=y_train,
            show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_multiple_confidence_levels(self):
        """Test plot with multiple confidence bands."""
        x = np.linspace(0, 10, 50)
        y_pred = np.sin(x)
        y_std = 0.1 * np.ones_like(x)
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, 
            confidence_levels=[1.0, 2.0, 3.0],
            show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_varying_uncertainty(self):
        """Test plot with spatially varying uncertainty."""
        x = np.linspace(-5, 5, 100)
        y_pred = np.sin(x)
        y_std = 0.1 + 0.2 * np.abs(x) / 5  # Higher uncertainty far from center
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestUncertaintyVsError:
    """Test calibration analysis plots."""
    
    def test_basic_calibration_plot(self):
        """Test basic calibration analysis."""
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        y_std = np.abs(y_true - y_pred) + 0.05
        
        fig = plot_uncertainty_vs_error(
            y_true, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_perfect_calibration(self):
        """Test plot with perfectly calibrated uncertainty."""
        np.random.seed(42)
        y_true = np.random.randn(100)
        y_std = 0.2 * np.ones(100)
        y_pred = y_true + y_std * np.random.randn(100)
        
        fig = plot_uncertainty_vs_error(
            y_true, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_different_bin_counts(self):
        """Test with different binning resolutions."""
        y_true = np.random.randn(100)
        y_pred = y_true + 0.1 * np.random.randn(100)
        y_std = 0.2 * np.ones(100)
        
        for bins in [10, 20, 30]:
            fig = plot_uncertainty_vs_error(
                y_true, y_pred, y_std, bins=bins, show=False
            )
            assert isinstance(fig, Figure)
            plt.close(fig)


class TestEnergyLandscape:
    """Test energy landscape visualization."""
    
    def test_simple_quadratic(self):
        """Test with simple quadratic energy."""
        def energy(x):
            return x[0]**2 + x[1]**2
        
        fig = plot_energy_landscape_2d(
            energy, xlim=(-2, 2), ylim=(-2, 2),
            resolution=30, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_samples(self):
        """Test energy landscape with sample overlay."""
        def energy(x):
            return (x[0]**2 - 1)**2 + (x[1]**2 - 1)**2
        
        samples = np.random.randn(50, 2)
        
        fig = plot_energy_landscape_2d(
            energy, xlim=(-2, 2), ylim=(-2, 2),
            samples=samples, resolution=30, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_trajectory(self):
        """Test energy landscape with optimization trajectory."""
        def energy(x):
            return x[0]**2 + x[1]**2
        
        # Fake trajectory moving toward minimum
        t = np.linspace(0, 1, 20)
        trajectory = np.column_stack([
            2 * (1 - t),
            -2 * (1 - t)
        ])
        
        fig = plot_energy_landscape_2d(
            energy, xlim=(-3, 3), ylim=(-3, 3),
            trajectory=trajectory, resolution=30, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_samples_and_trajectory(self):
        """Test with both samples and trajectory."""
        def energy(x):
            return np.sum(x**2)
        
        samples = np.random.randn(30, 2)
        trajectory = np.linspace([-2, -2], [0, 0], 15)
        
        fig = plot_energy_landscape_2d(
            energy, xlim=(-3, 3), ylim=(-3, 3),
            samples=samples, trajectory=trajectory,
            resolution=25, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestIsingVisualization:
    """Test Ising model state visualization."""
    
    def test_1d_chain(self):
        """Test 1D Ising chain visualization."""
        state = np.random.choice([0, 1], size=20)
        
        fig = plot_ising_state(state, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_2d_grid(self):
        """Test 2D Ising grid visualization."""
        state = np.random.choice([0, 1], size=(16, 16))
        
        fig = plot_ising_state(state, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_spin_values(self):
        """Test with spin values {-1, +1} instead of {0, 1}."""
        state = np.random.choice([-1, 1], size=(10, 10))
        
        fig = plot_ising_state(state, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_without_colorbar(self):
        """Test 2D visualization without colorbar."""
        state = np.random.choice([0, 1], size=(12, 12))
        
        fig = plot_ising_state(state, colorbar=False, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_custom_figsize(self):
        """Test with custom figure size."""
        state = np.random.choice([0, 1], size=15)
        
        fig = plot_ising_state(state, figsize=(10, 3), show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestPhaseTransition:
    """Test phase transition visualization."""
    
    def test_basic_phase_transition(self):
        """Test basic phase transition plot."""
        temperatures = np.linspace(0.5, 4.0, 20)
        magnetizations = 1.0 / (1.0 + temperatures)  # Fake phase transition
        
        fig = plot_phase_transition(
            temperatures, magnetizations, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_error_bars(self):
        """Test phase transition with error bars."""
        temperatures = np.linspace(0.5, 4.0, 20)
        magnetizations = 1.0 / (1.0 + temperatures)
        errors = 0.05 * np.ones_like(temperatures)
        
        fig = plot_phase_transition(
            temperatures, magnetizations,
            magnetization_errors=errors,
            show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_critical_temperature(self):
        """Test with critical temperature marker."""
        temperatures = np.linspace(0.5, 4.0, 20)
        magnetizations = 1.0 / (1.0 + temperatures)
        
        fig = plot_phase_transition(
            temperatures, magnetizations,
            critical_temp=2.269,
            show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestSamplingDiagnostics:
    """Test sampling diagnostic plots."""
    
    def test_basic_diagnostics(self):
        """Test basic sampling diagnostics."""
        samples = np.random.randn(1000)
        
        fig = plot_sampling_diagnostics(samples, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_multidimensional_samples(self):
        """Test diagnostics with multidimensional samples."""
        samples = np.random.randn(1000, 3)
        
        fig = plot_sampling_diagnostics(samples, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_with_true_distribution(self):
        """Test with true distribution overlay."""
        samples = np.random.randn(1000)
        
        def true_dist(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
        fig = plot_sampling_diagnostics(
            samples, true_distribution=true_dist, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_correlated_samples(self):
        """Test diagnostics with autocorrelated samples."""
        # Create autocorrelated samples
        samples = np.zeros(1000)
        samples[0] = np.random.randn()
        for i in range(1, 1000):
            samples[i] = 0.9 * samples[i-1] + 0.1 * np.random.randn()
        
        fig = plot_sampling_diagnostics(samples, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestActiveLearningCurve:
    """Test active learning performance plots."""
    
    def test_basic_learning_curve(self):
        """Test basic active learning curve."""
        n_labeled = np.array([10, 20, 30, 40, 50])
        acc_active = np.array([0.5, 0.65, 0.75, 0.82, 0.87])
        acc_random = np.array([0.45, 0.58, 0.68, 0.76, 0.82])
        
        fig = plot_active_learning_curve(
            n_labeled, acc_active, acc_random, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_equal_performance(self):
        """Test when active and random perform equally."""
        n_labeled = np.array([10, 20, 30, 40, 50])
        acc = np.array([0.5, 0.6, 0.7, 0.8, 0.85])
        
        fig = plot_active_learning_curve(
            n_labeled, acc, acc, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_active_worse_than_random(self):
        """Test when active learning underperforms."""
        n_labeled = np.array([10, 20, 30, 40, 50])
        acc_active = np.array([0.4, 0.5, 0.6, 0.7, 0.75])
        acc_random = np.array([0.5, 0.6, 0.7, 0.8, 0.85])
        
        fig = plot_active_learning_curve(
            n_labeled, acc_active, acc_random, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_point_prediction(self):
        """Test uncertainty plot with single point."""
        x = np.array([0.5])
        y_pred = np.array([1.0])
        y_std = np.array([0.1])
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_zero_uncertainty(self):
        """Test with zero uncertainty."""
        x = np.linspace(0, 1, 20)
        y_pred = np.sin(x)
        y_std = np.zeros_like(x)
        
        fig = plot_predictions_with_uncertainty(
            x, y_pred, y_std, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_small_ising_chain(self):
        """Test with very small Ising chain."""
        state = np.array([0, 1, 0])
        
        fig = plot_ising_state(state, show=False)
        
        assert isinstance(fig, Figure)
        plt.close(fig)
    
    def test_single_temperature_point(self):
        """Test phase transition with single point."""
        temperatures = np.array([2.0])
        magnetizations = np.array([0.5])
        
        fig = plot_phase_transition(
            temperatures, magnetizations, show=False
        )
        
        assert isinstance(fig, Figure)
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


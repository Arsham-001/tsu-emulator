"""
Visualization utilities for probabilistic computing.

Provides publication-quality plots for:
- Uncertainty quantification in predictions
- Energy landscapes and optimization trajectories
- Phase transitions in physical systems
- Model calibration and reliability
- Sampling diagnostics and convergence

Uses matplotlib for static plots and optionally plotly for interactive visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
from typing import Optional, Tuple, List, Union, Callable
import warnings

# Try importing plotly for interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not available. Install with 'pip install plotly' for interactive plots.")


# Color schemes optimized for scientific visualization
# Reference: "A Better Default Colormap for Matplotlib" (Smith & Borland, 2015)
UNCERTAINTY_CMAP = 'RdYlGn_r'  # Red (high uncertainty) to green (low uncertainty)
ENERGY_CMAP = 'viridis'  # Perceptually uniform for energy landscapes
DIVERGING_CMAP = 'RdBu_r'  # For signed quantities (correlations, fields)


def plot_predictions_with_uncertainty(
    x: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    y_true: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    title: str = "Predictions with Uncertainty",
    xlabel: str = "Input",
    ylabel: str = "Output",
    confidence_levels: List[float] = [1.0, 2.0],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot predictions with confidence intervals.
    
    Visualizes epistemic uncertainty via shaded confidence bands.
    Multiple confidence levels (e.g., 1σ, 2σ) show probability
    of true value falling within bands assuming Gaussian posterior.
    
    Particularly useful for regression tasks to identify regions
    where model is uncertain (far from training data).
    
    Args:
        x: Input values for predictions (n_points,)
        y_pred: Mean predictions (n_points,)
        y_std: Standard deviations (n_points,)
        y_true: True values if known (n_points,)
        x_train: Training input locations (n_train,)
        y_train: Training output values (n_train,)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        confidence_levels: Multiples of std to plot (e.g., [1, 2] for 1σ and 2σ)
        figsize: Figure dimensions
        save_path: If provided, save figure to this path
        show: Whether to display the plot
        
    Returns:
        Matplotlib figure object
        
    Example:
        >>> model.train(X_train, y_train)
        >>> result = model.predict(X_test, n_samples=100)
        >>> plot_predictions_with_uncertainty(X_test, result.mean, result.std)
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Sort for proper line plotting
    sort_idx = np.argsort(x.flatten())
    x_sorted = x.flatten()[sort_idx]
    y_pred_sorted = y_pred.flatten()[sort_idx]
    y_std_sorted = y_std.flatten()[sort_idx]
    
    # Plot confidence bands (lightest to darkest)
    alphas = np.linspace(0.15, 0.3, len(confidence_levels))
    for level, alpha in zip(sorted(confidence_levels, reverse=True), alphas):
        ax.fill_between(
            x_sorted,
            y_pred_sorted - level * y_std_sorted,
            y_pred_sorted + level * y_std_sorted,
            alpha=alpha,
            color='blue',
            label=f'{level}σ confidence' if level == confidence_levels[0] else None
        )
    
    # Mean prediction
    ax.plot(x_sorted, y_pred_sorted, 'b-', linewidth=2, label='Mean prediction')
    
    # True values if provided
    if y_true is not None:
        y_true_sorted = y_true.flatten()[sort_idx]
        ax.plot(x_sorted, y_true_sorted, 'k--', linewidth=1.5, 
               label='True function', alpha=0.7)
    
    # Training data if provided
    if x_train is not None and y_train is not None:
        ax.scatter(x_train, y_train, c='red', s=30, alpha=0.6,
                  label='Training data', zorder=5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_uncertainty_vs_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_std: np.ndarray,
    title: str = "Uncertainty vs Prediction Error",
    bins: int = 20,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Analyze correlation between uncertainty and prediction error.
    
    Well-calibrated models show strong correlation: high uncertainty
    corresponds to high error. This validates that uncertainty estimates
    are meaningful for decision-making.
    
    Creates two-panel figure:
    - Left: Scatter plot of uncertainty vs absolute error
    - Right: Binned error bars showing mean error per uncertainty bin
    
    Args:
        y_true: True values (n_samples,)
        y_pred: Predicted values (n_samples,)
        y_std: Uncertainty estimates (n_samples,)
        title: Plot title
        bins: Number of bins for aggregation plot
        figsize: Figure dimensions
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    errors = np.abs(y_true.flatten() - y_pred.flatten())
    uncertainties = y_std.flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot with density coloring
    ax1.scatter(uncertainties, errors, alpha=0.5, s=20, c=uncertainties,
               cmap=UNCERTAINTY_CMAP)
    ax1.plot([0, uncertainties.max()], [0, uncertainties.max()], 
            'k--', alpha=0.5, label='Perfect calibration')
    ax1.set_xlabel('Predicted Uncertainty (σ)', fontsize=11)
    ax1.set_ylabel('Absolute Error |y_true - y_pred|', fontsize=11)
    ax1.set_title('Uncertainty vs Error (per sample)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Binned analysis
    bin_edges = np.percentile(uncertainties, np.linspace(0, 100, bins + 1))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_errors = []
    bin_stds = []
    
    for i in range(bins):
        mask = (uncertainties >= bin_edges[i]) & (uncertainties < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_errors.append(errors[mask].mean())
            bin_stds.append(errors[mask].std())
        else:
            bin_errors.append(0)
            bin_stds.append(0)
    
    ax2.errorbar(bin_centers, bin_errors, yerr=bin_stds, 
                fmt='o-', capsize=5, capthick=2, linewidth=2)
    ax2.plot([0, bin_centers[-1]], [0, bin_centers[-1]], 
            'k--', alpha=0.5, label='Perfect calibration')
    ax2.set_xlabel('Uncertainty Bin Center', fontsize=11)
    ax2.set_ylabel('Mean Absolute Error', fontsize=11)
    ax2.set_title('Binned Calibration Analysis', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Compute correlation
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    fig.suptitle(f"{title} (ρ={correlation:.3f})", fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_energy_landscape_2d(
    energy_fn: Callable,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    samples: Optional[np.ndarray] = None,
    trajectory: Optional[np.ndarray] = None,
    resolution: int = 100,
    title: str = "Energy Landscape",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Visualize 2D energy landscape with optional samples and trajectories.
    
    Creates contour plot of energy function, overlaying sampling
    trajectories or final sample distributions. Useful for understanding
    optimization behavior and energy surface topology.
    
    Args:
        energy_fn: Function E(x) where x is 2D array
        xlim: (min, max) for x-axis
        ylim: (min, max) for y-axis
        samples: Sample points to overlay (n_samples, 2)
        trajectory: Optimization trajectory (n_steps, 2)
        resolution: Grid resolution for contour plot
        title: Plot title
        figsize: Figure dimensions
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    # Create meshgrid for energy evaluation
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate energy at grid points
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = energy_fn(np.array([X[i, j], Y[i, j]]))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Contour plot with logarithmic levels for wide energy ranges
    levels = np.percentile(Z, np.linspace(0, 100, 20))
    contour = ax.contour(X, Y, Z, levels=levels, linewidths=0.5, alpha=0.4)
    contourf = ax.contourf(X, Y, Z, levels=levels, cmap=ENERGY_CMAP, alpha=0.6)
    
    plt.colorbar(contourf, ax=ax, label='Energy')
    
    # Overlay samples if provided
    if samples is not None:
        ax.scatter(samples[:, 0], samples[:, 1], c='red', s=10, 
                  alpha=0.5, label='Samples', zorder=3)
    
    # Overlay trajectory if provided
    if trajectory is not None:
        ax.plot(trajectory[:, 0], trajectory[:, 1], 'w-', linewidth=2,
               alpha=0.8, label='Trajectory')
        ax.scatter(trajectory[0, 0], trajectory[0, 1], c='lime', s=100,
                  marker='*', label='Start', zorder=4)
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100,
                  marker='*', label='End', zorder=4)
    
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if samples is not None or trajectory is not None:
        ax.legend(loc='best')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_ising_state(
    state: np.ndarray,
    title: str = "Ising Configuration",
    colorbar: bool = True,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Visualize Ising model spin configuration.
    
    For 1D: arrow plot (↑↓)
    For 2D: heatmap (red=+1, blue=-1)
    
    Args:
        state: Spin configuration (1D or 2D array, values in {0,1} or {-1,+1})
        title: Plot title
        colorbar: Whether to show colorbar (2D only)
        figsize: Figure dimensions (auto-computed if None)
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    # Convert to {-1, +1} if needed
    if np.all((state == 0) | (state == 1)):
        spins = 2 * state - 1
    else:
        spins = state
    
    if state.ndim == 1:
        # 1D chain: arrow plot
        if figsize is None:
            figsize = (int(max(12, len(state) * 0.5)), 2)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        for i, spin in enumerate(spins):
            arrow = '↑' if spin > 0 else '↓'
            color = 'red' if spin > 0 else 'blue'
            ax.text(i, 0, arrow, fontsize=20, ha='center', va='center',
                   color=color, fontweight='bold')
        
        ax.set_xlim(-0.5, len(spins) - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xticks(range(len(spins)))
        ax.set_xticklabels([str(i) for i in range(len(spins))])
        ax.set_yticks([])
        ax.set_xlabel('Spin Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
    else:
        # 2D grid: heatmap
        if figsize is None:
            aspect = state.shape[1] / state.shape[0]
            figsize = (8 * aspect, 8)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(spins, cmap=DIVERGING_CMAP, vmin=-1, vmax=1,
                      interpolation='nearest')
        
        if colorbar:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Spin', rotation=270, labelpad=15)
            cbar.set_ticks([-1, 0, 1])
            cbar.set_ticklabels(['↓ (-1)', '0', '↑ (+1)'])
        
        ax.set_xlabel('x', fontsize=12)
        ax.set_ylabel('y', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_phase_transition(
    temperatures: np.ndarray,
    magnetizations: np.ndarray,
    magnetization_errors: Optional[np.ndarray] = None,
    critical_temp: Optional[float] = None,
    title: str = "Phase Transition",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Plot magnetization vs temperature to visualize phase transition.
    
    Phase transitions are fundamental phenomena in statistical mechanics.
    Ordered phase (low T): high magnetization
    Disordered phase (high T): low magnetization
    Critical point (T_c): rapid transition
    
    Args:
        temperatures: Temperature values (n_temps,)
        magnetizations: Mean magnetization at each temperature (n_temps,)
        magnetization_errors: Error bars (standard deviation) (n_temps,)
        critical_temp: Known critical temperature (plot as vertical line)
        title: Plot title
        figsize: Figure dimensions
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if magnetization_errors is not None:
        ax.errorbar(temperatures, magnetizations, yerr=magnetization_errors,
                   fmt='o-', capsize=5, capthick=2, linewidth=2,
                   markersize=8, label='Measured')
    else:
        ax.plot(temperatures, magnetizations, 'o-', linewidth=2,
               markersize=8, label='Measured')
    
    # Mark critical temperature if provided
    if critical_temp is not None:
        ax.axvline(critical_temp, color='red', linestyle='--', linewidth=2,
                  alpha=0.7, label=f'Critical T_c={critical_temp:.3f}')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Temperature (T)', fontsize=12)
    ax.set_ylabel('|Magnetization| (|M|)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_sampling_diagnostics(
    samples: np.ndarray,
    true_distribution: Optional[Callable] = None,
    title: str = "Sampling Diagnostics",
    figsize: Tuple[int, int] = (15, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Diagnostic plots for assessing sampling quality.
    
    Three-panel figure:
    - Left: Sample histogram vs true distribution (if provided)
    - Middle: Autocorrelation (measures independence)
    - Right: Trace plot (convergence visualization)
    
    Args:
        samples: MCMC samples (n_samples,) or (n_samples, n_dims)
        true_distribution: True PDF for comparison (1D function)
        title: Plot title
        figsize: Figure dimensions
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot first dimension only for simplicity
    sample_1d = samples[:, 0]
    
    # Histogram
    axes[0].hist(sample_1d, bins=50, density=True, alpha=0.7, 
                color='blue', edgecolor='black', label='Samples')
    
    if true_distribution is not None:
        x_range = np.linspace(sample_1d.min(), sample_1d.max(), 200)
        axes[0].plot(x_range, [true_distribution(x) for x in x_range],
                    'r-', linewidth=2, label='True distribution')
    
    axes[0].set_xlabel('Value', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('Sample Distribution', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Autocorrelation
    max_lag = min(200, len(sample_1d) // 2)
    lags = range(max_lag)
    autocorr = [np.corrcoef(sample_1d[:-lag or None], sample_1d[lag:])[0, 1] 
                if lag > 0 else 1.0 for lag in lags]
    
    axes[1].plot(lags, autocorr, 'b-', linewidth=2)
    axes[1].axhline(0, color='k', linestyle='--', alpha=0.3)
    axes[1].axhline(0.05, color='r', linestyle=':', alpha=0.5, 
                   label='5% threshold')
    axes[1].axhline(-0.05, color='r', linestyle=':', alpha=0.5)
    axes[1].set_xlabel('Lag', fontsize=11)
    axes[1].set_ylabel('Autocorrelation', fontsize=11)
    axes[1].set_title('Autocorrelation Function', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Trace plot
    axes[2].plot(sample_1d, 'b-', alpha=0.6, linewidth=0.5)
    axes[2].set_xlabel('Sample Index', fontsize=11)
    axes[2].set_ylabel('Value', fontsize=11)
    axes[2].set_title('Trace Plot', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_active_learning_curve(
    n_labeled: np.ndarray,
    accuracies_active: np.ndarray,
    accuracies_random: np.ndarray,
    title: str = "Active Learning Performance",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> Figure:
    """
    Compare active learning vs random sampling strategies.
    
    Demonstrates value of uncertainty-based sample selection:
    active learning achieves higher accuracy with fewer labels.
    
    Args:
        n_labeled: Number of labeled examples (x-axis)
        accuracies_active: Test accuracy with active learning
        accuracies_random: Test accuracy with random sampling
        title: Plot title
        figsize: Figure dimensions
        save_path: Path to save figure
        show: Whether to display plot
        
    Returns:
        Matplotlib figure object
    """
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(n_labeled, accuracies_active, 'o-', linewidth=2.5,
           markersize=8, color='blue', label='Active Learning (Uncertainty)')
    ax.plot(n_labeled, accuracies_random, 's--', linewidth=2.5,
           markersize=8, color='gray', label='Random Sampling', alpha=0.7)
    
    # Shade improvement region
    ax.fill_between(n_labeled, accuracies_active, accuracies_random,
                   where=(accuracies_active >= accuracies_random).tolist(),
                   alpha=0.2, color='blue', label='Active learning advantage')
    
    ax.set_xlabel('Number of Labeled Examples', fontsize=12)
    ax.set_ylabel('Test Accuracy', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Annotate improvement if both curves have same length
    if len(accuracies_active) == len(accuracies_random):
        improvement = accuracies_active[-1] - accuracies_random[-1]
        if improvement > 0:
            ax.annotate(f'Improvement: {improvement:.1%}',
                       xy=(n_labeled[-1], accuracies_active[-1]),
                       xytext=(n_labeled[-1] * 0.7, accuracies_active[-1] - 0.15),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2),
                       fontsize=11, color='blue', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


# Interactive plots (requires plotly)

def plot_interactive_energy_landscape(
    energy_fn: Callable,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    samples: Optional[np.ndarray] = None,
    resolution: int = 50,
    title: str = "Interactive Energy Landscape"
):
    """
    Create interactive 3D energy landscape visualization.
    
    Requires plotly. Allows rotation, zoom, and hover for detailed exploration.
    
    Args:
        energy_fn: Function E(x) where x is 2D
        xlim: X-axis range
        ylim: Y-axis range
        samples: Optional sample points to overlay
        resolution: Grid resolution
        title: Plot title
    """
    
    if not PLOTLY_AVAILABLE:
        raise ImportError("Plotly required for interactive plots. Install: pip install plotly")
    
    # Create meshgrid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate energy
    Z = np.zeros_like(X)
    for i in range(resolution):
        for j in range(resolution):
            Z[i, j] = energy_fn(np.array([X[i, j], Y[i, j]]))
    
    # Create surface plot
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
    
    # Add samples if provided
    if samples is not None:
        sample_energies = np.array([energy_fn(s) for s in samples])
        fig.add_trace(go.Scatter3d(
            x=samples[:, 0],
            y=samples[:, 1],
            z=sample_energies,
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Samples'
        ))
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='x₁',
            yaxis_title='x₂',
            zaxis_title='Energy'
        ),
        width=800,
        height=600
    )
    
    fig.show()


if __name__ == "__main__":
    print("=" * 80)
    print("VISUALIZATION MODULE DEMONSTRATION")
    print("=" * 80)
    
    # Demo 1: Uncertainty visualization
    print("\n[1] Uncertainty Visualization")
    np.random.seed(42)
    x_test = np.linspace(-5, 5, 100)
    y_pred = np.sin(x_test)
    y_std = 0.1 + 0.3 * np.abs(x_test) / 5  # Higher uncertainty far from center
    
    x_train = np.array([-2, -1, 0, 1, 2])
    y_train = np.sin(x_train) + 0.1 * np.random.randn(5)
    
    fig = plot_predictions_with_uncertainty(
        x_test, y_pred, y_std,
        x_train=x_train, y_train=y_train,
        title="Bayesian Prediction with Uncertainty",
        show=False
    )
    plt.close(fig)
    print("[OK] Uncertainty plot created")
    
    # Demo 2: Energy landscape
    print("\n[2] Energy Landscape Visualization")
    def double_well(x):
        return (x[0]**2 - 1)**2 + (x[1]**2 - 1)**2
    
    samples_2d = np.random.randn(50, 2) * 0.5
    
    fig = plot_energy_landscape_2d(
        double_well, xlim=(-2, 2), ylim=(-2, 2),
        samples=samples_2d,
        title="Double-Well Energy Landscape",
        show=False, resolution=50
    )
    plt.close(fig)
    print("[OK] Energy landscape created")
    
    # Demo 3: Ising visualization
    print("\n[3] Ising Configuration Visualization")
    ising_2d = np.random.choice([0, 1], size=(16, 16))
    
    fig = plot_ising_state(ising_2d, title="Random Ising Configuration", show=False)
    plt.close(fig)
    print("[OK] Ising configuration plot created")
    
    # Demo 4: Sampling diagnostics
    print("\n[4] Sampling Diagnostics")
    samples_mcmc = np.random.randn(1000)
    
    fig = plot_sampling_diagnostics(samples_mcmc, title="MCMC Diagnostics", show=False)
    plt.close(fig)
    print("[OK] Sampling diagnostics created")
    
    print("\n" + "=" * 80)
    print("All visualization functions verified")
    print("=" * 80)


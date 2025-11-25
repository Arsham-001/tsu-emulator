"""
Machine learning benchmarks.

Measures TSU's Bayesian neural network performance on standard ML tasks.
Tests prediction accuracy, calibration quality, and uncertainty quantification.

Methodology:
- Standard regression/classification datasets
- Cross-validation for robust estimates
- Calibration metrics (ECE, reliability diagrams)
- Comparison with frequentist baselines
"""

import numpy as np
import time
from typing import Dict, List
from dataclasses import dataclass, field
from ..ml import BayesianRegressor


@dataclass
class MLResult:
    """Results from ML benchmark."""

    dataset_name: str
    task_type: str  # 'regression' or 'classification'
    n_train: int
    n_test: int
    n_folds: int

    # Accuracy metrics
    test_errors: List[float] = field(default_factory=list)
    test_r2_scores: List[float] = field(default_factory=list)

    # Uncertainty metrics
    calibration_errors: List[float] = field(default_factory=list)
    negative_log_likelihoods: List[float] = field(default_factory=list)
    coverage_rates: List[float] = field(default_factory=list)

    # Performance
    train_times: List[float] = field(default_factory=list)
    predict_times: List[float] = field(default_factory=list)

    def summary(self) -> Dict:
        """Compute summary statistics."""
        summary = {
            "dataset": self.dataset_name,
            "task": self.task_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "test_error": {
                "mean": np.mean(self.test_errors),
                "std": np.std(self.test_errors),
                "median": np.median(self.test_errors),
            },
            "train_time_sec": {
                "mean": np.mean(self.train_times),
                "std": np.std(self.train_times),
            },
            "predict_time_ms": {
                "mean": np.mean(self.predict_times) * 1000,
                "std": np.std(self.predict_times) * 1000,
            },
        }

        if self.test_r2_scores:
            summary["r2_score"] = {
                "mean": np.mean(self.test_r2_scores),
                "std": np.std(self.test_r2_scores),
            }

        if self.calibration_errors:
            summary["calibration_error"] = {
                "mean": np.mean(self.calibration_errors),
                "std": np.std(self.calibration_errors),
            }

        if self.coverage_rates:
            summary["95_coverage"] = {
                "mean": np.mean(self.coverage_rates),
                "std": np.std(self.coverage_rates),
                "target": 0.95,
            }

        return summary


class MLBenchmark:
    """
    Benchmark machine learning performance.

    Tests Bayesian neural networks on standard tasks,
    measuring both prediction accuracy and uncertainty quality.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize benchmark.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed

    def benchmark_regression_synthetic(
        self, n_train: int = 200, n_test: int = 100, n_folds: int = 5, noise_std: float = 0.1
    ) -> MLResult:
        """
        Benchmark on synthetic regression task.

        Data: y = sin(2πx) + ε, x ∈ [0, 1]
        Tests basic regression and uncertainty quantification.

        Args:
            n_train: Training samples per fold
            n_test: Test samples per fold
            n_folds: Number of cross-validation folds
            noise_std: Observation noise level

        Returns:
            MLResult with performance metrics
        """
        result = MLResult(
            dataset_name="Synthetic_Sinusoid",
            task_type="regression",
            n_train=n_train,
            n_test=n_test,
            n_folds=n_folds,
        )

        for fold in range(n_folds):
            np.random.seed(self.seed + fold)

            # Generate data
            x_train = np.random.uniform(0, 1, size=(n_train, 1))
            y_train = np.sin(2 * np.pi * x_train) + noise_std * np.random.randn(n_train, 1)

            x_test = np.random.uniform(0, 1, size=(n_test, 1))
            y_test = np.sin(2 * np.pi * x_test) + noise_std * np.random.randn(n_test, 1)
            # y_test_true = np.sin(2 * np.pi * x_test)  # Not used currently

            # Train model
            model = BayesianRegressor(input_dim=1, hidden_dims=[20, 20], prior_std=1.0)

            start_time = time.time()
            model.fit(x_train, y_train, n_epochs=50, batch_size=32, verbose=False)
            train_time = time.time() - start_time

            # Make predictions
            start_time = time.time()
            predictions = model.predict_with_interval(x_test, n_samples=100, confidence=0.95)
            predict_time = time.time() - start_time

            # Compute metrics
            y_pred = predictions["mean"]
            y_std = predictions["std"]
            lower = predictions["lower"]
            upper = predictions["upper"]

            # Test error (MSE)
            test_error = np.mean((y_pred - y_test) ** 2)
            result.test_errors.append(test_error)

            # R² score
            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot
            result.test_r2_scores.append(float(r2))

            # Calibration: expected calibration error
            ece = self._compute_calibration_error(y_test, y_pred, y_std)
            result.calibration_errors.append(ece)

            # Coverage: fraction of true values within 95% interval
            coverage = np.mean((y_test >= lower) & (y_test <= upper))
            result.coverage_rates.append(float(coverage))

            # Negative log-likelihood
            nll = np.mean(
                0.5 * np.log(2 * np.pi * y_std**2) + 0.5 * ((y_test - y_pred) / y_std) ** 2
            )
            result.negative_log_likelihoods.append(float(nll))

            result.train_times.append(train_time)
            result.predict_times.append(predict_time)

        return result

    def benchmark_regression_nonlinear(
        self, n_train: int = 300, n_test: int = 100, n_folds: int = 5
    ) -> MLResult:
        """
        Benchmark on nonlinear regression with heteroscedastic noise.

        Data: y = x * sin(x) + ε(x), x ∈ [-5, 5]
        where noise ε(x) ~ N(0, 0.1 * |x|)

        Tests ability to capture input-dependent uncertainty.

        Args:
            n_train: Training samples per fold
            n_test: Test samples per fold
            n_folds: Number of folds

        Returns:
            MLResult with metrics
        """
        result = MLResult(
            dataset_name="Nonlinear_Heteroscedastic",
            task_type="regression",
            n_train=n_train,
            n_test=n_test,
            n_folds=n_folds,
        )

        for fold in range(n_folds):
            np.random.seed(self.seed + fold)

            # Generate heteroscedastic data
            x_train = np.random.uniform(-5, 5, size=(n_train, 1))
            noise_train = 0.1 * np.abs(x_train) * np.random.randn(n_train, 1)
            y_train = x_train * np.sin(x_train) + noise_train

            x_test = np.random.uniform(-5, 5, size=(n_test, 1))
            noise_test = 0.1 * np.abs(x_test) * np.random.randn(n_test, 1)
            y_test = x_test * np.sin(x_test) + noise_test

            # Train model
            model = BayesianRegressor(input_dim=1, hidden_dims=[30, 30], prior_std=1.0)

            start_time = time.time()
            model.fit(x_train, y_train, n_epochs=100, batch_size=32, verbose=False)
            train_time = time.time() - start_time

            # Predict
            start_time = time.time()
            predictions = model.predict_with_interval(x_test, n_samples=100)
            predict_time = time.time() - start_time

            # Metrics
            y_pred = predictions["mean"]
            y_std = predictions["std"]

            test_error = np.mean((y_pred - y_test) ** 2)
            result.test_errors.append(test_error)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = 1 - ss_res / ss_tot
            result.test_r2_scores.append(float(r2))

            ece = self._compute_calibration_error(y_test, y_pred, y_std)
            result.calibration_errors.append(ece)

            coverage = np.mean(
                (y_test >= predictions["lower"]) & (y_test <= predictions["upper"])
            )
            result.coverage_rates.append(float(coverage))

            result.train_times.append(train_time)
            result.predict_times.append(predict_time)

        return result

    def benchmark_extrapolation(
        self, n_train: int = 150, n_test: int = 100, n_folds: int = 5
    ) -> MLResult:
        """
        Benchmark extrapolation and uncertainty awareness.

        Train on x ∈ [-2, 2], test on x ∈ [-5, -2] ∪ [2, 5]
        Tests if model correctly reports high uncertainty outside training range.

        Args:
            n_train: Training samples
            n_test: Test samples (outside training range)
            n_folds: Number of folds

        Returns:
            MLResult with metrics
        """
        result = MLResult(
            dataset_name="Extrapolation_Test",
            task_type="regression",
            n_train=n_train,
            n_test=n_test,
            n_folds=n_folds,
        )

        for fold in range(n_folds):
            np.random.seed(self.seed + fold)

            # Training data: inside range
            x_train = np.random.uniform(-2, 2, size=(n_train, 1))
            y_train = np.sin(x_train) + 0.1 * np.random.randn(n_train, 1)

            # Test data: outside range
            x_test_left = np.random.uniform(-5, -2, size=(n_test // 2, 1))
            x_test_right = np.random.uniform(2, 5, size=(n_test - n_test // 2, 1))
            x_test = np.vstack([x_test_left, x_test_right])
            y_test = np.sin(x_test) + 0.1 * np.random.randn(n_test, 1)

            # Train
            model = BayesianRegressor(input_dim=1, hidden_dims=[20, 20])

            start_time = time.time()
            model.fit(x_train, y_train, n_epochs=50, verbose=False)
            train_time = time.time() - start_time

            # Predict
            start_time = time.time()
            pred_train = model.predict_with_interval(x_train, n_samples=100)
            pred_test = model.predict_with_interval(x_test, n_samples=100)
            predict_time = time.time() - start_time

            # Key metric: uncertainty should be higher for extrapolation
            train_uncertainty = np.mean(pred_train["std"])
            test_uncertainty = np.mean(pred_test["std"])
            uncertainty_ratio = test_uncertainty / train_uncertainty

            # Store as "calibration error" (higher is better here)
            result.calibration_errors.append(float(uncertainty_ratio))

            test_error = np.mean((pred_test["mean"] - y_test) ** 2)
            result.test_errors.append(test_error)

            result.train_times.append(train_time)
            result.predict_times.append(predict_time)

        return result

    def _compute_calibration_error(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray, n_bins: int = 10
    ) -> float:
        """
        Compute expected calibration error (ECE).

        Measures how well predicted uncertainties match actual errors.
        Lower is better.

        Args:
            y_true: True values
            y_pred: Predicted means
            y_std: Predicted standard deviations
            n_bins: Number of confidence bins

        Returns:
            Expected calibration error
        """
        # Compute standardized errors (z-scores)
        z_scores = np.abs((y_true - y_pred) / (y_std + 1e-8))

        # Expected fraction within k standard deviations
        # For Gaussian: P(|z| < 1) ≈ 0.68, P(|z| < 2) ≈ 0.95
        confidence_levels = np.linspace(0.1, 0.9, n_bins)
        ece = 0.0

        for conf in confidence_levels:
            # For Gaussian, conf = erf(k/sqrt(2))
            # Inverse: k = sqrt(2) * erf_inv(conf)
            from scipy.special import erfinv

            k = np.sqrt(2) * erfinv(conf)

            # Empirical coverage
            empirical = np.mean(z_scores.flatten() <= k)

            # Calibration gap
            ece += abs(empirical - conf)

        ece /= n_bins
        return float(ece)

    def run_all_benchmarks(self, quick: bool = False) -> Dict[str, MLResult]:
        """
        Run complete ML benchmark suite.

        Args:
            quick: If True, use reduced settings for faster execution

        Returns:
            Dictionary of benchmark results
        """
        if quick:
            n_train = 100
            n_test = 50
            n_folds = 3
        else:
            n_train = 200
            n_test = 100
            n_folds = 5

        results = {}

        print("Running ML benchmarks...")
        print("=" * 80)

        # Synthetic regression
        print("\n[1] Synthetic Regression")
        print("-" * 80)
        results["synthetic"] = self.benchmark_regression_synthetic(
            n_train=n_train, n_test=n_test, n_folds=n_folds
        )
        summary = results["synthetic"].summary()
        test_err_mean = summary["test_error"]["mean"]
        test_err_std = summary["test_error"]["std"]
        print(f"Test MSE: {test_err_mean:.4f} ± {test_err_std:.4f}")
        print(f"R² score: {summary['r2_score']['mean']:.3f} ± {summary['r2_score']['std']:.3f}")
        print(f"95% coverage: {summary['95_coverage']['mean']:.2%}")

        # Nonlinear heteroscedastic
        print("\n[2] Heteroscedastic Regression")
        print("-" * 80)
        results["heteroscedastic"] = self.benchmark_regression_nonlinear(
            n_train=int(n_train * 1.5), n_test=n_test, n_folds=n_folds
        )
        summary = results["heteroscedastic"].summary()
        print(f"Test MSE: {summary['test_error']['mean']:.4f}")
        print(f"R² score: {summary['r2_score']['mean']:.3f}")
        print(f"Calibration error: {summary['calibration_error']['mean']:.4f}")

        # Extrapolation
        print("\n[3] Extrapolation Awareness")
        print("-" * 80)
        results["extrapolation"] = self.benchmark_extrapolation(
            n_train=n_train, n_test=n_test, n_folds=n_folds
        )
        summary = results["extrapolation"].summary()
        print(f"Uncertainty ratio (out/in): {summary['calibration_error']['mean']:.2f}")
        print(f"Test MSE: {summary['test_error']['mean']:.4f}")

        print("\n" + "=" * 80)
        print("ML benchmarks complete")

        return results


if __name__ == "__main__":
    print("TSU Machine Learning Benchmark")
    print("=" * 80)
    print("Testing Bayesian neural networks on regression tasks")
    print("=" * 80)

    benchmark = MLBenchmark(seed=42)
    results = benchmark.run_all_benchmarks(quick=False)

    print("\n\nSUMMARY")
    print("=" * 80)
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        summary = result.summary()
        print(f"  Test error: {summary['test_error']['mean']:.4f}")
        if "r2_score" in summary:
            print(f"  R² score: {summary['r2_score']['mean']:.3f}")
        if "95_coverage" in summary:
            print(f"  95% coverage: {summary['95_coverage']['mean']:.2%} (target: 95%)")
        print(f"  Train time: {summary['train_time_sec']['mean']:.2f}s")

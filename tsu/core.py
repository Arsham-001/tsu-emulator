"""
Thermodynamic Sampling Unit (TSU) Emulator
Core engine for probabilistic computing via Langevin dynamics
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass


# Exception classes for input validation
class TSUError(Exception):
    """Base exception for TSU platform"""

    pass


class ConfigurationError(TSUError):
    """Invalid configuration parameters"""

    pass


class SamplingError(TSUError):
    """Error during sampling process"""

    pass


@dataclass
class TSUConfig:
    """Configuration for TSU behavior"""

    temperature: float = 1.0  # kT - controls noise amplitude
    dt: float = 0.01  # time step for discretization
    friction: float = 1.0  # gamma - damping coefficient
    n_burnin: int = 100  # steps to discard before collecting samples
    n_steps: int = 500  # steps per sample

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.temperature <= 0:
            raise ConfigurationError(f"Temperature must be positive, got {self.temperature}")
        if self.dt <= 0 or self.dt > 0.1:
            raise ConfigurationError(f"Time step dt must be in (0, 0.1], got {self.dt}")
        if self.friction <= 0:
            raise ConfigurationError(f"Friction must be positive, got {self.friction}")
        if self.n_burnin < 0:
            raise ConfigurationError(f"Burn-in steps must be non-negative, got {self.n_burnin}")
        if self.n_steps <= 0:
            raise ConfigurationError(f"Number of steps must be positive, got {self.n_steps}")


class ThermalSamplingUnit:
    """
    Core TSU emulator using Langevin dynamics.
    Samples from probability distributions via thermal noise simulation.
    """

    def __init__(self, config: Optional[TSUConfig] = None):
        self.config = config or TSUConfig()
        self.sample_count = 0

    def _langevin_step(self, x: np.ndarray, grad_energy: np.ndarray) -> np.ndarray:
        """
        Single step of overdamped Langevin dynamics:
        dx = -∇E(x)dt + √(2kT)dW

        This is the fundamental TSU operation.
        """
        cfg = self.config

        # Deterministic drift term (follows energy gradient downhill)
        drift = -grad_energy * cfg.dt / cfg.friction

        # Stochastic diffusion term (thermal noise)
        noise_scale = np.sqrt(2 * cfg.temperature * cfg.dt / cfg.friction)
        diffusion = noise_scale * np.random.randn(*x.shape)

        return x + drift + diffusion

    def _numerical_gradient(
        self, energy_fn: Callable, x: np.ndarray, eps: float = 1e-5
    ) -> np.ndarray:
        """Compute gradient via finite differences"""
        x = np.atleast_1d(x)
        grad = np.zeros_like(x)

        for i in range(len(x)):
            x_plus = x.copy()
            x_plus[i] += eps
            x_minus = x.copy()
            x_minus[i] -= eps
            val_plus = float(energy_fn(x_plus))
            val_minus = float(energy_fn(x_minus))
            grad[i] = (val_plus - val_minus) / (2 * eps)

        return grad

    def sample_from_energy(
        self,
        energy_fn: Callable,
        x_init: np.ndarray,
        n_samples: int = 1,
        return_trajectory: bool = False,
    ):
        """
        Sample from distribution P(x) ∝ exp(-E(x)/kT)

        Args:
            energy_fn: Function E(x) defining the distribution
            x_init: Initial state
            n_samples: Number of independent samples to generate
            return_trajectory: If True, return full Langevin trajectory

        Returns:
            samples array of shape (n_samples, dim) or trajectory

        Raises:
            SamplingError: If n_samples invalid or energy_fn fails
        """
        if n_samples <= 0:
            raise SamplingError(f"n_samples must be positive, got {n_samples}")

        try:
            # Test energy function on initial state
            test_energy = energy_fn(x_init)
            if not isinstance(test_energy, (int, float, np.number)):
                raise SamplingError(
                    f"Energy function must return scalar, got {type(test_energy)}"
                )
        except Exception as e:
            raise SamplingError(f"Energy function failed on initial state: {e}")

        cfg = self.config
        x = np.atleast_1d(x_init).copy()
        samples = []
        trajectory = [] if return_trajectory else None

        for sample_idx in range(n_samples):
            # Randomize starting point slightly for each sample
            if sample_idx > 0:
                x = x_init + 0.1 * np.random.randn(*x_init.shape)

            # Burn-in period
            for _ in range(cfg.n_burnin):
                grad = self._numerical_gradient(energy_fn, x)
                x = self._langevin_step(x, grad)

            # Sampling period
            for step in range(cfg.n_steps):
                grad = self._numerical_gradient(energy_fn, x)
                x = self._langevin_step(x, grad)

                if return_trajectory and trajectory is not None:
                    trajectory.append(x.copy())

            samples.append(x.copy())
            self.sample_count += 1

        samples = np.array(samples)
        return (samples, trajectory) if return_trajectory else samples

    def p_bit(self, prob: float, n_samples: int = 1) -> np.ndarray:
        """
        Probabilistic bit: fundamental TSU building block.

        Args:
            prob: Probability of returning 1 (must be in [0,1])
            n_samples: Number of samples to generate

        Returns:
            Array of binary samples

        Raises:
            ConfigurationError: If prob not in [0,1] or n_samples invalid
        """
        if not 0 <= prob <= 1:
            raise ConfigurationError(f"Probability must be in [0,1], got {prob}")
        if n_samples <= 0:
            raise ConfigurationError(f"n_samples must be positive, got {n_samples}")

        prob_clipped = float(np.clip(prob, 1e-10, 1 - 1e-10))

        # For efficiency, use small state space
        def energy(x):
            x0 = float(np.atleast_1d(x)[0])
            x_clipped = float(np.clip(x0, 1e-10, 1 - 1e-10))
            return -np.log(prob_clipped) * x_clipped - np.log(1 - prob_clipped) * (
                1 - x_clipped
            )

        # Start near the expected value
        x_init = np.array([prob_clipped])
        result = self.sample_from_energy(energy, x_init, n_samples)

        if isinstance(result, tuple):
            samples = result[0]
        else:
            samples = result

        # Threshold to binary
        return (samples.flatten() > 0.5).astype(int)

    def sample_gaussian(
        self, mu: float = 0.0, sigma: float = 1.0, n_samples: int = 1
    ) -> np.ndarray:
        """
        Sample from Gaussian distribution N(mu, sigma^2)

        Args:
            mu: Mean of the distribution
            sigma: Standard deviation (must be positive)
            n_samples: Number of samples to generate

        Returns:
            Array of samples

        Raises:
            ConfigurationError: If sigma <= 0 or n_samples <= 0
        """
        if sigma <= 0:
            raise ConfigurationError(f"Sigma must be positive, got {sigma}")
        if n_samples <= 0:
            raise ConfigurationError(f"n_samples must be positive, got {n_samples}")

        def energy(x):
            # Ensure energy returns a scalar float even when x is an array
            x0 = float(np.atleast_1d(x)[0])
            return 0.5 * ((x0 - mu) / sigma) ** 2

        x_init = np.array([mu])
        result = self.sample_from_energy(energy, x_init, n_samples)

        if isinstance(result, tuple):
            samples = result[0]
        else:
            samples = result

        return samples.flatten()

    def sample_categorical(self, probs: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Sample from categorical distribution.
        Uses Gibbs distribution over discrete states.
        """
        probs = np.array(probs)
        probs = probs / probs.sum()  # normalize

        def energy(x):
            # Map continuous x to discrete categories via modulo. Convert x to
            # a scalar first (1-element arrays are common in this code).
            x0 = int(abs(float(np.atleast_1d(x)[0])))
            idx = x0 % len(probs)
            return -np.log(probs[idx] + 1e-10)

        x_init = np.array([0.0])
        result = self.sample_from_energy(energy, x_init, n_samples)

        if isinstance(result, tuple):
            samples_cont = result[0]
        else:
            samples_cont = result

        # Map to discrete categories
        samples = np.abs(samples_cont.flatten()).astype(int) % len(probs)
        return samples


class ProbabilisticNeuron:
    """
    Single probabilistic neuron using TSU sampling.
    Naturally implements stochastic activation.
    """

    def __init__(self, tsu: ThermalSamplingUnit):
        self.tsu = tsu

    def activate(self, weights: np.ndarray, inputs: np.ndarray, bias: float = 0.0) -> int:
        """
        Probabilistic activation: output ~ Bernoulli(σ(w·x + b))
        """
        logit = np.dot(weights, inputs) + bias
        prob = 1.0 / (1.0 + np.exp(-logit))  # sigmoid
        return self.tsu.p_bit(prob, n_samples=1)[0]

    def forward_stochastic(
        self, weights: np.ndarray, inputs: np.ndarray, bias: float = 0.0, n_samples: int = 10
    ) -> float:
        """
        Get expected output by sampling multiple times
        """
        outputs = [self.activate(weights, inputs, bias) for _ in range(n_samples)]
        return float(np.mean(outputs))


# Utility functions for validation
def validate_distribution(
    samples: np.ndarray, expected_dist: str, params: dict, alpha: float = 0.05
) -> dict:
    """
    Validate samples match expected distribution using statistical tests.
    Returns test results and statistics.
    """
    from scipy import stats

    results = {"mean": np.mean(samples), "std": np.std(samples), "n_samples": len(samples)}

    if expected_dist == "gaussian":
        mu, sigma = params.get("mu", 0), params.get("sigma", 1)
        results["expected_mean"] = mu
        results["expected_std"] = sigma

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(samples, "norm", args=(mu, sigma))
        results["ks_statistic"] = ks_stat
        results["ks_pvalue"] = p_value
        results["passes_ks_test"] = p_value > alpha

    elif expected_dist == "bernoulli":
        p = params.get("p", 0.5)
        results["expected_mean"] = p
        results["empirical_prob"] = np.mean(samples)
        results["error"] = abs(np.mean(samples) - p)
        results["passes_test"] = results["error"] < 0.05

    return results


if __name__ == "__main__":
    # Quick demonstration
    print("=== TSU Emulator Demo ===\n")

    # Initialize TSU
    config = TSUConfig(temperature=1.0, n_steps=300)
    tsu = ThermalSamplingUnit(config)

    # Test 1: Gaussian sampling
    print("1. Gaussian Sampling N(0, 1)")
    gaussian_samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)
    stats = validate_distribution(gaussian_samples, "gaussian", {"mu": 0, "sigma": 1})
    print(f"   Mean: {stats['mean']:.3f} (expected: 0.000)")
    print(f"   Std:  {stats['std']:.3f} (expected: 1.000)")
    print(f"   KS test p-value: {stats['ks_pvalue']:.3f}")
    print(f"   Passes: {'[OK]' if stats['passes_ks_test'] else '[FAIL]'}\n")

    # Test 2: Probabilistic bit
    print("2. Probabilistic Bit (p=0.7)")
    pbit_samples = tsu.p_bit(prob=0.7, n_samples=1000)
    stats = validate_distribution(pbit_samples, "bernoulli", {"p": 0.7})
    print(f"   Empirical prob: {stats['empirical_prob']:.3f} (expected: 0.700)")
    print(f"   Error: {stats['error']:.3f}")
    print(f"   Passes: {'[OK]' if stats['passes_test'] else '[FAIL]'}\n")

    # Test 3: Probabilistic neuron
    print("3. Probabilistic Neuron")
    neuron = ProbabilisticNeuron(tsu)
    weights = np.array([0.5, -0.3, 0.8])
    inputs = np.array([1.0, 0.5, -0.2])
    expected_output = neuron.forward_stochastic(weights, inputs, n_samples=100)
    print(f"   Weights: {weights}")
    print(f"   Inputs: {inputs}")
    print(f"   Expected output: {expected_output:.3f}")

    print(f"\nTotal samples generated: {tsu.sample_count}")

    # Test error handling
    print("\n" + "=" * 70)
    print("TESTING INPUT VALIDATION & ERROR HANDLING")
    print("=" * 70)

    test_passed = 0
    test_failed = 0

    # Test 1: Negative temperature
    print("\n[Test 1] Negative temperature")
    try:
        config = TSUConfig(temperature=-1.0)
        print("  [ERROR] Failed to catch negative temperature")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 2: Invalid dt (too large)
    print("\n[Test 2] Invalid time step (dt > 0.1)")
    try:
        config = TSUConfig(dt=0.5)
        print("  [ERROR] Failed to catch invalid dt")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 3: Negative friction
    print("\n[Test 3] Negative friction")
    try:
        config = TSUConfig(friction=-0.5)
        print("  [ERROR] Failed to catch negative friction")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 4: Negative n_steps
    print("\n[Test 4] Negative n_steps")
    try:
        config = TSUConfig(n_steps=-10)
        print("  [ERROR] Failed to catch negative n_steps")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 5: Invalid probability (too high)
    print("\n[Test 5] Invalid probability (p > 1)")
    try:
        tsu = ThermalSamplingUnit()
        tsu.p_bit(prob=1.5)
        print("  [ERROR] Failed to catch invalid probability")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 6: Invalid probability (too low)
    print("\n[Test 6] Invalid probability (p < 0)")
    try:
        tsu = ThermalSamplingUnit()
        tsu.p_bit(prob=-0.5)
        print("  [ERROR] Failed to catch invalid probability")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 7: Negative n_samples in p_bit
    print("\n[Test 7] Negative n_samples in p_bit")
    try:
        tsu = ThermalSamplingUnit()
        tsu.p_bit(prob=0.5, n_samples=-5)
        print("  [ERROR] Failed to catch negative n_samples")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 8: Negative sigma in sample_gaussian
    print("\n[Test 8] Negative sigma in sample_gaussian")
    try:
        tsu = ThermalSamplingUnit()
        tsu.sample_gaussian(sigma=-1.0)
        print("  [ERROR] Failed to catch negative sigma")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 9: Negative n_samples in sample_gaussian
    print("\n[Test 9] Negative n_samples in sample_gaussian")
    try:
        tsu = ThermalSamplingUnit()
        tsu.sample_gaussian(n_samples=-10)
        print("  [ERROR] Failed to catch negative n_samples")
        test_failed += 1
    except ConfigurationError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 10: Negative n_samples in sample_from_energy
    print("\n[Test 10] Negative n_samples in sample_from_energy")
    try:
        tsu = ThermalSamplingUnit()
        tsu.sample_from_energy(lambda x: np.sum(x**2), np.array([0]), n_samples=-1)
        print("  [ERROR] Failed to catch negative n_samples")
        test_failed += 1
    except SamplingError as e:
        print(f"  [OK] Caught error: {e}")
        test_passed += 1

    # Test 11: Valid configuration (should not raise)
    print("\n[Test 11] Valid configuration")
    try:
        config = TSUConfig(temperature=1.0, dt=0.01, friction=1.0, n_burnin=100, n_steps=500)
        print("  [OK] Valid configuration accepted")
        test_passed += 1
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        test_failed += 1

    # Test 12: Valid p_bit call (should not raise)
    print("\n[Test 12] Valid p_bit call")
    try:
        tsu = ThermalSamplingUnit()
        samples = tsu.p_bit(prob=0.5, n_samples=5)
        print(f"  [OK] Generated {len(samples)} valid samples")
        test_passed += 1
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        test_failed += 1

    # Test 13: Valid sample_gaussian call (should not raise)
    print("\n[Test 13] Valid sample_gaussian call")
    try:
        tsu = ThermalSamplingUnit()
        samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=10)
        print(f"  [OK] Generated {len(samples)} valid samples")
        test_passed += 1
    except Exception as e:
        print(f"  [ERROR] Failed: {e}")
        test_failed += 1

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Passed: {test_passed}/13 [OK]")
    print(f"Tests Failed: {test_failed}/13")

    if test_failed == 0:
        print("\n ALL ERROR HANDLING TESTS PASSED!")
    else:
        print(f"\n[WARNING] {test_failed} test(s) failed")

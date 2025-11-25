"""
Test suite for TSU core functionality.
Run with: pytest tests/
"""

import pytest
import numpy as np
from scipy import stats

import sys
from tsu.core import ThermalSamplingUnit, TSUConfig, ConfigurationError, validate_distribution


class TestTSUConfig:
    """Test configuration validation"""

    def test_valid_config(self):
        """Valid configuration should work"""
        config = TSUConfig(temperature=1.0, dt=0.01, n_steps=100)
        assert config.temperature == 1.0

    def test_negative_temperature(self):
        """Negative temperature should raise error"""
        with pytest.raises(ConfigurationError):
            TSUConfig(temperature=-1.0)

    def test_invalid_dt(self):
        """Invalid dt should raise error"""
        with pytest.raises(ConfigurationError):
            TSUConfig(dt=-0.01)
        with pytest.raises(ConfigurationError):
            TSUConfig(dt=1.0)  # Too large

    def test_negative_steps(self):
        """Negative steps should raise error"""
        with pytest.raises(ConfigurationError):
            TSUConfig(n_steps=-10)


class TestGaussianSampling:
    """Test Gaussian distribution sampling"""

    def test_sample_shape(self):
        """Samples should have correct shape"""
        tsu = ThermalSamplingUnit()
        samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=100)
        assert len(samples) == 100

    def test_mean_approximately_correct(self):
        """Sample mean should be close to true mean"""
        tsu = ThermalSamplingUnit(TSUConfig(n_steps=300))
        samples = tsu.sample_gaussian(mu=5.0, sigma=1.0, n_samples=1000)

        sample_mean = np.mean(samples)
        # Should be within 0.2 of true mean (95% confidence)
        assert abs(sample_mean - 5.0) < 0.2, f"Mean {sample_mean} too far from 5.0"

    def test_std_approximately_correct(self):
        """Sample std should be close to true std"""
        tsu = ThermalSamplingUnit(TSUConfig(n_steps=300))
        samples = tsu.sample_gaussian(mu=0, sigma=2.0, n_samples=1000)

        sample_std = np.std(samples)
        # Should be within 0.3 of true std
        assert abs(sample_std - 2.0) < 0.3, f"Std {sample_std} too far from 2.0"

    def test_statistical_validity(self):
        """Samples should pass KS test for normality"""
        tsu = ThermalSamplingUnit(TSUConfig(n_steps=300))
        samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)

        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.kstest(samples, "norm")
        assert p_value > 0.05, f"Samples failed KS test (p={p_value})"

    def test_invalid_sigma(self):
        """Negative sigma should raise error"""
        tsu = ThermalSamplingUnit()
        with pytest.raises(ConfigurationError):
            tsu.sample_gaussian(sigma=-1.0)


class TestProbabilisticBit:
    """Test p-bit sampling"""

    def test_sample_shape(self):
        """P-bit samples should have correct shape"""
        tsu = ThermalSamplingUnit()
        samples = tsu.p_bit(prob=0.5, n_samples=100)
        assert len(samples) == 100

    def test_binary_output(self):
        """P-bit should only output 0 or 1"""
        tsu = ThermalSamplingUnit()
        samples = tsu.p_bit(prob=0.7, n_samples=100)
        assert set(samples).issubset({0, 1})

    def test_probability_approximately_correct(self):
        """Empirical probability should match input"""
        tsu = ThermalSamplingUnit()

        for true_prob in [0.2, 0.5, 0.8]:
            samples = tsu.p_bit(prob=true_prob, n_samples=1000)
            empirical_prob = np.mean(samples)

            # Should be within 0.05 of true probability
            assert (
                abs(empirical_prob - true_prob) < 0.05
            ), f"Empirical {empirical_prob} too far from {true_prob}"

    def test_invalid_probability(self):
        """Invalid probabilities should raise error"""
        tsu = ThermalSamplingUnit()
        with pytest.raises(ConfigurationError):
            tsu.p_bit(prob=-0.1)
        with pytest.raises(ConfigurationError):
            tsu.p_bit(prob=1.5)


class TestValidation:
    """Test validation utilities"""

    def test_gaussian_validation(self):
        """Validate Gaussian samples"""
        tsu = ThermalSamplingUnit(TSUConfig(n_steps=300))
        samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)

        result = validate_distribution(samples, "gaussian", {"mu": 0, "sigma": 1})

        assert result["passes_ks_test"]
        assert abs(result["mean"] - 0) < 0.2
        assert abs(result["std"] - 1) < 0.3

    def test_bernoulli_validation(self):
        """Validate Bernoulli samples"""
        tsu = ThermalSamplingUnit()
        samples = tsu.p_bit(prob=0.7, n_samples=1000)

        result = validate_distribution(samples, "bernoulli", {"p": 0.7})

        assert result["passes_test"]
        assert abs(result["empirical_prob"] - 0.7) < 0.05


if __name__ == "__main__":
    # Run tests manually if pytest not available
    print("Running manual tests...")

    test_config = TestTSUConfig()
    test_config.test_valid_config()
    print("[OK] Config validation works")

    test_gaussian = TestGaussianSampling()
    test_gaussian.test_sample_shape()
    test_gaussian.test_mean_approximately_correct()
    print("[OK] Gaussian sampling works")

    test_pbit = TestProbabilisticBit()
    test_pbit.test_sample_shape()
    test_pbit.test_probability_approximately_correct()
    print("[OK] P-bit sampling works")

    print("\n[OK] All manual tests passed!")

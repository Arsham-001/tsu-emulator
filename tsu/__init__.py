"""
TSU Platform - Thermodynamic Sampling Unit Emulator

A software platform for probabilistic computing using thermodynamic sampling.
"""

__version__ = "0.1.0"
__author__ = "Arsham Rocky"

# Import low-level core components
from .core import (
    ThermalSamplingUnit,
    TSUConfig,
    ProbabilisticNeuron,
    validate_distribution,
    TSUError,
    ConfigurationError,
    SamplingError,
)

# Create a convenient alias for the main class
from .core import ThermalSamplingUnit as TSU

# Import hardware-accurate Gibbs sampling (matches Extropic X0 chip)
from .gibbs import (
    GibbsSampler,
    GibbsConfig,
    HardwareEmulator,
)

# Import energy-based models (Ising, Boltzmann machines, etc.)
from .models import (
    IsingModel,
    IsingChain,
    IsingGrid,
    demonstrate_phase_transition,
)

# Import and expose the high-level, user-friendly API
from .api import (
    Backend,
    SamplingResult,
    Sampler,
    GaussianSampler,
    MultimodalSampler,
    BayesianSampler,
    OptimizationProblem,
    MaxCutProblem,
    ProbabilisticLayer,
    StochasticLinear,
    BernoulliActivation,
    ProbabilisticModel,
    sample_gaussian,
    sample_multimodal,
    optimize,
)

__all__ = [
    # Core Langevin dynamics
    "ThermalSamplingUnit",
    "TSU",
    "TSUConfig",
    "ProbabilisticNeuron",
    "validate_distribution",
    "TSUError",
    "ConfigurationError",
    "SamplingError",
    # Hardware-accurate Gibbs sampling (Extropic-compatible)
    "GibbsSampler",
    "GibbsConfig",
    "HardwareEmulator",
    # Energy-based models
    "IsingModel",
    "IsingChain",
    "IsingGrid",
    "demonstrate_phase_transition",
    # High-level API
    "Backend",
    "SamplingResult",
    "Sampler",
    "GaussianSampler",
    "MultimodalSampler",
    "BayesianSampler",
    "OptimizationProblem",
    "MaxCutProblem",
    "ProbabilisticLayer",
    "StochasticLinear",
    "BernoulliActivation",
    "ProbabilisticModel",
    "sample_gaussian",
    "sample_multimodal",
    "optimize",
]


def quick_demo():
    """Run a quick demo of TSU capabilities from the package root."""
    print("TSU Platform Quick Demo")
    print("=" * 50)
    
    tsu = ThermalSamplingUnit()
    
    # Gaussian sampling
    print("\n1. Gaussian Sampling")
    samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=100)
    print(f"   Generated {len(samples)} samples with Mean: {samples.mean():.3f}, Std: {samples.std():.3f}")
    
    # P-bit
    print("\n2. Probabilistic Bit")
    bits = tsu.p_bit(prob=0.7, n_samples=100)
    print(f"   P(1) = {bits.mean():.2f} (expected: 0.70)")
    
    print("\n" + "=" * 50)
    print("[OK] Quick demo complete!")
    print("  Run 'python -m tsu.demos' for the full demonstration.")
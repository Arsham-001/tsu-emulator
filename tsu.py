"""
TSU (Thermodynamic Sampling Unit) - Main Package
Re-exports public API from tsu_api for easy importing

Usage:
    import tsu
    samples = tsu.sample_gaussian(mu=0, sigma=1, n=1000)
"""

# Import all public classes and functions from tsu_api
from tsu_api import (
    # Enums
    Backend,
    
    # Data classes
    SamplingResult,
    
    # Base classes
    Sampler,
    OptimizationProblem,
    ProbabilisticLayer,
    ProbabilisticModel,
    
    # Samplers
    GaussianSampler,
    MultimodalSampler,
    BayesianSampler,
    
    # Optimization
    MaxCutProblem,
    
    # Layers
    StochasticLinear,
    BernoulliActivation,
    
    # Functions
    sample_gaussian,
    sample_multimodal,
    compare_samplers,
    optimize,
)

# Version info
__version__ = "0.1.0"
__author__ = "TSU Research Team"

# Public API
__all__ = [
    # Enums
    "Backend",
    
    # Data classes
    "SamplingResult",
    
    # Base classes
    "Sampler",
    "OptimizationProblem",
    "ProbabilisticLayer",
    "ProbabilisticModel",
    
    # Samplers
    "GaussianSampler",
    "MultimodalSampler",
    "BayesianSampler",
    
    # Optimization
    "MaxCutProblem",
    
    # Layers
    "StochasticLinear",
    "BernoulliActivation",
    
    # Functions
    "sample_gaussian",
    "sample_multimodal",
    "compare_samplers",
    "optimize",
]

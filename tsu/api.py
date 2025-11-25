"""
TSU Platform - High-Level Developer API
The TensorFlow/PyTorch equivalent for Thermodynamic Computing

Design principles:
- Intuitive: Feels like modern ML frameworks
- Modular: Easy to extend and customize
- Hardware-agnostic: Works with emulator or real TSU chips
- Production-ready: Not just research code

Example usage:
    import tsu

    # Sample from distribution
    sampler = tsu.GaussianSampler(mu=0, sigma=1)
    samples = sampler.sample(n=1000)

    # Solve optimization
    problem = tsu.MaxCut(graph)
    solution = tsu.optimize(problem, method='tsu')

    # Build probabilistic model
    model = tsu.ProbabilisticModel()
    model.add(tsu.layers.StochasticLinear(10, 5))
    model.add(tsu.layers.BernoulliActivation())
    samples = model.sample(input_data)
"""

import numpy as np
from typing import Union, Callable, Optional, List, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .core import ThermalSamplingUnit, TSUConfig


class Backend(Enum):
    """Available computation backends"""

    EMULATOR = "emulator"  # Software emulation (what we have)
    CLOUD = "cloud"  # Remote execution
    HARDWARE = "hardware"  # Real TSU chips (future)
    HYBRID = "hybrid"  # Mix of backends


@dataclass
class SamplingResult:
    """Container for sampling results with metadata"""

    samples: np.ndarray
    energy: Optional[np.ndarray] = None
    acceptance_rate: Optional[float] = None
    time_elapsed: Optional[float] = None
    backend_used: str = "emulator"
    hardware_projection: Optional[Dict] = None


class Sampler(ABC):
    """
    Base class for all TSU samplers.
    Provides consistent interface across different distributions.
    """

    def __init__(self, backend: Backend = Backend.EMULATOR, config: Optional[TSUConfig] = None):
        self.backend = backend
        self.config = config or TSUConfig()
        self._tsu = ThermalSamplingUnit(self.config)

    @abstractmethod
    def energy_function(self, x: np.ndarray) -> float:
        """Energy function defining the distribution"""
        pass

    def sample(
        self, n: int = 1000, return_metadata: bool = False
    ) -> Union[np.ndarray, SamplingResult]:
        """
        Sample from the distribution.

        Args:
            n: Number of samples
            return_metadata: If True, return SamplingResult with timing info

        Returns:
            Samples array or SamplingResult object
        """
        import time

        start = time.time()

        if self.backend == Backend.EMULATOR:
            x_init = self._get_initial_state()
            result = self._tsu.sample_from_energy(self.energy_function, x_init, n_samples=n)
            if isinstance(result, tuple):
                samples = result[0]
            else:
                samples = result
        else:
            raise NotImplementedError(f"Backend {self.backend} not yet implemented")

        elapsed = time.time() - start

        if return_metadata:
            return SamplingResult(
                samples=samples, time_elapsed=elapsed, backend_used=str(self.backend.value)
            )
        return samples

    @abstractmethod
    def _get_initial_state(self) -> np.ndarray:
        """Get initial state for sampling"""
        pass


class GaussianSampler(Sampler):
    """Sample from Gaussian distribution N(μ, σ²)"""

    def __init__(self, mu: float = 0.0, sigma: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.mu = mu
        self.sigma = sigma

    def energy_function(self, x: np.ndarray) -> float:
        result = 0.5 * ((x - self.mu) / self.sigma) ** 2
        return float(np.mean(result)) if isinstance(result, np.ndarray) else float(result)

    def _get_initial_state(self) -> np.ndarray:
        return np.array([self.mu])


class MultimodalSampler(Sampler):
    """Sample from mixture of Gaussians"""

    def __init__(
        self, centers: Union[List[np.ndarray], List[List]], weights: List[float], **kwargs
    ):
        super().__init__(**kwargs)
        self.centers = [np.array(c) for c in centers]
        self.weights = np.array(weights) / np.sum(weights)
        self.dim = len(self.centers[0])

    def energy_function(self, x: np.ndarray) -> float:
        x = np.atleast_1d(x)
        prob = 0
        for i, center in enumerate(self.centers):
            dist_sq = np.sum((x - center) ** 2)
            prob += self.weights[i] * np.exp(-0.5 * dist_sq)
        return -np.log(prob + 1e-10)

    def _get_initial_state(self) -> np.ndarray:
        return np.random.randn(self.dim) * 0.5


class BayesianSampler(Sampler):
    """
    Sample from Bayesian posterior P(θ|data) ∝ P(data|θ) * P(θ)

    Example:
        def log_likelihood(theta, X, y):
            return -0.5 * np.sum((y - X @ theta)**2)

        def log_prior(theta):
            return -0.5 * np.sum(theta**2)

        sampler = BayesianSampler(log_likelihood, log_prior, X, y)
        posterior_samples = sampler.sample(1000)
    """

    def __init__(
        self,
        log_likelihood: Callable,
        log_prior: Callable,
        *args,
        dim: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.log_likelihood = log_likelihood
        self.log_prior = log_prior
        self.likelihood_args = args
        self.dim = dim

    def energy_function(self, theta: np.ndarray) -> float:
        theta = np.atleast_1d(theta)
        log_lik = self.log_likelihood(theta, *self.likelihood_args)
        log_pri = self.log_prior(theta)
        return -(log_lik + log_pri)

    def _get_initial_state(self) -> np.ndarray:
        if self.dim is None:
            raise ValueError("Must specify dimension for Bayesian sampler")
        return np.random.randn(self.dim) * 0.1


# High-level convenience functions (like TensorFlow's functional API)


def sample_gaussian(
    mu: float = 0, sigma: float = 1, n: int = 1000, backend: Backend = Backend.EMULATOR
) -> np.ndarray:
    """Quick Gaussian sampling (functional API)"""
    sampler = GaussianSampler(mu, sigma, backend=backend)
    result = sampler.sample(n, return_metadata=False)
    return result if isinstance(result, np.ndarray) else result.samples


def sample_multimodal(
    centers: List, weights: List, n: int = 1000, backend: Backend = Backend.EMULATOR
) -> np.ndarray:
    """Quick multimodal sampling (functional API)"""
    sampler = MultimodalSampler(centers, weights, backend=backend)
    result = sampler.sample(n, return_metadata=False)
    return result if isinstance(result, np.ndarray) else result.samples


def compare_samplers(
    distribution, methods: List[str] = ["tsu", "mcmc"], n_samples: int = 1000
) -> Dict[str, SamplingResult]:
    """
    Compare different sampling methods on same distribution.
    Returns performance metrics for each.
    """
    results = {}

    for method in methods:
        if method == "tsu":
            sampler = distribution  # Assume it's a Sampler object
            result = sampler.sample(n_samples, return_metadata=True)
            results["tsu"] = result
        elif method == "mcmc":
            pass

    return results


# Optimization interface (like scipy.optimize but for TSU)


class OptimizationProblem(ABC):
    """Base class for optimization problems"""

    @abstractmethod
    def objective(self, x: np.ndarray) -> float:
        """Objective function to minimize"""
        pass

    @abstractmethod
    def dimension(self) -> int:
        """Problem dimension"""
        pass


class MaxCutProblem(OptimizationProblem):
    """Maximum Cut problem on a graph"""

    def __init__(self, adjacency_matrix: np.ndarray):
        self.adj = adjacency_matrix
        self.n = len(adjacency_matrix)

    def objective(self, x: np.ndarray) -> float:
        """Energy = -Σ A_ij * s_i * s_j"""
        spins = np.sign(x)
        spins[spins == 0] = 1

        energy = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                energy -= self.adj[i, j] * spins[i] * spins[j]
        return energy

    def dimension(self) -> int:
        return self.n


def optimize(
    problem: OptimizationProblem, method: str = "tsu", max_iterations: int = 1000
) -> Dict[str, Any]:
    """
    Solve optimization problem.

    Args:
        problem: OptimizationProblem instance
        method: 'tsu' or 'classical'
        max_iterations: Iteration budget

    Returns:
        Dictionary with solution and metadata
    """
    if method == "tsu":
        # Use TSU-based simulated annealing
        config = TSUConfig(temperature=1.0, n_steps=100)
        tsu = ThermalSamplingUnit(config)

        x = np.random.randn(problem.dimension())
        best_x = x.copy()
        best_energy = problem.objective(x)

        energy_history = [best_energy]

        for iteration in range(max_iterations):
            # Annealing schedule
            T = 1.0 * (1 - iteration / max_iterations)
            config.temperature = max(T, 0.01)

            # Sample new configuration
            samples = tsu.sample_from_energy(problem.objective, x, n_samples=1)
            x_new = samples[0]
            energy_new = problem.objective(x_new)

            if energy_new < best_energy:
                best_energy = energy_new
                best_x = x_new.copy()

            x = x_new
            energy_history.append(energy_new)

        return {
            "solution": best_x,
            "energy": best_energy,
            "energy_history": energy_history,
            "method": "tsu",
        }
    else:
        raise NotImplementedError(f"Method {method} not implemented")


# Model building (like Keras/PyTorch)


class ProbabilisticLayer(ABC):
    """Base class for probabilistic layers"""

    @abstractmethod
    def forward(self, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Forward pass with sampling"""
        pass


class StochasticLinear(ProbabilisticLayer):
    """Linear layer with stochastic weights"""

    def __init__(self, in_features: int, out_features: int):
        self.in_features = in_features
        self.out_features = out_features
        # Initialize with small random weights
        self.weight_mean = np.random.randn(out_features, in_features) * 0.1
        self.weight_std = np.ones((out_features, in_features)) * 0.01

    def forward(self, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """
        Forward pass: y = Wx where W is stochastic
        Returns n_samples different outputs
        """
        outputs = []
        for _ in range(n_samples):
            # Sample weights from N(weight_mean, weight_std²)
            weights = self.weight_mean + self.weight_std * np.random.randn(
                self.out_features, self.in_features
            )
            output = weights @ x
            outputs.append(output)
        return np.array(outputs)


class BernoulliActivation(ProbabilisticLayer):
    """Stochastic binary activation"""

    def __init__(self):
        self.tsu = ThermalSamplingUnit(TSUConfig(temperature=1.0))

    def forward(self, x: np.ndarray, n_samples: int = 1) -> np.ndarray:
        """Convert continuous values to stochastic binary outputs"""
        # Sigmoid to get probabilities
        probs = 1 / (1 + np.exp(-x))

        outputs = []
        for _ in range(n_samples):
            output = np.array([self.tsu.p_bit(p, n_samples=1)[0] for p in probs])
            outputs.append(output)

        return np.array(outputs)


class ProbabilisticModel:
    """
    Sequential probabilistic model (like Keras Sequential)

    Example:
        model = ProbabilisticModel()
        model.add(StochasticLinear(10, 5))
        model.add(BernoulliActivation())
        outputs = model.sample(input_data, n_samples=100)
    """

    def __init__(self):
        self.layers = []

    def add(self, layer: ProbabilisticLayer):
        """Add a layer to the model"""
        self.layers.append(layer)

    def sample(self, x: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """
        Sample outputs from the model.
        Returns n_samples different outputs for the same input.
        """
        outputs = [x for _ in range(n_samples)]

        for layer in self.layers:
            # Pass all samples through layer
            new_outputs = []
            for out in outputs:
                layer_out = layer.forward(out, n_samples=1)
                new_outputs.append(layer_out[0])
            outputs = new_outputs

        return np.array(outputs)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Single forward pass (mean prediction)"""
        for layer in self.layers:
            x = layer.forward(x, n_samples=1)[0]
        return x


# Example usage and testing
if __name__ == "__main__":
    print("TSU Platform API Demo")
    print("=" * 60)

    # 1. Simple sampling
    print("\n1. Gaussian Sampling")
    samples = sample_gaussian(mu=0, sigma=1, n=100)
    print(f"   Generated {len(samples)} samples")
    print(f"   Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")

    # 2. Multimodal sampling
    print("\n2. Multimodal Sampling")
    centers = [[0, 0], [3, 3], [-2, 2]]
    weights = [0.3, 0.5, 0.2]
    sampler = MultimodalSampler(centers, weights)
    result = sampler.sample(200, return_metadata=True)
    if isinstance(result, SamplingResult):
        print(f"   Generated {len(result.samples)} samples in {result.time_elapsed:.2f}s")
    else:
        print(f"   Generated {len(result)} samples")

    # 3. Optimization
    print("\n3. Optimization Example")
    # Create random graph
    adj = np.random.rand(10, 10)
    adj = (adj + adj.T) / 2  # Make symmetric
    problem = MaxCutProblem(adj)
    result = optimize(problem, method="tsu", max_iterations=100)
    print(f"   Best energy found: {result['energy']:.2f}")

    # 4. Probabilistic model
    print("\n4. Probabilistic Model")
    model = ProbabilisticModel()
    model.add(StochasticLinear(5, 3))
    model.add(BernoulliActivation())

    input_data = np.random.randn(5)
    outputs = model.sample(input_data, n_samples=10)
    print(f"   Input shape: {input_data.shape}")
    print(f"   Output samples shape: {outputs.shape}")
    print(f"   Output mean: {np.mean(outputs, axis=0)}")

    print("\n" + "=" * 60)
    print("TSU Platform API is operational!")

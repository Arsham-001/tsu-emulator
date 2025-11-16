# TSU Core Documentation

## Overview
Complete API documentation for `tsu_core.py` with comprehensive docstrings for all major functions and classes.

---

## ThermalSamplingUnit

### `__init__(config: Optional[TSUConfig] = None) -> None`

Initialize a Thermal Sampling Unit (TSU).

TSU uses overdamped Langevin dynamics to sample from probability distributions defined by energy functions. This creates a natural hardware mapping for probabilistic computing.

**Args:**
- `config`: TSUConfig object specifying temperature, timestep, friction, and sampling parameters. If None, uses default configuration (T=1.0, dt=0.01, friction=1.0, n_steps=500).

**Returns:** None

**Example:**
```python
# Using default configuration
tsu = ThermalSamplingUnit()

# Using custom configuration
config = TSUConfig(temperature=2.0, n_steps=1000)
tsu = ThermalSamplingUnit(config)
```

---

### `_langevin_step(x: np.ndarray, grad_energy: np.ndarray) -> np.ndarray`

Execute one step of overdamped Langevin dynamics.

This is the **core TSU operation** that implements stochastic gradient descent with thermal noise. The equation is:
```
dx = -∇E(x)·dt/γ + √(2kT·dt/γ)·dW
```

where:
- E(x) is the energy (potential) function
- ∇E(x) is the energy gradient (deterministic force)
- T is temperature (controls noise amplitude)
- γ is friction coefficient (controls relaxation timescale)
- dW is Gaussian white noise

The drift term pulls toward lower energy states (like gradient descent), while noise allows escape from local minima (exploration).

**Args:**
- `x`: Current state as ndarray of shape (d,) where d is dimension
- `grad_energy`: Gradient of energy function ∂E/∂x at current state, same shape as x

**Returns:** Updated state after one Langevin step, shape (d,)

**Example:**
```python
tsu = ThermalSamplingUnit(TSUConfig(temperature=1.0, dt=0.01))
x = np.array([0.0, 1.0])
grad_E = np.array([0.5, -0.3])
x_new = tsu._langevin_step(x, grad_E)
# x_new.shape == (2,)
```

---

### `sample_from_energy(energy_fn: Callable, x_init: np.ndarray, n_samples: int = 1, return_trajectory: bool = False)`

Generate samples from an arbitrary probability distribution via energy function.

Uses Langevin dynamics to sample from the Gibbs distribution:
```
P(x) ∝ exp(-E(x)/kT)
```

where E(x) is any user-provided energy (potential) function and T is temperature. By adjusting E(x), you can sample from any distribution.

**Algorithm:**
1. Start from x_init (optionally perturbed per sample)
2. Discard n_burnin steps to reach equilibrium
3. Collect n_steps samples with full Langevin dynamics
4. Return the final state of each sampling trajectory

**Args:**
- `energy_fn`: Callable E(x) → scalar energy. Should accept ndarray of any shape matching x_init.
- `x_init`: Initial state as ndarray, shape (d,). Defines dimensionality.
- `n_samples`: Number of independent samples to collect. Default=1.
- `return_trajectory`: If True, return (samples, trajectory). If False, return only samples. Default=False.

**Returns:**
- If return_trajectory=False: ndarray of shape (n_samples, d) containing samples from P(x) ∝ exp(-E(x)/kT)
- If return_trajectory=True: Tuple (samples, trajectory) where trajectory is list of all intermediate states during sampling

**Raises:**
- `SamplingError`: If n_samples ≤ 0, if energy_fn returns non-scalar, or if energy_fn raises exception on x_init

**Example:**
```python
tsu = ThermalSamplingUnit(TSUConfig(n_steps=100))

def gaussian_energy(x):
    return 0.5 * np.sum(x**2)  # N(0, I) in D dimensions

x_init = np.zeros(5)
samples = tsu.sample_from_energy(gaussian_energy, x_init, n_samples=1000)
# samples.shape == (1000, 5)
```

---

### `p_bit(prob: float, n_samples: int = 1) -> np.ndarray`

Generate probabilistic bit samples (fundamental TSU building block).

A p-bit is a random variable that returns 1 with probability `prob` and 0 with probability (1 - prob). This is the basic stochastic component used to build probabilistic circuits and neural networks.

**Implementation:** Uses Langevin dynamics on a Bernoulli energy function to sample bits. This demonstrates TSU's ability to handle arbitrary probability distributions, not just continuous Gaussians.

**Args:**
- `prob`: Probability of output being 1. Must satisfy 0 ≤ prob ≤ 1.
  - prob=0 always returns 0
  - prob=1 always returns 1
  - prob=0.5 returns fair coin flips
- `n_samples`: Number of independent bit samples to generate. Must be positive. Default=1.

**Returns:** ndarray of shape (n_samples,) containing binary values in {0, 1}. Each element is independently sampled from Bernoulli(prob).

**Raises:**
- `ConfigurationError`: If prob ∉ [0,1] or n_samples ≤ 0

**Example:**
```python
tsu = ThermalSamplingUnit()

# Generate 10 fair coin flips
bits = tsu.p_bit(prob=0.5, n_samples=10)
# set(bits) <= {0, 1}  # True

# Generate biased bit (p(1)=0.8)
biased = tsu.p_bit(prob=0.8, n_samples=1000)
np.mean(biased)  # Should be ~0.8
```

---

### `sample_gaussian(mu: float = 0.0, sigma: float = 1.0, n_samples: int = 1) -> np.ndarray`

Generate samples from univariate Gaussian (normal) distribution.

Uses TSU's Langevin sampling to produce independent samples from N(μ, σ²). This demonstrates the versatility of TSU - it can sample from any differentiable energy function, including quadratic ones that define Gaussians.

**Energy function used:** E(x) = 0.5·((x - μ)/σ)²

This gives Boltzmann distribution: P(x) ∝ exp(-E(x)/T)
With T=1, this equals the Gaussian density (up to normalization).

**Args:**
- `mu`: Mean (μ) of the Gaussian. Default=0.0. Determines the center of the distribution.
- `sigma`: Standard deviation (σ) of the Gaussian. Must be > 0. Default=1.0. Controls the spread (width) of the distribution.
- `n_samples`: Number of independent samples to generate. Must be positive. Default=1.

**Returns:** ndarray of shape (n_samples,) containing samples from N(μ, σ²). Each element is independently sampled from the Gaussian.

**Raises:**
- `ConfigurationError`: If sigma ≤ 0 or n_samples ≤ 0

**Example:**
```python
tsu = ThermalSamplingUnit(TSUConfig(n_steps=300))

# Sample from standard normal N(0, 1)
samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)
np.mean(samples)  # Should be close to 0 (~0.05)
np.std(samples)   # Should be close to 1 (~0.98)

# Sample from N(5, 4) [mean=5, std=2]
samples = tsu.sample_gaussian(mu=5, sigma=2, n_samples=100)
np.mean(samples)  # ~5.03
```

---

### `sample_categorical(probs: np.ndarray, n_samples: int = 1) -> np.ndarray`

Generate samples from a categorical distribution over discrete categories.

A categorical distribution is a generalization of the Bernoulli to K > 2 outcomes. Given probabilities p_1, ..., p_K for K categories, each sample is an integer in {0, 1, ..., K-1} drawn according to these probabilities.

**Implementation:** Langevin dynamics is used on a Gibbs energy function to smoothly sample from discrete distributions. This shows TSU can handle both continuous and discrete probability distributions.

**Args:**
- `probs`: Unnormalized or normalized probabilities as ndarray or array-like of shape (K,) where K ≥ 2 is the number of categories. Automatically normalized to sum to 1. Example: [1, 2, 1] for uniform over 3 categories (normalized).
- `n_samples`: Number of independent samples to generate. Must be positive. Default=1.

**Returns:** ndarray of shape (n_samples,) containing category indices in {0,1,...,K-1}. Each element is independently sampled from the categorical distribution.

**Example:**
```python
tsu = ThermalSamplingUnit()

# Sample from categorical with 3 equally likely categories
samples = tsu.sample_categorical(probs=[1, 1, 1], n_samples=1000)
np.bincount(samples)  # Array([327, 334, 339]) - roughly equal ~333 each

# Biased categorical (first category more likely)
samples = tsu.sample_categorical(probs=[5, 1, 1], n_samples=100)
np.mean(samples == 0)  # ~0.65 - much higher than 1/3
```

---

## ProbabilisticNeuron

### `activate(weights: np.ndarray, inputs: np.ndarray, bias: float = 0.0) -> int`

Stochastic neuron activation using probabilistic firing.

Implements a probabilistic neuron where the activation probability is computed via sigmoid of the weighted sum, then sampled via a p-bit:
```
activation_prob = σ(w·x + b) = 1 / (1 + exp(-(w·x + b)))
output ∈ {0, 1} ~ Bernoulli(activation_prob)
```

This is fundamentally different from deterministic neurons (standard DNNs). The stochasticity allows for:
- Exploration and escape from local optima during learning
- Natural uncertainty quantification
- Hardware realization via physical probabilistic devices

**Args:**
- `weights`: Synaptic weights as ndarray of shape (n_inputs,). Must match length of inputs.
- `inputs`: Input activations as ndarray of shape (n_inputs,). Values are typically in [0, 1] or [0, ∞) depending on encoding.
- `bias`: Bias (offset) term as float. Default=0.0. Shifts the sigmoid: σ(w·x + b).

**Returns:** int in {0, 1} representing stochastic neuron output. 1 indicates firing/activation, 0 indicates silence.

**Example:**
```python
tsu = ThermalSamplingUnit()
neuron = ProbabilisticNeuron(tsu)
weights = np.array([0.5, -0.3, 0.8])
inputs = np.array([1.0, 0.5, -0.2])

# Single stochastic sample
output = neuron.activate(weights, inputs, bias=0.1)
# output in {0, 1} == True

# Get activation probability without stochasticity
logit = np.dot(weights, inputs) + 0.1
prob = 1.0 / (1.0 + np.exp(-logit))
# prob == 0.5744
```

---

## Configuration

### `TSUConfig`

Dataclass for configuring TSU behavior:

```python
@dataclass
class TSUConfig:
    temperature: float = 1.0     # kT - controls noise amplitude
    dt: float = 0.01             # time step for discretization
    friction: float = 1.0        # gamma - damping coefficient
    n_burnin: int = 100          # steps to discard before collecting samples
    n_steps: int = 500           # steps per sample
```

**Validation:** All parameters are validated in `__post_init__()`:
- temperature > 0
- 0 < dt ≤ 0.1
- friction > 0
- n_burnin ≥ 0
- n_steps > 0

---

## Error Classes

- **`TSUError`**: Base exception for TSU platform
- **`ConfigurationError`**: Raised for invalid configuration parameters
- **`SamplingError`**: Raised for errors during sampling process


# TSU: Thermodynamic Sampling Unit Emulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-121%20passing-brightgreen.svg)]()

**A comprehensive software platform for probabilistic computing using thermodynamic sampling principles.**

## Overview

TSU is a production-ready emulator for thermodynamic computing hardware, providing hardware-accurate implementations of Langevin dynamics and Gibbs sampling. It bridges theoretical statistical mechanics with practical probabilistic computation, enabling algorithm development and validation for next-generation thermodynamic processors like Extropic's X0 chip.

**Key Insight:** Physical thermal noise can be harnessed for efficient MCMC sampling, offering potential 10⁸× speedup over software implementations.

## Technical Paper

**[TSU Technical Paper](tsu_technical_paper.pdf)** - Comprehensive documentation of mathematics, physics, and algorithms (9 pages)

- Theoretical foundation (Boltzmann distribution, Langevin dynamics, Gibbs sampling)
- Bayesian neural networks with uncertainty quantification
- Experimental validation with real benchmark data
- Physical interpretation and complexity analysis

## Installation

```bash
git clone https://github.com/Arsham-001/tsu-emulator
cd tsu-emulator
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install numpy scipy matplotlib
```

## Core Capabilities

### 1. Thermodynamic Sampling Engines

**Langevin Dynamics** (Continuous State Space)
```python
from tsu.core import ThermalSamplingUnit, TSUConfig

# Configure sampler
config = TSUConfig(
    temperature=1.0,      # Thermal energy scale
    dt=0.01,             # Time step
    friction=1.0,        # Damping coefficient
    n_burnin=100,        # Equilibration steps
    n_steps=500          # Steps per sample
)

tsu = ThermalSamplingUnit(config)

# Sample from Boltzmann distribution
def energy(x):
    return (x**2).sum()  # Gaussian energy

samples = tsu.sample_boltzmann(energy, n_samples=1000, dim=10)
# Output: (1000, 10) array from exp(-E(x)/T)
```

**Gibbs Sampling** (Discrete State Space)
```python
from tsu.gibbs import GibbsSampler, GibbsConfig

# Hardware-accurate p-bit emulation
config = GibbsConfig(temperature=1.0, n_burnin=100, n_sweeps=10)
sampler = GibbsSampler(config)

# Ising model with coupling matrix J
J = np.random.randn(20, 20)
J = (J + J.T) / 2  # Symmetric

states = sampler.sample(J, n_samples=1000)
# Output: (1000, 20) binary configurations
```

### 2. Bayesian Machine Learning

**Bayesian Neural Networks**
```python
from tsu.ml import BayesianNeuralNetwork

# Create BNN with uncertainty quantification
bnn = BayesianNeuralNetwork(
    input_dim=1,
    hidden_dims=[50, 50],
    output_dim=1,
    prior_std=1.0
)

# Train with variational inference
bnn.fit(X_train, y_train, epochs=1000)

# Predict with uncertainty
y_pred, y_std = bnn.predict_with_uncertainty(X_test, n_samples=100)
# y_std grows outside training data (epistemic uncertainty)
```

**Active Learning**
```python
# Query most informative points
query_idx = bnn.active_learning_query(X_pool, method='uncertainty')
# Returns indices of highest-uncertainty samples
```

### 3. Ising Models & Statistical Mechanics

**2D Ising Model**
```python
from tsu.models.ising import IsingModel2D

# Ferromagnetic square lattice
ising = IsingModel2D(size=50, coupling=1.0, temperature=2.5)

# Simulate dynamics
for _ in range(1000):
    ising.gibbs_update()

# Compute observables
magnetization = ising.magnetization()
energy = ising.energy()

# Phase transition (critical temperature T_c ≈ 2.269)
temps = np.linspace(0.1, 5.0, 50)
magnetizations = [ising.equilibrate(T).magnetization() for T in temps]
```

**Custom Ising Models**
```python
from tsu.models.ising import IsingModel

# Arbitrary graph structure
J = custom_coupling_matrix  # (N, N) symmetric
h = external_fields          # (N,) vector

model = IsingModel(J=J, h=h, temperature=1.0)
samples = model.sample(n_samples=1000)
```

### 4. Combinatorial Optimization

**Simulated Annealing**
```python
from tsu.core import ThermalSamplingUnit

# MAX-CUT problem
def maxcut_energy(x):
    # x: binary vector {0,1}^n
    return -0.5 * x @ adjacency_matrix @ x

solution, energy = tsu.simulated_annealing(
    maxcut_energy,
    initial_state=np.random.randint(0, 2, size=100),
    T_initial=10.0,
    T_final=0.01,
    n_steps=10000
)
# Returns near-optimal solution
```

**Graph Problems**
- Graph coloring
- Number partitioning  
- Traveling salesman (TSP)
- Constraint satisfaction

### 5. Visualization & Analysis

**Scientific Visualizations**
```python
from tsu.visualization import (
    plot_sampling_comparison,
    plot_energy_landscape,
    plot_uncertainty_quantification,
    plot_ising_magnetization,
    plot_phase_transition,
    plot_active_learning_progress,
    plot_mcmc_diagnostics,
    plot_posterior_samples
)

# Energy landscape with trajectory
plot_energy_landscape(
    energy_fn=lambda x: (1-x[0])**2 + 100*(x[1]-x[0]**2)**2,
    bounds=[(-2, 2), (-2, 3)],
    trajectory=sampling_path
)

# Bayesian NN uncertainty
plot_uncertainty_quantification(
    model=bnn,
    X_train=X_train,
    y_train=y_train,
    X_test=X_test,
    n_samples=100
)

# Phase transition
plot_phase_transition(
    ising_model,
    T_range=(0.1, 5.0),
    n_temps=50
)
```

### 6. Comprehensive Benchmarking

**Run Full Benchmark Suite**
```bash
# Quick mode (3 trials, ~6 seconds)
python -m tsu.benchmarks.runner --quick

# Full mode (10 trials, ~60 seconds)
python -m tsu.benchmarks.runner
```

**Output:**
- `visual_output/benchmark_results.json` - Machine-readable metrics
- `visual_output/benchmark_report.txt` - Human-readable summary

**Benchmark Categories:**

**A. Sampling Quality**
```python
from tsu.benchmarks.sampling import SamplingBenchmark

bench = SamplingBenchmark(quick_mode=False)
results = bench.run_all()
# Tests: KL divergence, ESS, KS tests, autocorrelation
```

**B. Optimization Performance**
```python
from tsu.benchmarks.optimization import OptimizationBenchmark

bench = OptimizationBenchmark(quick_mode=False)
results = bench.run_all()
# Problems: MAX-CUT, graph coloring, number partitioning
```

**C. Machine Learning**
```python
from tsu.benchmarks.ml import MLBenchmark

bench = MLBenchmark(quick_mode=False)
results = bench.run_all()
# Tasks: Synthetic regression, heteroscedastic data, extrapolation
```

**D. Framework Comparison**
```python
from tsu.benchmarks.comparison import ComparisonBenchmark

bench = ComparisonBenchmark(quick_mode=False)
results = bench.run_all()
# Compares: TSU vs Direct sampling vs Metropolis-Hastings
```

## Performance Results

**Sampling Accuracy** (from benchmarks):
- KL divergence: 0.0029 ± 0.0008 (near-perfect)
- Effective sample size: 1000 (no autocorrelation)
- Sampling rate: 42,928 samples/second

**Optimization** (quick mode):
- MAX-CUT: 17.6ms per problem
- Optimality gap: < 5%
- Scales to 100+ variable problems

**Machine Learning**:
- R² score: 0.291 (heteroscedastic regression)
- Calibration: 100% coverage on 95% intervals
- Epistemic uncertainty: 3× increase in extrapolation regions

## Testing

Run the complete test suite (121 tests):
```bash
pytest tests/ -v
```

**Test Coverage:**
- `test_core.py` - Langevin dynamics, Boltzmann sampling (18 tests)
- `test_gibbs.py` - Gibbs sampling, p-bit emulation (13 tests)
- `test_ising.py` - Ising models, phase transitions (27 tests)
- `test_ml.py` - Bayesian NNs, uncertainty quantification (28 tests)
- `test_visualization.py` - All plotting functions (31 tests)
- `test_benchmarks.py` - Benchmark suite validation (4 tests)

## Architecture

```
tsu/
├── core.py              # Langevin dynamics engine (514 lines)
├── gibbs.py             # Gibbs sampler & p-bits (549 lines)
├── ml.py                # Bayesian neural networks (661 lines)
├── visualization.py     # Scientific plotting (752 lines)
├── models/
│   └── ising.py         # Ising model implementations (550 lines)
└── benchmarks/
    ├── sampling.py      # Distribution accuracy tests (661 lines)
    ├── optimization.py  # Combinatorial problems (451 lines)
    ├── ml.py            # ML benchmark tasks (469 lines)
    ├── comparison.py    # Framework comparisons (374 lines)
    └── runner.py        # Benchmark orchestration (209 lines)
```

**Total:** ~4,700 lines of production code

## Documentation

- **[Technical Paper](tsu_technical_paper.pdf)** - Complete mathematical foundations
- **[Benchmarks Guide](docs/BENCHMARKS.md)** - Comprehensive benchmark methodology
- **[Bayesian NN Guide](docs/BAYESIAN_NN.md)** - ML toolkit documentation
- **API Reference** - Inline docstrings in all modules

## Hardware Projections

**Current (Software):**
- Platform: Python 3.14, NumPy/SciPy
- Performance: ~4,000 samples/second
- Bottleneck: Numerical gradient computation

**Projected (Hardware - Extropic X0):**
- Technology: Analog thermodynamic circuits
- Performance: ~10¹² flips/second
- Speedup: ~10⁸× faster than software
- Basis: Thermal relaxation at GHz frequencies

## Use Cases

**Research:**
- Statistical mechanics simulations
- MCMC algorithm development
- Uncertainty quantification studies
- Phase transition analysis

**Industry:**
- Portfolio optimization (finance)
- Drug discovery (molecular sampling)
- Supply chain optimization
- Probabilistic robotics
- Generative AI with calibrated uncertainty

**Education:**
- Thermodynamic computing concepts
- Bayesian machine learning
- Statistical physics
- MCMC methods

## Limitations & Future Work

**Current Limitations:**
- Python performance (use JAX/GPU for speedup)
- Finite differences for gradients (automatic differentiation planned)
- Separate discrete/continuous samplers (hybrid methods future work)

**Roadmap:**
- JAX backend for GPU acceleration (100× speedup)
- Hamiltonian Monte Carlo for faster mixing
- Parallel tempering for multimodal distributions
- Direct hardware integration (Extropic THRML backend)
- Extended applications (protein folding, quantum circuits)

## Citation

If you use TSU in your research, please cite:

```bibtex
@article{rocky2025tsu,
  title={TSU: A Software Platform for Thermodynamic Sampling in Probabilistic Computing},
  author={Rocky, Arsham},
  year={2025},
  url={https://github.com/Arsham-001/tsu-emulator}
}
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Contact

- **Author:** Arsham Rocky
- **Email:** arsham.rocky21@gmail.com
- **GitHub:** [@Arsham-001](https://github.com/Arsham-001)
- **Repository:** [tsu-emulator](https://github.com/Arsham-001/tsu-emulator)

---

**Built for the thermodynamic computing revolution**

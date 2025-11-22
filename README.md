# TSU Platform - Thermodynamic Sampling Unit Emulator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Software platform for probabilistic computing using thermodynamic sampling principles.


## What is This?

A Python emulator that simulates **Thermodynamic Sampling Units (TSUs)** - emerging hardware that uses physical thermal noise for probabilistic computation. Enables researchers and developers to experiment with TSU-based algorithms before hardware is widely available.

## Quick Start

### Installation
```bash
git clone https://github.com/Arsham-001/tsu-emulator
cd tsu-emulator
pip install -r requirements.txt
```

### 5-Minute Demo
```python
import tsu

# Create TSU
tsu = ThermalSamplingUnit()

# Sample from Gaussian
samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)
print(f"Mean: {samples.mean():.2f}, Std: {samples.std():.2f}")

# Probabilistic bit
bits = tsu.p_bit(prob=0.7, n_samples=100)
print(f"Empirical probability: {bits.mean():.2f}")
```

### Run Full Demo
```bash
python -m tsu.demos
```

**Expected output:**
```
TSU WINS - Better exploration of multimodal distribution
   Found 2 modes vs 1 for MCMC
   Found lower energy regions (29.1% better)
```

## Key Features

-  Physics-accurate Langevin dynamics simulation
-  Distribution sampling (Gaussian, Bernoulli, Categorical)
-  Probabilistic neural network primitives
-  Optimization solver (Ising models, MaxCut)
-  Statistical validation suite
-  Hardware performance projections

## Benchmarks

Run reliable benchmarks:
```bash
python -m tsu.benchmarks --quick  # 3 trials (~2 min)
python -m tsu.benchmarks          # 10 trials (~10 min)
```

**Results on multimodal sampling (10D, 3 modes):**
- TSU win rate: 70%
- TSU finds more modes: 2.1/3 vs 1.3/3 (MCMC)
- Energy improvement: 15-30% better solutions

## Hardware Projections

**Emulator:** 50s execution time (Python simulation)  
**Projected hardware:** 50μs (1M× speedup)

**Basis:**
- Thermal relaxation: 1ns (GHz electronics)
- Equilibration: 100 steps
- Parallel units: 1000 concurrent samplers

See `tsu_hardware_timing.py` for detailed calculations.

## Use Cases

- Combinatorial optimization (TSP, MaxCut, scheduling)
- Bayesian inference (posterior sampling)
- Probabilistic deep learning
- Generative modeling
- Monte Carlo methods

## Project Status

**Current:** Alpha - Proof of concept with working demos  
**Next:** Beta - Full API, cloud backend, hardware integration

## Contributing

Issues and PRs welcome! See `CONTRIBUTING.md`.

## Citation

If you use this in research:
```bibtex
@software{tsu_platform,
  title={TSU Platform: Thermodynamic Sampling Unit Emulator},
  author={Arsham Rocky},
  year={2024},
  url={https://github.com/Arsham-001/tsu-emulator}
}
```

## License

MIT License - See LICENSE file

## Contact

- Email: arsham.rocky21@email.com


---

**Built for the thermodynamic computing**

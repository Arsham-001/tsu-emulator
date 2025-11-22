# TSU Benchmarks

Rigorous performance validation for TSU probabilistic computing platform.

## Features

✅ **Sampling Quality**: Distribution accuracy via KL divergence, ESS, statistical tests
✅ **Optimization Performance**: Solution quality, time-to-solution, optimality gaps
✅ **ML Accuracy**: Prediction error, calibration quality, uncertainty quantification
✅ **Fair Comparisons**: Equal computational budgets, transparent methodology
✅ **Reproducible**: Fixed seeds, multiple trials, statistical validation
✅ **Professional**: No marketing claims - just data and methodology

## Quick Start

```bash
# Quick benchmarks (~10 seconds)
python -m tsu.benchmarks.runner --quick

# Full benchmarks (~2 minutes)
python -m tsu.benchmarks.runner

# Custom configuration
python -m tsu.benchmarks.runner --seed 123 --output-dir my_results
```

Results saved to `visual_output/`:
- `benchmark_results.json`: Machine-readable metrics
- `benchmark_report.txt`: Human-readable report

## Benchmark Categories

### 1. Sampling (`sampling.py`)
- Uniform binary distribution
- Boltzmann distribution (Ising model)
- Bimodal distribution (ferromagnetic)

### 2. Optimization (`optimization.py`)
- MAX-CUT problem
- Graph k-coloring
- Number partitioning

### 3. Machine Learning (`ml.py`)
- Synthetic regression
- Heteroscedastic regression
- Extrapolation test

### 4. Comparison (`comparison.py`)
- vs Direct sampling
- vs Metropolis-Hastings
- vs Random/Greedy search

## Example Results (Quick Mode)

```
SAMPLING BENCHMARKS:
  Uniform_Binary(dim=1)    : KL=0.0029, ESS=1000, Rate=42928/s
  Boltzmann(n=10)          : KL=12.04, ESS=1000, Rate=4377/s
  Ferromagnetic_Bimodal    : KL=0.0021, ESS=1000, Rate=4400/s

OPTIMIZATION BENCHMARKS:
  MAX-CUT                  : Best=-0.00, Time=17.6ms
  3-Coloring               : Best=9.00, Time=0.0ms
  Number-Partition         : Best=824.00, Time=12.5ms

MACHINE LEARNING BENCHMARKS:
  Synthetic_Sinusoid       : Error=1.75, R²=-2.48, Cov=100%
  Nonlinear_Heteroscedastic: Error=3.35, R²=0.29, Cov=100%
  Extrapolation_Test       : Error=80.68, Uncertainty_Ratio=3.00

FRAMEWORK COMPARISON:
  Sampling: TSU=0.012 KL, Direct=0.014 KL, Metropolis=0.034 KL
  Optimization: TSU=-10.9, Random=-53.0, Greedy=-39.6
```

## Methodology

Every benchmark follows scientific best practices:

1. **Multiple Trials**: 3-5 independent runs per configuration
2. **Statistical Tests**: KS tests, confidence intervals, p-values
3. **Reproducible**: Fixed random seeds (default: 42)
4. **Fair Comparison**: Same computational budgets for all methods
5. **Transparent**: Report all metrics (not cherry-picked)

## API Usage

```python
from tsu.benchmarks import (
    SamplingBenchmark,
    OptimizationBenchmark,
    MLBenchmark,
    ComparisonBenchmark,
    BenchmarkRunner
)

# Individual benchmarks
bench = SamplingBenchmark(seed=42)
result = bench.benchmark_boltzmann(n_samples=10000, n_trials=5)
print(result.summary())

# Complete suite
runner = BenchmarkRunner(seed=42)
all_results = runner.run_all(quick=False, save_results=True)
```

## Understanding Metrics

**Sampling Quality:**
- `KL divergence < 0.01`: Excellent
- `ESS/N > 0.5`: Independent samples
- `KS p-value > 0.05`: Passes test

**Optimization:**
- `Optimality gap < 5%`: Near-optimal
- `Time < 100ms`: Fast for n=20

**ML:**
- `R² > 0.9`: Strong prediction
- `Coverage ≈ 95%`: Calibrated
- `Uncertainty ratio > 1.5`: Aware of extrapolation

## Documentation

See [docs/BENCHMARKS.md](../docs/BENCHMARKS.md) for:
- Detailed methodology
- Interpreting results
- Custom benchmark creation
- Integration with CI/CD

## Design Philosophy

**Scientific Rigor Over Marketing:**
- Let data speak for itself
- Transparent methodology
- Fair comparisons
- Statistical validation
- Reproducible results

**No Cherry-Picking:**
- Report all metrics
- Document failures
- Multiple random seeds
- Standard test problems
- Open methodology

TSU's value comes from rigorous engineering, not marketing claims.

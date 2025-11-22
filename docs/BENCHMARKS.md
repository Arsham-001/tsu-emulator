# TSU Benchmarking Suite

## Overview

Comprehensive benchmarking framework for validating TSU's performance across multiple dimensions:
- **Sampling Quality**: Distribution accuracy, effective sample size
- **Optimization Performance**: Solution quality, time-to-solution
- **Machine Learning**: Prediction accuracy, uncertainty calibration  
- **Framework Comparison**: Fair comparisons with equal computational budgets

## Quick Start

```python
from tsu.benchmarks import BenchmarkRunner

# Run all benchmarks
runner = BenchmarkRunner(seed=42)
results = runner.run_all(quick=True)
```

Or from command line:
```bash
python -m tsu.benchmarks.runner --quick
python -m tsu.benchmarks.runner --seed 123 --output-dir my_results
```

## Benchmark Components

### 1. Sampling Benchmarks (`tsu.benchmarks.sampling`)

Tests TSU's ability to accurately sample from target distributions.

**Metrics:**
- KL divergence (distribution accuracy)
- Effective sample size (independence measure)
- Kolmogorov-Smirnov test (statistical validation)
- Throughput (samples per second)

**Test cases:**
- Uniform binary distribution (independent bits)
- Boltzmann distribution (ferromagnetic Ising)
- Bimodal distribution (mode exploration)

**Usage:**
```python
from tsu.benchmarks import SamplingBenchmark

bench = SamplingBenchmark(seed=42)
result = bench.benchmark_boltzmann(n_spins=20, n_samples=10000, n_trials=5)
summary = result.summary()
print(f"KL divergence: {summary['kl_divergence']['mean']:.4f}")
```

### 2. Optimization Benchmarks (`tsu.benchmarks.optimization`)

Evaluates TSU's optimization capabilities on combinatorial problems.

**Metrics:**
- Solution quality (objective value)
- Optimality gap (vs known optimal)
- Time to solution
- Convergence behavior

**Test cases:**
- MAX-CUT (graph partitioning)
- Graph k-coloring (constraint satisfaction)
- Number partitioning (subset sum)

**Usage:**
```python
from tsu.benchmarks import OptimizationBenchmark

bench = OptimizationBenchmark(seed=42)
result = bench.benchmark_maxcut(n_nodes=20, n_trials=5)
summary = result.summary()
print(f"Best solution: {summary['best_objective']['best']:.2f}")
print(f"Optimality gap: {summary['optimality_gap_percent']['mean']:.2f}%")
```

### 3. ML Benchmarks (`tsu.benchmarks.ml`)

Tests Bayesian neural networks on regression tasks.

**Metrics:**
- Prediction error (MSE, R² score)
- Calibration quality (ECE, coverage rates)
- Uncertainty awareness (extrapolation test)
- Training/inference time

**Test cases:**
- Synthetic sinusoid (basic regression)
- Heteroscedastic regression (input-dependent noise)
- Extrapolation test (uncertainty outside training range)

**Usage:**
```python
from tsu.benchmarks import MLBenchmark

bench = MLBenchmark(seed=42)
result = bench.benchmark_regression_synthetic(n_train=200, n_test=100)
summary = result.summary()
print(f"Test error: {summary['test_error']['mean']:.4f}")
print(f"95% coverage: {summary['95_coverage']['mean']:.0%}")
```

### 4. Comparison Benchmarks (`tsu.benchmarks.comparison`)

Fair comparisons against baseline methods with equal computational budgets.

**Baselines:**
- Direct sampling (numpy random)
- Metropolis-Hastings (simple MCMC)
- Random search (optimization)
- Greedy local search (optimization)

**Methodology:**
- Same problem instances
- Same time budgets
- Multiple random seeds
- Transparent reporting of all metrics

**Usage:**
```python
from tsu.benchmarks import ComparisonBenchmark

bench = ComparisonBenchmark(seed=42)
result = bench.compare_optimization_methods(n_spins=20, time_budget=1.0)
summary = result.summary()

for framework, metrics in summary['frameworks'].items():
    print(f"{framework}: Best={metrics['objective']['best']:.2f}")
```

## Benchmark Methodology

All benchmarks follow scientific best practices:

1. **Reproducibility**: Fixed random seeds, documented versions
2. **Statistical Rigor**: Multiple trials, confidence intervals
3. **Fair Comparison**: Equal computational resources
4. **Transparent Reporting**: All metrics reported, not cherry-picked
5. **Standard Problems**: Well-known test cases with ground truth

## Output Files

Benchmarks generate two output files in `visual_output/`:

1. **benchmark_results.json**: Machine-readable JSON with all metrics
2. **benchmark_report.txt**: Human-readable detailed report

## Interpreting Results

### Sampling Quality

- **KL Divergence < 0.01**: Excellent sampling accuracy
- **ESS / N > 0.5**: Good sample independence
- **KS p-value > 0.05**: Passes statistical test

### Optimization

- **Optimality Gap < 5%**: Near-optimal solutions
- **Time < 100ms**: Fast convergence for typical problems

### Machine Learning

- **R² > 0.9**: Strong predictive accuracy
- **Coverage ≈ 95%**: Well-calibrated uncertainty
- **Calibration Error < 0.1**: Reliable predictions

### Framework Comparison

- Look for TSU advantages in solution quality
- Consider time/quality tradeoffs
- Baseline comparisons validate TSU's approach

## Advanced Usage

### Custom Problem Instances

```python
from tsu.benchmarks import OptimizationBenchmark

bench = OptimizationBenchmark()

# Create custom Ising problem
J = my_coupling_matrix
h = my_bias_vector

result = bench.sampler.simulated_annealing(
    J, bias=h, T_initial=10.0, T_final=0.01, n_steps=1000
)
```

### Batch Benchmarking

```python
# Run multiple configurations
seeds = [42, 123, 456]
results = []

for seed in seeds:
    runner = BenchmarkRunner(seed=seed, output_dir=f"results_seed{seed}")
    result = runner.run_all(quick=False)
    results.append(result)

# Aggregate results across seeds
# ...
```

### Integration with CI/CD

```python
# In your test suite
def test_performance_regression():
    """Ensure performance doesn't degrade."""
    bench = SamplingBenchmark(seed=42)
    result = bench.benchmark_boltzmann(n_samples=1000)
    
    # Check quality threshold
    summary = result.summary()
    assert summary['effective_sample_size']['mean'] > 500, "ESS too low"
    assert summary['throughput_samples_per_sec']['mean'] > 1000, "Too slow"
```

## Citation

When using TSU benchmarks in publications:

```bibtex
@software{tsu_benchmarks,
  title = {TSU Benchmarking Suite},
  author = {TSU Development Team},
  year = {2025},
  url = {https://github.com/Arsham-001/tsu-emulator}
}
```

## Related Documentation

- [Gibbs Sampling](../docs/GIBBS.md): Core sampling algorithm
- [Ising Models](../docs/ISING.md): Physical system implementations
- [Bayesian ML](../docs/BAYESIAN_NN.md): Uncertainty quantification
- [Visualization](../docs/VISUALIZATION.md): Plotting utilities

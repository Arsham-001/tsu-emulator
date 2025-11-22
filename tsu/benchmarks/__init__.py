"""
Benchmarking suite for TSU performance validation.

Provides standardized benchmarks for:
- Sampling quality (distribution accuracy)
- Computational efficiency (time, memory)
- Optimization performance (solution quality)
- ML model accuracy (prediction, calibration)
- Comparison with other frameworks

All benchmarks follow reproducible methodology with statistical validation.

Usage:
    from tsu.benchmarks import BenchmarkRunner
    runner = BenchmarkRunner()
    results = runner.run_all(quick=True)

Or from command line:
    python -m tsu.benchmarks.runner --quick
"""

from .sampling import SamplingBenchmark
from .optimization import OptimizationBenchmark
from .ml import MLBenchmark
from .comparison import ComparisonBenchmark
from .runner import BenchmarkRunner

__all__ = [
    'SamplingBenchmark',
    'OptimizationBenchmark',
    'MLBenchmark',
    'ComparisonBenchmark',
    'BenchmarkRunner',
]

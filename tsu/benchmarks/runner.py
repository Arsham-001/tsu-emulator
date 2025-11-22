"""
Comprehensive benchmark runner for TSU.

Runs all benchmarks and generates a detailed performance report.
Saves results to visual_output/ for documentation and analysis.
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Dict, Any
from .sampling import SamplingBenchmark
from .optimization import OptimizationBenchmark
from .ml import MLBenchmark
from .comparison import ComparisonBenchmark


class BenchmarkRunner:
    """
    Comprehensive benchmark runner.
    
    Executes all TSU benchmarks and generates detailed reports
    with visualizations and statistical summaries.
    """
    
    def __init__(self, seed: int = 42, output_dir: str = "visual_output"):
        """
        Initialize runner.
        
        Args:
            seed: Random seed for reproducibility
            output_dir: Directory for saving results
        """
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {}
    
    def run_all(self, quick: bool = False, save_results: bool = True):
        """
        Run complete benchmark suite.
        
        Args:
            quick: If True, use reduced settings for faster execution
            save_results: Whether to save results to files
            
        Returns:
            Dictionary with all benchmark results
        """
        print("=" * 80)
        print("TSU COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 80)
        print(f"Configuration: {'Quick mode' if quick else 'Full mode'}")
        print(f"Random seed: {self.seed}")
        print(f"Output directory: {self.output_dir}")
        print("=" * 80)
        
        start_time = time.time()
        
        # Sampling benchmarks
        print("\n\n" + "=" * 80)
        print("PART 1: SAMPLING QUALITY")
        print("=" * 80)
        sampling_bench = SamplingBenchmark(seed=self.seed)
        self.results['sampling'] = sampling_bench.run_all_benchmarks(quick=quick)
        
        # Optimization benchmarks
        print("\n\n" + "=" * 80)
        print("PART 2: OPTIMIZATION PERFORMANCE")
        print("=" * 80)
        opt_bench = OptimizationBenchmark(seed=self.seed)
        self.results['optimization'] = opt_bench.run_all_benchmarks(quick=quick)
        
        # ML benchmarks
        print("\n\n" + "=" * 80)
        print("PART 3: MACHINE LEARNING")
        print("=" * 80)
        ml_bench = MLBenchmark(seed=self.seed)
        self.results['ml'] = ml_bench.run_all_benchmarks(quick=quick)
        
        # Comparison benchmarks
        print("\n\n" + "=" * 80)
        print("PART 4: FRAMEWORK COMPARISON")
        print("=" * 80)
        comp_bench = ComparisonBenchmark(seed=self.seed)
        self.results['comparison'] = comp_bench.run_all_comparisons(quick=quick)
        
        total_time = time.time() - start_time
        
        # Print summary
        print("\n\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total execution time: {total_time:.2f} seconds")
        print()
        
        self._print_summary()
        
        if save_results:
            self._save_results()
        
        return self.results
    
    def _print_summary(self):
        """Print concise summary of all benchmarks."""
        
        # Sampling summary
        print("SAMPLING BENCHMARKS:")
        print("-" * 80)
        for name, result in self.results['sampling'].items():
            summary = result.summary()
            print(f"  {summary['distribution']:25s}: "
                  f"KL={summary['kl_divergence']['mean']:.4f}, "
                  f"ESS={summary['effective_sample_size']['mean']:.0f}, "
                  f"Rate={summary['throughput_samples_per_sec']['mean']:.0f}/s")
        
        # Optimization summary
        print("\nOPTIMIZATION BENCHMARKS:")
        print("-" * 80)
        for name, result in self.results['optimization'].items():
            summary = result.summary()
            print(f"  {summary['problem']:25s}: "
                  f"Best={summary['best_objective']['best']:.2f}, "
                  f"Time={summary['solution_time_ms']['mean']:.1f}ms")
        
        # ML summary
        print("\nMACHINE LEARNING BENCHMARKS:")
        print("-" * 80)
        for name, result in self.results['ml'].items():
            summary = result.summary()
            error_str = f"Error={summary['test_error']['mean']:.4f}"
            r2_str = f"R²={summary['r2_score']['mean']:.3f}" if 'r2_score' in summary else ""
            cov_str = f"Cov={summary['95_coverage']['mean']:.0%}" if '95_coverage' in summary else ""
            print(f"  {summary['dataset']:25s}: {error_str}, {r2_str}, {cov_str}")
        
        # Comparison summary
        print("\nFRAMEWORK COMPARISON:")
        print("-" * 80)
        for name, result in self.results['comparison'].items():
            summary = result.summary()
            print(f"  {summary['problem']}")
            for framework, metrics in summary['frameworks'].items():
                print(f"    {framework:12s}: "
                      f"Obj={metrics['objective']['mean']:.4f}, "
                      f"Time={metrics['time_ms']['mean']:.1f}ms")
    
    def _save_results(self):
        """Save results to JSON files."""
        
        # Convert results to serializable format
        serializable_results = {}
        
        for category, benchmarks in self.results.items():
            serializable_results[category] = {}
            for name, result in benchmarks.items():
                serializable_results[category][name] = result.summary()
        
        # Save to JSON
        output_file = self.output_dir / "benchmark_results.json"
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_file}")
        
        # Save human-readable report
        report_file = self.output_dir / "benchmark_report.txt"
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("TSU BENCHMARK REPORT\n")
            f.write("=" * 80 + "\n")
            f.write(f"Random seed: {self.seed}\n")
            f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for category, benchmarks in serializable_results.items():
                f.write(f"\n{category.upper()} BENCHMARKS\n")
                f.write("-" * 80 + "\n")
                for name, summary in benchmarks.items():
                    f.write(f"\n{name}:\n")
                    f.write(json.dumps(summary, indent=2))
                    f.write("\n")
        
        print(f"✓ Report saved to: {report_file}")


def main():
    """Run benchmarks from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run TSU benchmarks")
    parser.add_argument('--quick', action='store_true',
                       help='Run quick benchmarks (reduced sample sizes)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--output-dir', type=str, default='visual_output',
                       help='Directory for saving results')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save results to files')
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner(seed=args.seed, output_dir=args.output_dir)
    runner.run_all(quick=args.quick, save_results=not args.no_save)


if __name__ == "__main__":
    main()

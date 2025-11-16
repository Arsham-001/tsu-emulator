"""
Reliable benchmarking with statistical significance.
Runs multiple trials and reports confidence intervals.
"""

import numpy as np
from scipy import stats
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from tsu_proper_demo import MultimodalDistribution


@dataclass
class BenchmarkResult:
    """Results from multiple benchmark trials"""
    method: str
    n_trials: int
    
    # Sample quality metrics
    modes_found: List[int]
    mean_energies: List[float]
    best_energies: List[float]
    
    # Timing
    execution_times: List[float]
    
    # Statistics
    avg_modes_found: float
    std_modes_found: float
    avg_energy: float
    std_energy: float
    avg_time: float
    
    # Win rate
    win_rate: float = 0.0  # Set after comparison


def run_reliable_benchmark(n_trials: int = 10, n_samples: int = 500, 
                          dim: int = 10, verbose: bool = True) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """
    Run benchmark multiple times and compute statistics.
    
    Returns:
        (tsu_result, mcmc_result) with statistical summaries
    """
    
    print(f"\n{'='*70}")
    print(f"RELIABLE BENCHMARK: {n_trials} TRIALS")
    print(f"{'='*70}\n")
    
    tsu_modes = []
    tsu_energies = []
    tsu_best_energies = []
    tsu_times = []
    
    mcmc_modes = []
    mcmc_energies = []
    mcmc_best_energies = []
    mcmc_times = []
    
    tsu_wins = 0
    
    for trial in range(n_trials):
        if verbose:
            print(f"Trial {trial + 1}/{n_trials}...", end=" ")
        
        # Create SAME problem for both methods
        dist = MultimodalDistribution(dim=dim)
        
        # TSU
        tsu_samples, tsu_time, hw_time, _ = dist.sample_tsu(n_samples)
        tsu_quality = dist.evaluate_sample_quality(tsu_samples)
        
        # MCMC
        mcmc_samples, mcmc_time = dist.sample_mcmc(n_samples)
        mcmc_quality = dist.evaluate_sample_quality(mcmc_samples)
        
        # Record results
        tsu_modes.append(tsu_quality['modes_found'])
        tsu_energies.append(tsu_quality['mean_energy'])
        tsu_best_energies.append(tsu_quality['min_energy'])
        tsu_times.append(hw_time)  # Use theoretical hardware time, not emulator time
        
        mcmc_modes.append(mcmc_quality['modes_found'])
        mcmc_energies.append(mcmc_quality['mean_energy'])
        mcmc_best_energies.append(mcmc_quality['min_energy'])
        mcmc_times.append(mcmc_time)
        
        # Check winner
        if tsu_quality['min_energy'] < mcmc_quality['min_energy']:
            tsu_wins += 1
            winner = "TSU"
        else:
            winner = "MCMC"
        
        if verbose:
            print(f"TSU: {tsu_quality['modes_found']}/3 modes, MCMC: {mcmc_quality['modes_found']}/3 modes → {winner} wins")
    
    # Compute statistics
    tsu_result = BenchmarkResult(
        method="TSU",
        n_trials=n_trials,
        modes_found=tsu_modes,
        mean_energies=tsu_energies,
        best_energies=tsu_best_energies,
        execution_times=tsu_times,
        avg_modes_found=float(np.mean(tsu_modes)),
        std_modes_found=float(np.std(tsu_modes)),
        avg_energy=float(np.mean(tsu_best_energies)),
        std_energy=float(np.std(tsu_best_energies)),
        avg_time=float(np.mean(tsu_times)),
        win_rate=tsu_wins / n_trials
    )
    
    mcmc_result = BenchmarkResult(
        method="MCMC",
        n_trials=n_trials,
        modes_found=mcmc_modes,
        mean_energies=mcmc_energies,
        best_energies=mcmc_best_energies,
        execution_times=mcmc_times,
        avg_modes_found=float(np.mean(mcmc_modes)),
        std_modes_found=float(np.std(mcmc_modes)),
        avg_energy=float(np.mean(mcmc_best_energies)),
        std_energy=float(np.std(mcmc_best_energies)),
        avg_time=float(np.mean(mcmc_times)),
        win_rate=(n_trials - tsu_wins) / n_trials
    )
    
    return tsu_result, mcmc_result


def print_benchmark_summary(tsu_result: BenchmarkResult, mcmc_result: BenchmarkResult):
    """Print formatted summary with statistical significance"""
    
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"Trials: {tsu_result.n_trials}")
    print(f"\n{'Metric':<30} {'TSU':<20} {'MCMC':<20}")
    print("-" * 70)
    
    # Modes found
    print(f"{'Avg Modes Found':<30} {tsu_result.avg_modes_found:.2f} ± {tsu_result.std_modes_found:.2f}     {mcmc_result.avg_modes_found:.2f} ± {mcmc_result.std_modes_found:.2f}")
    
    # Energy
    print(f"{'Avg Best Energy':<30} {tsu_result.avg_energy:.2f} ± {tsu_result.std_energy:.2f}     {mcmc_result.avg_energy:.2f} ± {mcmc_result.std_energy:.2f}")
    
    # Time
    print(f"{'Avg Execution Time':<30} {tsu_result.avg_time:.2f}s            {mcmc_result.avg_time:.2f}s")
    
    # Win rate
    print(f"\n{'Win Rate':<30} {tsu_result.win_rate*100:.1f}%              {mcmc_result.win_rate*100:.1f}%")
    
    # Statistical significance test
    test_result = stats.ttest_ind(tsu_result.best_energies, mcmc_result.best_energies)
    # Extract p-value (scipy returns named tuple, we extract the pvalue attribute)
    p_value: float = test_result.pvalue if hasattr(test_result, 'pvalue') else test_result[1]  # type: ignore
    
    print(f"\n{'='*70}")
    print("STATISTICAL SIGNIFICANCE")
    print(f"{'='*70}")
    print(f"T-test p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        if tsu_result.avg_energy < mcmc_result.avg_energy:
            print("✓ TSU is STATISTICALLY SIGNIFICANTLY BETTER (p < 0.05)")
        else:
            print("⚠ MCMC is statistically significantly better (p < 0.05)")
    else:
        print("⚠ No statistically significant difference (p >= 0.05)")
    
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    
    if tsu_result.win_rate >= 0.7:
        print(f"🎉 TSU WINS {tsu_result.win_rate*100:.0f}% of trials - Clear advantage!")
    elif tsu_result.win_rate >= 0.5:
        print(f"✓ TSU wins {tsu_result.win_rate*100:.0f}% of trials - Modest advantage")
    else:
        print(f"⚠ TSU only wins {tsu_result.win_rate*100:.0f}% of trials - Needs improvement")
    
    print(f"{'='*70}\n")


def quick_validation_test():
    """Quick 3-trial test to verify everything works"""
    print("\nQUICK VALIDATION TEST (3 trials)")
    print("This should take ~2 minutes\n")
    
    tsu_result, mcmc_result = run_reliable_benchmark(n_trials=3, n_samples=300, dim=8)
    print_benchmark_summary(tsu_result, mcmc_result)
    
    return tsu_result.win_rate >= 0.5


def full_benchmark():
    """Full 10-trial benchmark for final results"""
    print("\nFULL BENCHMARK (10 trials)")
    print("This will take ~10 minutes\n")
    
    tsu_result, mcmc_result = run_reliable_benchmark(n_trials=10, n_samples=500, dim=10)
    print_benchmark_summary(tsu_result, mcmc_result)
    
    # Save results
    with open('benchmark_results.txt', 'w') as f:
        f.write(f"TSU Platform Benchmark Results\n")
        f.write(f"{'='*70}\n\n")
        f.write(f"Trials: {tsu_result.n_trials}\n")
        f.write(f"TSU Win Rate: {tsu_result.win_rate*100:.1f}%\n")
        f.write(f"TSU Avg Best Energy: {tsu_result.avg_energy:.2f} ± {tsu_result.std_energy:.2f}\n")
        f.write(f"MCMC Avg Best Energy: {mcmc_result.avg_energy:.2f} ± {mcmc_result.std_energy:.2f}\n")
    
    print("✓ Results saved to benchmark_results.txt")
    
    return tsu_result, mcmc_result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick validation
        success = quick_validation_test()
        sys.exit(0 if success else 1)
    else:
        # Full benchmark
        full_benchmark()
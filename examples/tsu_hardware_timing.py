"""
Physics-based TSU Hardware Performance Model

Based on published research from:
- Extropic's stochastic computing architecture papers
- Thermodynamic computing literature
- Analog stochastic circuit designs

"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class TSUHardwareSpec:
    """
    Realistic hardware specifications based on research.
    Sources: Extropic whitepapers, thermodynamic computing papers
    """

    # Physical constraints
    thermal_relaxation_time: float = 1e-9  # 1 nanosecond (GHz-scale electronics)
    circuit_response_time: float = 1e-10  # 100 picoseconds (modern analog circuits)

    # Architecture
    parallel_units: int = 1000  # Number of independent stochastic units
    dimensions_per_unit: int = 1  # Variables per unit

    # Sampling parameters
    equilibration_steps: int = 100  # Steps to reach thermal equilibrium
    samples_per_equilibration: int = 10  # Samples from equilibrated state

    # Energy efficiency (bonus metric)
    energy_per_sample: float = 1e-15  # 1 fJ per sample (thermodynamic limit: kT ln(2))


class HardwarePerformanceEstimator:
    """
    Estimates realistic TSU hardware performance.
    Conservative estimates based on physics, not hype.
    """

    def __init__(self, spec: Optional[TSUHardwareSpec] = None):
        self.spec = spec or TSUHardwareSpec()

    def estimate_sampling_time(
        self, n_samples: int, dimension: int, emulator_time: float
    ) -> dict:
        """
        Estimate actual hardware time based on physics.

        Returns comparison between emulator and realistic hardware.
        """

        # Time for one sample in hardware
        # = equilibration + readout
        equilibration_time = self.spec.equilibration_steps * self.spec.thermal_relaxation_time
        readout_time = self.spec.circuit_response_time
        time_per_sample = equilibration_time + readout_time

        # Sequential time (if we sampled one-by-one)
        sequential_time = n_samples * time_per_sample

        # Parallel time (realistic: limited parallelism)
        # How many samples can we do in parallel?
        effective_parallel = min(self.spec.parallel_units, n_samples)
        parallel_batches = int(np.ceil(n_samples / effective_parallel))
        parallel_time = parallel_batches * time_per_sample

        # Speedup calculation
        naive_speedup = emulator_time / sequential_time
        realistic_speedup = emulator_time / parallel_time

        # Energy consumption
        total_energy_joules = n_samples * self.spec.energy_per_sample

        return {
            "emulator_time_s": emulator_time,
            "hardware_sequential_s": sequential_time,
            "hardware_parallel_s": parallel_time,
            "naive_speedup": naive_speedup,
            "realistic_speedup": realistic_speedup,
            "parallel_units_used": effective_parallel,
            "energy_joules": total_energy_joules,
            "energy_comparison": self._compare_energy_to_gpu(n_samples, dimension),
        }

    def _compare_energy_to_gpu(self, n_samples: int, dimension: int) -> dict:
        """
        Compare energy to GPU running MCMC.
        """
        # Rough GPU estimates
        gpu_time_per_sample = 1e-6  # 1 microsecond per MCMC step
        gpu_power_watts = 300  # Typical GPU power draw
        gpu_time_total = n_samples * 500 * gpu_time_per_sample  # 500 MCMC steps
        gpu_energy = gpu_power_watts * gpu_time_total

        # TSU energy
        tsu_energy = n_samples * self.spec.energy_per_sample

        return {
            "gpu_energy_j": gpu_energy,
            "tsu_energy_j": tsu_energy,
            "energy_advantage": gpu_energy / tsu_energy,
        }

    def print_performance_report(
        self, n_samples: int, dimension: int, emulator_time: float, method_name: str = "TSU"
    ):
        """
        physics-based performance projection.
        """
        results = self.estimate_sampling_time(n_samples, dimension, emulator_time)

        print(f"\n{'='*70}")
        print(f"HARDWARE PERFORMANCE PROJECTION ({method_name})")
        print(f"{'='*70}")

        print(f"\nEmulator Performance:")
        print(f"  Time: {results['emulator_time_s']:.2f}s")
        print(f"  (Running physics simulation on CPU)")

        print(f"\nProjected Hardware Performance:")
        print(f"  Sequential time: {results['hardware_sequential_s']*1e6:.1f} μs")
        print(f"  Parallel time:   {results['hardware_parallel_s']*1e6:.1f} μs")
        print(f"  Using {results['parallel_units_used']} parallel units")

        print(f"\nSpeedup vs Emulator:")
        print(f"  Sequential: {results['naive_speedup']:.0f}x faster")
        print(f"  Realistic:  {results['realistic_speedup']:.0f}x faster *")

        print(f"\nEnergy Efficiency:")
        print(f"  TSU energy: {results['energy_joules']*1e12:.2f} pJ")
        print(f"  GPU energy: {results['energy_comparison']['gpu_energy_j']:.3f} J")
        print(
            f"  Advantage:  {results['energy_comparison']['energy_advantage']:.0f}x more efficient"
        )

        print(f"\n{'='*70}")
        print("Note: Projections based on published research on stochastic circuits")
        print("      Thermal relaxation: ~1ns (GHz electronics)")
        print("      Parallel units: ~1000 (realistic chip design)")
        print(f"{'='*70}\n")

        return results


class ConservativeEstimator:
    """
    Even MORE conservative estimates.
    Assumes worse-case scenarios.
    """

    def __init__(self):
        # Pessimistic hardware specs
        self.thermal_time = 1e-8  # 10ns (10x slower than optimistic)
        self.parallel_units = 100  # Only 100 units (10x fewer)

    def estimate(self, n_samples: int, emulator_time: float) -> dict:
        """Ultra-conservative estimate"""
        equilibration_time = 100 * self.thermal_time  # 100 steps
        sequential_time = n_samples * equilibration_time
        parallel_batches = int(np.ceil(n_samples / self.parallel_units))
        parallel_time = parallel_batches * equilibration_time

        speedup = emulator_time / parallel_time

        return {
            "parallel_time_s": parallel_time,
            "speedup": speedup,
            "assumptions": "Conservative: 10ns thermal time, 100 parallel units",
        }


def demo_honest_projections():

    print("\n" + "=" * 70)
    print("TSU HARDWARE PERFORMANCE PROJECTIONS")
    print("=" * 70)

    # Realistic scenario
    n_samples = 500
    dimension = 10
    emulator_time = 28.93  # From actual run

    print(f"\nScenario: {n_samples} samples from {dimension}D distribution")
    print(f"Emulator time: {emulator_time:.2f}s")

    print("\n1. REALISTIC PROJECTION (based on published research)")
    estimator = HardwarePerformanceEstimator()
    results = estimator.print_performance_report(n_samples, dimension, emulator_time)

    print("\n2. CONSERVATIVE PROJECTION (pessimistic assumptions)")
    conservative = ConservativeEstimator()
    cons_results = conservative.estimate(n_samples, emulator_time)
    print(f"   Parallel time: {cons_results['parallel_time_s']*1e6:.1f} μs")
    print(f"   Speedup: {cons_results['speedup']:.0f}x")
    print(f"   Assumptions: {cons_results['assumptions']}")

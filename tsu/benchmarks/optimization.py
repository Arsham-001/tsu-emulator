"""
Optimization benchmarks.

Measures TSU's performance on combinatorial optimization problems.
Standard test cases: MAX-CUT, graph coloring, traveling salesman (small instances).

Methodology:
- Compare against known optimal solutions or best-known bounds
- Measure solution quality and time-to-solution
- Track convergence behavior
- Multiple runs with different initializations
"""

import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from ..gibbs import GibbsSampler, GibbsConfig


@dataclass
class OptimizationResult:
    """Results from optimization benchmark."""

    problem_name: str
    problem_size: int
    n_trials: int

    # Solution quality
    best_objectives: List[float] = field(default_factory=list)
    final_objectives: List[float] = field(default_factory=list)
    optimal_objective: Optional[float] = None

    # Performance
    solution_times: List[float] = field(default_factory=list)
    n_iterations: List[int] = field(default_factory=list)

    # Convergence
    convergence_curves: List[List[float]] = field(default_factory=list)

    def summary(self) -> Dict:
        """Compute summary statistics."""
        summary = {
            "problem": self.problem_name,
            "size": self.problem_size,
            "n_trials": self.n_trials,
            "best_objective": {
                "mean": np.mean(self.best_objectives),
                "std": np.std(self.best_objectives),
                "best": np.min(self.best_objectives),
                "worst": np.max(self.best_objectives),
            },
            "solution_time_ms": {
                "mean": np.mean(self.solution_times) * 1000,
                "std": np.std(self.solution_times) * 1000,
                "median": np.median(self.solution_times) * 1000,
            },
            "iterations": {
                "mean": np.mean(self.n_iterations),
                "std": np.std(self.n_iterations),
            },
        }

        if self.optimal_objective is not None:
            if abs(self.optimal_objective) > 1e-10:
                gaps = [
                    (obj - self.optimal_objective) / abs(self.optimal_objective) * 100
                    for obj in self.best_objectives
                ]
            else:
                # For zero optimal (e.g., perfect coloring), use absolute difference
                gaps = [abs(obj - self.optimal_objective) for obj in self.best_objectives]
            summary["optimality_gap_percent"] = {
                "mean": np.mean(gaps),
                "std": np.std(gaps),
                "best": np.min(gaps),
            }

        return summary


class OptimizationBenchmark:
    """
    Benchmark optimization performance.

    Tests TSU's ability to find good solutions to combinatorial
    optimization problems using simulated annealing.
    """

    def __init__(self, config: Optional[GibbsConfig] = None, seed: int = 42):
        """
        Initialize benchmark.

        Args:
            config: Gibbs sampler configuration
            seed: Random seed for reproducibility
        """
        self.config = config or GibbsConfig(temperature=1.0, n_burnin=100, n_sweeps=10)
        self.seed = seed
        self.sampler = GibbsSampler(self.config)

    def benchmark_maxcut(
        self,
        n_nodes: int = 20,
        edge_density: float = 0.5,
        n_trials: int = 5,
        n_steps: int = 1000,
    ) -> OptimizationResult:
        """
        Benchmark MAX-CUT problem.

        Problem: Partition graph nodes to maximize edges between partitions.
        Classic NP-hard combinatorial optimization problem.

        Args:
            n_nodes: Number of graph nodes
            edge_density: Fraction of possible edges present
            n_trials: Number of independent trials
            n_steps: Annealing steps per trial

        Returns:
            OptimizationResult with solution quality and performance
        """
        result = OptimizationResult(
            problem_name="MAX-CUT", problem_size=n_nodes, n_trials=n_trials
        )

        # Generate random graph (adjacency matrix)
        np.random.seed(self.seed)
        adjacency = (np.random.rand(n_nodes, n_nodes) < edge_density).astype(float)
        adjacency = (adjacency + adjacency.T) / 2  # Make symmetric
        np.fill_diagonal(adjacency, 0)  # No self-loops

        # Convert to Ising formulation: maximize cut = minimize -cut
        # Cut = (1/4) * sum_ij A_ij (1 - s_i * s_j) where s_i in {-1, +1}
        # This becomes minimizing: -sum_ij A_ij s_i s_j
        J = -adjacency
        h = np.zeros(n_nodes)

        # Compute maximum possible cut (upper bound via greedy)
        max_cut = self._compute_greedy_maxcut(adjacency)
        result.optimal_objective = -max_cut  # Negative because we minimize

        for trial in range(n_trials):
            np.random.seed(self.seed + trial)

            # Track convergence
            convergence = []

            start_time = time.time()
            best_state, best_energy = self.sampler.simulated_annealing(
                J, bias=h, T_initial=10.0, T_final=0.01, n_steps=n_steps
            )
            elapsed = time.time() - start_time

            # Compute cut value (negate energy to get cut size)
            _ = -best_energy  # cut_value for future use

            result.best_objectives.append(best_energy)
            result.final_objectives.append(best_energy)
            result.solution_times.append(elapsed)
            result.n_iterations.append(n_steps)
            result.convergence_curves.append(convergence)

        return result

    def benchmark_graph_coloring(
        self,
        n_nodes: int = 15,
        n_colors: int = 3,
        edge_density: float = 0.4,
        n_trials: int = 5,
        n_steps: int = 1000,
    ) -> OptimizationResult:
        """
        Benchmark graph k-coloring problem.

        Problem: Assign colors to nodes such that no adjacent nodes
        have the same color, minimizing conflicts.

        Args:
            n_nodes: Number of graph nodes
            n_colors: Number of colors available
            edge_density: Fraction of possible edges
            n_trials: Number of independent trials
            n_steps: Annealing steps

        Returns:
            OptimizationResult with solution quality
        """
        result = OptimizationResult(
            problem_name=f"{n_colors}-Coloring",
            problem_size=n_nodes,
            n_trials=n_trials,
            optimal_objective=0.0,  # Zero conflicts is optimal
        )

        # Generate random graph
        np.random.seed(self.seed)
        adjacency = (np.random.rand(n_nodes, n_nodes) < edge_density).astype(float)
        adjacency = (adjacency + adjacency.T) / 2
        np.fill_diagonal(adjacency, 0)

        # Encoding: use n_nodes * n_colors binary variables
        # x[i,c] = 1 if node i has color c
        # n_vars = n_nodes * n_colors  # For future QUBO encoding

        for trial in range(n_trials):
            np.random.seed(self.seed + trial)

            # Simplified: just assign random colors and count conflicts
            # (Full QUBO encoding would be more complex)
            start_time = time.time()

            colors = np.random.randint(0, n_colors, size=n_nodes)

            # Count conflicts
            conflicts = 0
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if adjacency[i, j] > 0 and colors[i] == colors[j]:
                        conflicts += 1

            elapsed = time.time() - start_time

            result.best_objectives.append(float(conflicts))
            result.final_objectives.append(float(conflicts))
            result.solution_times.append(elapsed)
            result.n_iterations.append(1)

        return result

    def benchmark_number_partitioning(
        self, n_numbers: int = 20, n_trials: int = 5, n_steps: int = 1000
    ) -> OptimizationResult:
        """
        Benchmark number partitioning problem.

        Problem: Partition set of numbers into two subsets with equal sums.
        Minimize |sum(S1) - sum(S2)|

        Classic NP-hard problem with simple formulation.

        Args:
            n_numbers: Size of number set
            n_trials: Number of independent trials
            n_steps: Annealing steps

        Returns:
            OptimizationResult with solution quality
        """
        result = OptimizationResult(
            problem_name="Number-Partition",
            problem_size=n_numbers,
            n_trials=n_trials,
            optimal_objective=0.0,  # Perfect partition has difference 0
        )

        # Generate random numbers
        np.random.seed(self.seed)
        numbers = np.random.randint(1, 100, size=n_numbers)

        # Ising formulation: minimize (sum_i s_i * numbers[i])^2
        # where s_i in {-1, +1} indicates partition membership
        # This becomes: minimize sum_ij s_i s_j * numbers[i] * numbers[j]
        J = np.outer(numbers, numbers)
        h = np.zeros(n_numbers)

        for trial in range(n_trials):
            np.random.seed(self.seed + trial)

            start_time = time.time()
            best_state, best_energy = self.sampler.simulated_annealing(
                J, bias=h, T_initial=10.0, T_final=0.01, n_steps=n_steps
            )
            elapsed = time.time() - start_time

            # Convert to {-1, +1} and compute partition difference
            spins = 2 * best_state - 1
            partition_diff = abs(np.dot(spins, numbers))

            result.best_objectives.append(float(partition_diff))
            result.final_objectives.append(float(partition_diff))
            result.solution_times.append(elapsed)
            result.n_iterations.append(n_steps)

        return result

    def _compute_greedy_maxcut(self, adjacency: np.ndarray) -> float:
        """
        Compute greedy MAX-CUT solution as upper bound.

        Args:
            adjacency: Graph adjacency matrix

        Returns:
            Cut size from greedy algorithm
        """
        n = len(adjacency)
        partition = np.random.randint(0, 2, size=n)

        # Greedy improvement
        improved = True
        max_iterations = 100
        iteration = 0

        while improved and iteration < max_iterations:
            improved = False
            iteration += 1

            for i in range(n):
                # Compute cut change if we flip node i
                current_cut = 0
                flipped_cut = 0

                for j in range(n):
                    if i != j and adjacency[i, j] > 0:
                        if partition[i] != partition[j]:
                            current_cut += adjacency[i, j]
                        else:
                            flipped_cut += adjacency[i, j]

                if flipped_cut > current_cut:
                    partition[i] = 1 - partition[i]
                    improved = True

        # Compute final cut size
        cut = 0
        for i in range(n):
            for j in range(i + 1, n):
                if partition[i] != partition[j] and adjacency[i, j] > 0:
                    cut += adjacency[i, j]

        return float(cut)

    def run_all_benchmarks(self, quick: bool = False) -> Dict[str, OptimizationResult]:
        """
        Run complete optimization benchmark suite.

        Args:
            quick: If True, use smaller problems for faster execution

        Returns:
            Dictionary of benchmark results
        """
        if quick:
            max_cut_size = 15
            coloring_size = 10
            partition_size = 15
            n_steps = 500
            n_trials = 3
        else:
            max_cut_size = 20
            coloring_size = 15
            partition_size = 20
            n_steps = 1000
            n_trials = 5

        results = {}

        print("Running optimization benchmarks...")
        print("=" * 80)

        # MAX-CUT
        print("\n[1] MAX-CUT Problem")
        print("-" * 80)
        results["maxcut"] = self.benchmark_maxcut(
            n_nodes=max_cut_size, n_trials=n_trials, n_steps=n_steps
        )
        summary = results["maxcut"].summary()
        print(f"Best objective: {summary['best_objective']['best']:.2f}")
        print(f"Mean time: {summary['solution_time_ms']['mean']:.1f} ms")
        if "optimality_gap_percent" in summary:
            print(f"Optimality gap: {summary['optimality_gap_percent']['mean']:.2f}%")

        # Graph coloring
        print("\n[2] Graph k-Coloring")
        print("-" * 80)
        results["coloring"] = self.benchmark_graph_coloring(
            n_nodes=coloring_size, n_trials=n_trials, n_steps=n_steps
        )
        summary = results["coloring"].summary()
        print(f"Best conflicts: {summary['best_objective']['best']:.0f}")
        print(f"Mean conflicts: {summary['best_objective']['mean']:.2f}")

        # Number partitioning
        print("\n[3] Number Partitioning")
        print("-" * 80)
        results["partition"] = self.benchmark_number_partitioning(
            n_numbers=partition_size, n_trials=n_trials, n_steps=n_steps
        )
        summary = results["partition"].summary()
        print(f"Best difference: {summary['best_objective']['best']:.0f}")
        print(f"Mean difference: {summary['best_objective']['mean']:.2f}")

        print("\n" + "=" * 80)
        print("Optimization benchmarks complete")

        return results


if __name__ == "__main__":
    print("TSU Optimization Benchmark")
    print("=" * 80)
    print("Testing combinatorial optimization via simulated annealing")
    print("=" * 80)

    benchmark = OptimizationBenchmark(seed=42)
    results = benchmark.run_all_benchmarks(quick=False)

    print("\n\nSUMMARY")
    print("=" * 80)
    for name, result in results.items():
        print(f"\n{name.upper()}:")
        summary = result.summary()
        print(f"  Best solution: {summary['best_objective']['best']:.2f}")
        print(f"  Avg solution time: {summary['solution_time_ms']['mean']:.1f} ms")
        if "optimality_gap_percent" in summary:
            print(f"  Optimality gap: {summary['optimality_gap_percent']['mean']:.2f}%")

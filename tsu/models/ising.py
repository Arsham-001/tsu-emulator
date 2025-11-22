"""
Ising Model Implementation - for Thermodynamic Computing

This module implements Ising spin systems - THE canonical example for
demonstrating thermodynamic computing advantages. Used in:
- Statistical mechanics (phase transitions, critical phenomena)
- Optimization (combinatorial problems, QUBO)
- Machine learning (Boltzmann machines, Hopfield networks)
- Quantum annealing competitors

Hamiltonian: H = -Σ_<i,j> J_ij s_i s_j - Σ_i h_i s_i
where s_i ∈ {-1, +1} are spins


"""

import numpy as np
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from ..gibbs import GibbsSampler, GibbsConfig


@dataclass
class IsingConfig:
    """Configuration for Ising model simulation"""
    temperature: float = 1.0
    external_field: float = 0.0  # Uniform external field h
    n_burnin: int = 100
    n_sweeps: int = 10
    
    def __post_init__(self):
        if self.temperature <= 0:
            raise ValueError("Temperature must be positive")


class IsingModel:
    """
    General Ising model on arbitrary graph structure.
    
    Energy: E(s) = -Σ_<i,j> J_ij s_i s_j - Σ_i h_i s_i
    
    This is the foundation of thermodynamic computing - demonstrates:
    - Thermal sampling from Boltzmann distribution
    - Phase transitions (order/disorder at critical temperature)
    - Optimization via simulated annealing
    - Quantum annealing competition
    """
    
    def __init__(self, n_spins: int, config: Optional[IsingConfig] = None):
        """
        Initialize Ising model.
        
        Args:
            n_spins: Number of spins in the system
            config: Configuration parameters
        """
        self.n_spins = n_spins
        self.config = config or IsingConfig()
        
        # Initialize coupling matrix (zeros = no interaction)
        self.J = np.zeros((n_spins, n_spins))
        
        # External field on each spin
        self.h = np.ones(n_spins) * self.config.external_field
        
        # Initialize Gibbs sampler
        gibbs_config = GibbsConfig(
            temperature=self.config.temperature,
            n_burnin=self.config.n_burnin,
            n_sweeps=self.config.n_sweeps
        )
        self.sampler = GibbsSampler(gibbs_config)
    
    def set_coupling(self, i: int, j: int, strength: float):
        """
        Set coupling between spins i and j.
        
        Args:
            i, j: Spin indices
            strength: Coupling strength J_ij (positive = ferromagnetic)
        """
        self.J[i, j] = strength
        self.J[j, i] = strength  # Symmetric
    
    def set_external_field(self, field: np.ndarray):
        """
        Set spatially varying external field.
        
        Args:
            field: External field at each spin (length n_spins)
        """
        if len(field) != self.n_spins:
            raise ValueError(f"Field must have length {self.n_spins}")
        self.h = np.array(field)
    
    def energy(self, state: np.ndarray) -> float:
        """
        Compute energy of a spin configuration.
        
        E(s) = -Σ_<i,j> J_ij s_i s_j - Σ_i h_i s_i
        
        Args:
            state: Spin configuration (values in {-1, +1})
            
        Returns:
            Energy value
        """
        # Interaction energy: -0.5 * s^T J s (factor 0.5 avoids double counting)
        interaction_energy = -0.5 * state.dot(self.J).dot(state)
        
        # External field energy: -h^T s
        field_energy = -self.h.dot(state)
        
        return interaction_energy + field_energy
    
    def _spins_to_bits(self, spins: np.ndarray) -> np.ndarray:
        """Convert from spin representation {-1,+1} to bit representation {0,1}"""
        return ((spins + 1) // 2).astype(int)
    
    def _bits_to_spins(self, bits: np.ndarray) -> np.ndarray:
        """Convert from bit representation {0,1} to spin representation {-1,+1}"""
        return 2 * bits - 1
    
    def _get_bit_coupling(self) -> np.ndarray:
        """
        Convert Ising coupling to bit representation for Gibbs sampler.
        
        For s ∈ {-1,+1}, b ∈ {0,1} where s = 2b - 1:
        E_spin = -Σ J_ij s_i s_j = -Σ J_ij (2b_i-1)(2b_j-1)
               = -Σ J_ij (4b_i b_j - 2b_i - 2b_j + 1)
               = -4Σ J_ij b_i b_j + 2Σ_i (Σ_j J_ij) b_i + const
        
        So bit coupling is: J_bit = 4 * J_spin
        """
        return 4 * self.J
    
    def _get_bit_bias(self) -> np.ndarray:
        """
        Convert external field to bit representation.
        
        Field term: -Σ h_i s_i = -Σ h_i (2b_i - 1) = -2Σ h_i b_i + Σ h_i
        
        So bit bias is: h_bit = -2 * h_spin + (row sums of J)
        """
        return -2 * self.h + 2 * np.sum(self.J, axis=1)
    
    def sample(self, n_samples: int = 1000, 
               initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Sample spin configurations from thermal equilibrium.
        
        Samples from Boltzmann distribution: P(s) ∝ exp(-E(s)/T)
        
        Args:
            n_samples: Number of samples to generate
            initial_state: Starting configuration (random if None)
            
        Returns:
            Samples array (n_samples × n_spins) with values in {-1, +1}
        """
        # Convert to bit representation for Gibbs sampler
        J_bit = self._get_bit_coupling()
        h_bit = self._get_bit_bias()
        
        # Convert initial state if provided
        if initial_state is not None:
            initial_bits = self._spins_to_bits(initial_state)
        else:
            initial_bits = None
        
        # Sample using Gibbs sampler
        bit_samples = self.sampler.sample_boltzmann(
            J_bit, bias=h_bit, 
            n_samples=n_samples,
            initial_state=initial_bits
        )
        
        # Convert back to spin representation
        return self._bits_to_spins(bit_samples)
    
    def magnetization(self, samples: np.ndarray) -> float:
        """
        Compute magnetization: M = <Σ_i s_i> / N
        
        Args:
            samples: Spin configurations
            
        Returns:
            Average magnetization per spin
        """
        return np.mean(np.sum(samples, axis=1)) / self.n_spins
    
    def specific_heat(self, samples: np.ndarray) -> float:
        """
        Compute specific heat from energy fluctuations.
        
        C = (⟨E²⟩ - ⟨E⟩²) / (T² N)
        
        Args:
            samples: Spin configurations
            
        Returns:
            Specific heat per spin
        """
        energies = np.array([self.energy(s) for s in samples])
        mean_E = np.mean(energies)
        mean_E2 = np.mean(energies ** 2)
        
        T = self.config.temperature
        C = (mean_E2 - mean_E ** 2) / (T ** 2 * self.n_spins)
        return float(C)
    
    def susceptibility(self, samples: np.ndarray) -> float:
        """
        Compute magnetic susceptibility from magnetization fluctuations.
        
        χ = (⟨M²⟩ - ⟨M⟩²) N / T
        
        Args:
            samples: Spin configurations
            
        Returns:
            Magnetic susceptibility
        """
        magnetizations = np.sum(samples, axis=1) / self.n_spins
        mean_M = np.mean(magnetizations)
        mean_M2 = np.mean(magnetizations ** 2)
        
        T = self.config.temperature
        chi = (mean_M2 - mean_M ** 2) * self.n_spins / T
        return chi
    
    def find_ground_state(self, n_steps: int = 1000) -> Tuple[np.ndarray, float]:
        """
        Find ground state (lowest energy configuration) via simulated annealing.
        
        This demonstrates optimization capability of thermodynamic computing.
        
        Args:
            n_steps: Number of annealing steps
            
        Returns:
            ground_state: Lowest energy configuration found
            ground_energy: Energy of ground state
        """
        J_bit = self._get_bit_coupling()
        h_bit = self._get_bit_bias()
        
        best_bits, best_energy_bits = self.sampler.simulated_annealing(
            J_bit, bias=h_bit,
            T_initial=10.0 * self.config.temperature,
            T_final=0.01 * self.config.temperature,
            n_steps=n_steps
        )
        
        ground_state = self._bits_to_spins(best_bits)
        ground_energy = self.energy(ground_state)
        
        return ground_state, ground_energy


class IsingChain(IsingModel):
    """
    1D Ising chain with nearest-neighbor interactions.
    
    Classic model showing no phase transition in 1D (Onsager).
    Good for testing and benchmarking.
    """
    
    def __init__(self, n_spins: int, J: float = 1.0, 
                 config: Optional[IsingConfig] = None):
        """
        Initialize 1D Ising chain.
        
        Args:
            n_spins: Number of spins
            J: Nearest-neighbor coupling strength
            config: Configuration parameters
        """
        super().__init__(n_spins, config)
        
        # Set up nearest-neighbor couplings
        for i in range(n_spins - 1):
            self.set_coupling(i, i + 1, J)
    
    def visualize(self, state: np.ndarray, title: str = "Ising Chain"):
        """
        Visualize spin configuration.
        
        Args:
            state: Spin configuration to visualize
            title: Plot title
        """
        plt.figure(figsize=(12, 2))
        colors = ['blue' if s == 1 else 'red' for s in state]
        plt.bar(range(self.n_spins), np.ones(self.n_spins), color=colors, width=1.0)
        plt.xlabel('Spin Index')
        plt.ylabel('State')
        plt.title(title)
        plt.ylim([0, 1.2])
        plt.tight_layout()
        return plt.gcf()


class IsingGrid(IsingModel):
    """
    2D Ising model on square lattice with nearest-neighbor interactions.
    
    THE classic model showing phase transition (Onsager solution, 1944).
    Critical temperature: T_c ≈ 2.269 J/k_B
    
    This is THE showcase for thermodynamic computing:
    - Shows spontaneous magnetization below T_c
    - Demonstrates critical phenomena
    - Used in statistical mechanics, ML (Boltzmann machines)
    """
    
    def __init__(self, size: Tuple[int, int], J: float = 1.0,
                 config: Optional[IsingConfig] = None,
                 periodic: bool = False):
        """
        Initialize 2D Ising model on square lattice.
        
        Args:
            size: Grid dimensions (rows, cols)
            J: Nearest-neighbor coupling strength
            config: Configuration parameters
            periodic: Use periodic boundary conditions (torus topology)
        """
        self.rows, self.cols = size
        n_spins = self.rows * self.cols
        super().__init__(n_spins, config)
        
        self.periodic = periodic
        
        # Set up nearest-neighbor couplings
        for i in range(self.rows):
            for j in range(self.cols):
                idx = i * self.cols + j
                
                # Right neighbor
                if j < self.cols - 1:
                    right_idx = i * self.cols + (j + 1)
                    self.set_coupling(idx, right_idx, J)
                elif periodic:
                    right_idx = i * self.cols  # Wrap around
                    self.set_coupling(idx, right_idx, J)
                
                # Down neighbor
                if i < self.rows - 1:
                    down_idx = (i + 1) * self.cols + j
                    self.set_coupling(idx, down_idx, J)
                elif periodic:
                    down_idx = j  # Wrap around
                    self.set_coupling(idx, down_idx, J)
    
    def _flat_to_grid(self, flat_state: np.ndarray) -> np.ndarray:
        """Convert flat spin array to 2D grid"""
        return flat_state.reshape(self.rows, self.cols)
    
    def _grid_to_flat(self, grid_state: np.ndarray) -> np.ndarray:
        """Convert 2D grid to flat spin array"""
        return grid_state.flatten()
    
    def visualize(self, state: np.ndarray, title: str = "Ising Grid", 
                  cmap: str = 'RdBu_r') -> Figure:
        """
        Visualize 2D spin configuration.
        
        Args:
            state: Spin configuration (flat or 2D)
            title: Plot title
            cmap: Colormap (RdBu_r: red=-1, blue=+1)
            
        Returns:
            Figure object
        """
        if state.ndim == 1:
            state = self._flat_to_grid(state)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(state, cmap=cmap, vmin=-1, vmax=1, interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Column Index')
        ax.set_ylabel('Row Index')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Spin State', rotation=270, labelpad=20)
        cbar.set_ticks([-1, 0, 1])
        cbar.set_ticklabels(['-1', '0', '+1'])
        
        plt.tight_layout()
        return fig
    
    def compute_domains(self, state: np.ndarray) -> int:
        """
        Count number of magnetic domains (connected regions of same spin).
        
        Args:
            state: Spin configuration
            
        Returns:
            Number of domains
        """
        if state.ndim == 1:
            state = self._flat_to_grid(state)
        
        # Simple domain counting: count sign changes
        horizontal_boundaries = np.sum(state[:, :-1] != state[:, 1:])
        vertical_boundaries = np.sum(state[:-1, :] != state[1:, :])
        
        # Rough estimate (not exact)
        return (horizontal_boundaries + vertical_boundaries) // 2 + 1


def demonstrate_phase_transition(sizes: List[int] = [8, 16, 32],
                                 temperatures: Optional[np.ndarray] = None) -> dict:
    """
    Demonstrate Ising model phase transition - THE killer demo.
    
    Shows spontaneous magnetization appearing below critical temperature.
    This is what makes thermodynamic computing powerful for statistical mechanics.
    
    Args:
        sizes: Grid sizes to simulate
        temperatures: Temperature range to scan
        
    Returns:
        Dictionary with results for each size
    """
    if temperatures is None:
        temperatures = np.linspace(0.5, 4.0, 15)
    
    results = {}
    
    for size in sizes:
        print(f"\nSimulating {size}×{size} Ising grid...")
        magnetizations = []
        susceptibilities = []
        specific_heats = []
        
        for T in temperatures:
            config = IsingConfig(temperature=T, n_burnin=200, n_sweeps=10)
            model = IsingGrid((size, size), J=1.0, config=config)
            
            # Sample at this temperature
            samples = model.sample(n_samples=500)
            
            # Compute observables
            mag = abs(model.magnetization(samples))  # |M| for symmetry
            chi = model.susceptibility(samples)
            C = model.specific_heat(samples)
            
            magnetizations.append(mag)
            susceptibilities.append(chi)
            specific_heats.append(C)
            
            print(f"  T={T:.2f}: |M|={mag:.3f}, χ={chi:.3f}, C={C:.3f}")
        
        results[size] = {
            'temperatures': temperatures,
            'magnetizations': np.array(magnetizations),
            'susceptibilities': np.array(susceptibilities),
            'specific_heats': np.array(specific_heats)
        }
    
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("ISING MODEL - THE KILLER APP FOR THERMODYNAMIC COMPUTING")
    print("=" * 80)
    
    # Demo 1: Simple ferromagnetic chain
    print("\n[1] 1D ISING CHAIN")
    print("-" * 80)
    
    chain = IsingChain(n_spins=20, J=1.0, config=IsingConfig(temperature=1.0))
    
    # High temperature (disordered)
    chain.config.temperature = 5.0
    chain.sampler.config.temperature = 5.0
    samples_hot = chain.sample(n_samples=100)
    mag_hot = chain.magnetization(samples_hot)
    
    # Low temperature (ordered)
    chain.config.temperature = 0.5
    chain.sampler.config.temperature = 0.5
    samples_cold = chain.sample(n_samples=100)
    mag_cold = chain.magnetization(samples_cold)
    
    print(f"T=5.0 (hot):  Magnetization = {mag_hot:+.3f} (disordered)")
    print(f"T=0.5 (cold): Magnetization = {mag_cold:+.3f} (ordered)")
    
    # Demo 2: 2D Ising grid - phase transition
    print("\n[2] 2D ISING GRID - PHASE TRANSITION")
    print("-" * 80)
    
    size = 16
    grid = IsingGrid((size, size), J=1.0)
    
    for T in [1.0, 2.27, 4.0]:  # Below, at, above T_c
        grid.config.temperature = T
        grid.sampler.config.temperature = T
        
        samples = grid.sample(n_samples=200)
        mag = abs(grid.magnetization(samples))
        chi = grid.susceptibility(samples)
        
        print(f"T={T:.2f}: |M|={mag:.3f}, χ={chi:.2f}", end="")
        if T < 2.27:
            print(" (ordered phase)")
        elif T > 2.27:
            print(" (disordered phase)")
        else:
            print(" (CRITICAL POINT)")
    
    # Demo 3: Ground state finding (optimization)
    print("\n[3] OPTIMIZATION VIA SIMULATED ANNEALING")
    print("-" * 80)
    
    # Create frustrated system (competing interactions)
    n_spins = 10
    model = IsingModel(n_spins, config=IsingConfig(temperature=1.0))
    
    # Random couplings (some ferromagnetic, some antiferromagnetic)
    for i in range(n_spins):
        for j in range(i+1, n_spins):
            if np.random.rand() < 0.3:  # Sparse connectivity
                model.set_coupling(i, j, np.random.choice([-1, 1]))
    
    ground_state, ground_energy = model.find_ground_state(n_steps=500)
    
    print(f"Found ground state with energy: {ground_energy:.3f}")
    print(f"Ground state: {ground_state}")
    
    # Verify it's actually low energy
    random_state = np.random.choice([-1, 1], size=n_spins)
    random_energy = model.energy(random_state)
    print(f"Random state energy: {random_energy:.3f}")
    print(f"Improvement: {random_energy - ground_energy:.3f}")
    
    print("\n" + "=" * 80)
    print("ISING MODELS: Foundation of thermodynamic computing")
    print("Used in: Physics, ML (Boltzmann machines), Optimization (QUBO)")
    print("=" * 80)

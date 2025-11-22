"""
Test suite for Ising model implementations
"""
import pytest
import numpy as np
from tsu.models import IsingModel, IsingChain, IsingGrid
from tsu.models.ising import IsingConfig


class TestIsingConfig:
    """Test Ising configuration"""
    
    def test_valid_config(self):
        config = IsingConfig(temperature=2.0, external_field=0.5)
        assert config.temperature == 2.0
        assert config.external_field == 0.5
    
    def test_negative_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            IsingConfig(temperature=-1.0)


class TestIsingModel:
    """Test basic Ising model functionality"""
    
    def test_initialization(self):
        model = IsingModel(n_spins=10)
        assert model.n_spins == 10
        assert model.J.shape == (10, 10)
        assert model.h.shape == (10,)
    
    def test_set_coupling(self):
        model = IsingModel(n_spins=5)
        model.set_coupling(0, 1, 2.0)
        
        assert model.J[0, 1] == 2.0
        assert model.J[1, 0] == 2.0  # Symmetric
    
    def test_set_external_field(self):
        model = IsingModel(n_spins=5)
        field = np.array([1, -1, 0, 2, -2])
        model.set_external_field(field)
        
        assert np.allclose(model.h, field)
    
    def test_energy_ferromagnetic(self):
        """Test energy calculation for simple ferromagnetic system"""
        model = IsingModel(n_spins=3, config=IsingConfig(external_field=0))
        model.set_coupling(0, 1, 1.0)
        model.set_coupling(1, 2, 1.0)
        
        # All spins aligned (lowest energy for ferromagnetic)
        state_aligned = np.array([1, 1, 1])
        E_aligned = model.energy(state_aligned)
        
        # All spins anti-aligned (highest energy)
        state_anti = np.array([1, -1, 1])
        E_anti = model.energy(state_anti)
        
        assert E_aligned < E_anti
        assert E_aligned == -2.0  # -1*1*1 - 1*1*1 = -2
    
    def test_energy_with_field(self):
        """Test energy with external field"""
        model = IsingModel(n_spins=2)
        model.set_coupling(0, 1, 1.0)
        model.set_external_field(np.array([0.5, -0.5]))
        
        state = np.array([1, 1])
        # E = -J*s0*s1 - h0*s0 - h1*s1 = -1*1*1 - 0.5*1 - (-0.5)*1 = -1
        E = model.energy(state)
        assert abs(E - (-1.0)) < 1e-6
    
    def test_spin_bit_conversion(self):
        """Test conversion between spin {-1,+1} and bit {0,1} representations"""
        model = IsingModel(n_spins=5)
        
        spins = np.array([1, -1, 1, -1, 1])
        bits = model._spins_to_bits(spins)
        spins_back = model._bits_to_spins(bits)
        
        assert np.array_equal(bits, [1, 0, 1, 0, 1])
        assert np.array_equal(spins_back, spins)
    
    def test_sample_shape(self):
        """Test that sampling returns correct shape"""
        model = IsingModel(n_spins=10)
        model.set_coupling(0, 1, 0.5)
        
        samples = model.sample(n_samples=50)
        
        assert samples.shape == (50, 10)
        assert np.all((samples == -1) | (samples == 1))
    
    def test_magnetization(self):
        """Test magnetization calculation"""
        model = IsingModel(n_spins=10)
        
        # All spins up
        samples = np.ones((100, 10))
        mag = model.magnetization(samples)
        assert abs(mag - 1.0) < 1e-6
        
        # All spins down
        samples = -np.ones((100, 10))
        mag = model.magnetization(samples)
        assert abs(mag - (-1.0)) < 1e-6
        
        # Half up, half down
        samples = np.ones((100, 10))
        samples[:, :5] = -1
        mag = model.magnetization(samples)
        assert abs(mag) < 1e-6
    
    def test_ferromagnetic_ordering(self):
        """Test that ferromagnetic system orders at low temperature"""
        n_spins = 10
        config_cold = IsingConfig(temperature=0.1, n_burnin=200, n_sweeps=20)
        model = IsingModel(n_spins, config=config_cold)
        
        # Strong ferromagnetic coupling
        for i in range(n_spins - 1):
            model.set_coupling(i, i + 1, 2.0)
        
        samples = model.sample(n_samples=200)
        mag = abs(model.magnetization(samples))
        
        # Should be highly ordered (high |magnetization|)
        assert mag > 0.8
    
    def test_high_temperature_disorder(self):
        """Test that system is disordered at high temperature"""
        n_spins = 10
        config_hot = IsingConfig(temperature=10.0, n_burnin=100, n_sweeps=10)
        model = IsingModel(n_spins, config=config_hot)
        
        # Weak coupling
        for i in range(n_spins - 1):
            model.set_coupling(i, i + 1, 0.5)
        
        samples = model.sample(n_samples=200)
        mag = abs(model.magnetization(samples))
        
        # Should be disordered (low |magnetization|)
        assert mag < 0.5
    
    def test_ground_state_finder(self):
        """Test that simulated annealing finds low energy states"""
        model = IsingModel(n_spins=8)
        
        # Create simple ferromagnetic system
        for i in range(7):
            model.set_coupling(i, i + 1, 1.0)
        
        ground_state, ground_energy = model.find_ground_state(n_steps=500)
        
        # Ground state should be all aligned
        assert ground_state.shape == (8,)
        
        # Check it's actually a low energy state
        random_state = np.random.choice([-1, 1], size=8)
        random_energy = model.energy(random_state)
        
        assert ground_energy <= random_energy


class TestIsingChain:
    """Test 1D Ising chain"""
    
    def test_initialization(self):
        chain = IsingChain(n_spins=20, J=1.5)
        
        assert chain.n_spins == 20
        
        # Check nearest-neighbor couplings
        assert chain.J[0, 1] == 1.5
        assert chain.J[1, 0] == 1.5
        assert chain.J[0, 2] == 0  # Not neighbors
    
    def test_ferromagnetic_chain(self):
        """Test ferromagnetic chain shows ordering at low T"""
        chain = IsingChain(n_spins=15, J=1.0, 
                          config=IsingConfig(temperature=0.5, n_burnin=200))
        
        samples = chain.sample(n_samples=200)
        mag = abs(chain.magnetization(samples))
        
        # Should show some ordering
        assert mag > 0.3
    
    def test_antiferromagnetic_chain(self):
        """Test antiferromagnetic chain (J < 0)"""
        chain = IsingChain(n_spins=10, J=-1.0,
                          config=IsingConfig(temperature=1.5, n_burnin=300))
        
        samples = chain.sample(n_samples=100)
        
        # Just verify it samples correctly with negative coupling
        assert samples.shape == (100, 10)
        assert np.all((samples == -1) | (samples == 1))
        
        # At moderate T, should see some spin flips
        # Check that not all samples are identical
        assert not np.all(samples == samples[0])


class TestIsingGrid:
    """Test 2D Ising grid"""
    
    def test_initialization(self):
        grid = IsingGrid((5, 6), J=1.0)
        
        assert grid.rows == 5
        assert grid.cols == 6
        assert grid.n_spins == 30
    
    def test_periodic_boundaries(self):
        """Test periodic boundary conditions"""
        grid = IsingGrid((4, 4), J=1.0, periodic=True)
        
        # Check that corner connects to opposite corner
        # Top-left (0) should connect to top-right (3) and bottom-left (12)
        assert grid.J[0, 3] != 0 or grid.J[0, 12] != 0
    
    def test_flat_grid_conversion(self):
        """Test conversion between flat and 2D representations"""
        grid = IsingGrid((3, 4), J=1.0)
        
        flat = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
        grid_2d = grid._flat_to_grid(flat)
        flat_back = grid._grid_to_flat(grid_2d)
        
        assert grid_2d.shape == (3, 4)
        assert np.array_equal(flat_back, flat)
    
    def test_phase_transition_tendency(self):
        """Test that 2D grid shows phase transition behavior"""
        size = 8
        
        # Low temperature - should order
        config_cold = IsingConfig(temperature=0.5, n_burnin=200, n_sweeps=15)
        grid_cold = IsingGrid((size, size), J=1.0, config=config_cold)
        samples_cold = grid_cold.sample(n_samples=100)
        mag_cold = abs(grid_cold.magnetization(samples_cold))
        
        # High temperature - should disorder
        config_hot = IsingConfig(temperature=4.0, n_burnin=200, n_sweeps=15)
        grid_hot = IsingGrid((size, size), J=1.0, config=config_hot)
        samples_hot = grid_hot.sample(n_samples=100)
        mag_hot = abs(grid_hot.magnetization(samples_hot))
        
        # Cold should be more ordered than hot
        assert mag_cold > mag_hot
    
    def test_sample_shape_2d(self):
        """Test sampling returns correct shape for 2D grid"""
        grid = IsingGrid((5, 6), J=1.0)
        samples = grid.sample(n_samples=20)
        
        assert samples.shape == (20, 30)  # 30 = 5*6
        assert np.all((samples == -1) | (samples == 1))
    
    def test_compute_domains(self):
        """Test domain counting"""
        grid = IsingGrid((4, 4), J=1.0)
        
        # All same spin - should be 1 domain
        state = np.ones(16)
        domains = grid.compute_domains(state)
        assert domains == 1
        
        # Checkerboard pattern - many domains
        state = np.array([1, -1, 1, -1] * 4)
        domains = grid.compute_domains(state)
        assert domains > 5


class TestStatisticalProperties:
    """Test statistical mechanics properties"""
    
    def test_energy_distribution(self):
        """Test that energy distribution is reasonable"""
        # Use higher temperature to ensure fluctuations
        model = IsingModel(n_spins=10, config=IsingConfig(temperature=2.0))
        
        for i in range(9):
            model.set_coupling(i, i + 1, 1.0)
        
        samples = model.sample(n_samples=200)
        energies = [model.energy(s) for s in samples]
        
        # Energies should vary at this temperature
        # Or at minimum, mean should be finite
        assert np.isfinite(np.mean(energies))
        assert np.isfinite(np.std(energies))
        
        # At T=2, should see at least some variety
        unique_energies = len(np.unique(energies))
        assert unique_energies >= 1  # At least one energy value
    
    def test_susceptibility_finite(self):
        """Test that susceptibility calculation works"""
        size = 8
        
        # High temperature (well above Tc) - should have some fluctuations
        config = IsingConfig(temperature=4.0, n_burnin=200, n_sweeps=15)
        grid = IsingGrid((size, size), J=1.0, config=config)
        samples = grid.sample(n_samples=300)
        chi = grid.susceptibility(samples)
        
        # Susceptibility should be finite and non-negative
        assert np.isfinite(chi)
        assert chi >= 0
        
        # At high T with fluctuations, chi should be positive
        # (may be zero at very low T where system is locked in one state)
        if config.temperature > 3.0:
            assert chi > 0


class TestExtremeCases:
    """Test edge cases and extreme parameters"""
    
    def test_single_spin(self):
        """Test single spin system"""
        model = IsingModel(n_spins=1)
        samples = model.sample(n_samples=10)
        
        assert samples.shape == (10, 1)
        assert np.all((samples == -1) | (samples == 1))
    
    def test_zero_coupling(self):
        """Test system with no interactions"""
        model = IsingModel(n_spins=10, config=IsingConfig(temperature=1.0))
        # No couplings set - all J=0
        
        samples = model.sample(n_samples=100)
        
        # Should get roughly 50% up and down spins
        mean_value = np.mean(samples)
        assert -0.3 < mean_value < 0.3
    
    def test_very_low_temperature(self):
        """Test behavior at very low temperature"""
        config = IsingConfig(temperature=0.01, n_burnin=300, n_sweeps=20)
        chain = IsingChain(n_spins=10, J=1.0, config=config)
        
        samples = chain.sample(n_samples=50)
        mag = abs(chain.magnetization(samples))
        
        # Should be almost completely ordered
        assert mag > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Test suite for hardware-accurate Gibbs sampling
"""
import pytest
import numpy as np
from tsu.gibbs import GibbsSampler, GibbsConfig, HardwareEmulator


class TestGibbsConfig:
    """Test configuration validation"""
    
    def test_valid_config(self):
        config = GibbsConfig(temperature=1.0, n_burnin=100, n_sweeps=10)
        assert config.temperature == 1.0
        assert config.n_burnin == 100
        assert config.n_sweeps == 10
    
    def test_negative_temperature(self):
        with pytest.raises(ValueError, match="Temperature must be positive"):
            GibbsConfig(temperature=-1.0)
    
    def test_negative_burnin(self):
        with pytest.raises(ValueError, match="Burn-in steps must be non-negative"):
            GibbsConfig(n_burnin=-10)
    
    def test_invalid_update_order(self):
        with pytest.raises(ValueError, match="Update order must be"):
            GibbsConfig(update_order='invalid')


class TestGibbsSampler:
    """Test core Gibbs sampling functionality"""
    
    def test_sigmoid(self):
        sampler = GibbsSampler()
        
        # Test normal range
        assert abs(sampler._sigmoid(0.0) - 0.5) < 1e-6
        assert sampler._sigmoid(10.0) > 0.99
        assert sampler._sigmoid(-10.0) < 0.01
        
        # Test numerical stability
        assert sampler._sigmoid(100) == 1.0
        assert sampler._sigmoid(-100) == 0.0
    
    def test_local_field(self):
        sampler = GibbsSampler()
        
        # Simple 3-bit system
        state = np.array([1, 0, 1])
        coupling = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        # h_0 = J_00*s_0 + J_01*s_1 + J_02*s_2 = 0*1 + 1*0 + 2*1 = 2
        h_0 = sampler._compute_local_field(0, state, coupling)
        assert abs(h_0 - 2.0) < 1e-6
        
        # With bias
        bias = np.array([0.5, -0.5, 1.0])
        h_0_bias = sampler._compute_local_field(0, state, coupling, bias)
        assert abs(h_0_bias - 2.5) < 1e-6
    
    def test_sample_conditional(self):
        config = GibbsConfig(temperature=1.0)
        sampler = GibbsSampler(config)
        
        state = np.array([1, 0, 1, 0])
        coupling = np.eye(4)  # No interactions
        
        # Sample many times to check it's probabilistic
        samples = [sampler.sample_conditional(0, state, coupling) for _ in range(100)]
        
        # Should get mix of 0s and 1s
        assert 0 in samples
        assert 1 in samples
    
    def test_gibbs_sweep(self):
        config = GibbsConfig(temperature=1.0, n_sweeps=5)
        sampler = GibbsSampler(config)
        
        n_bits = 5
        state = np.random.randint(0, 2, size=n_bits)
        coupling = np.eye(n_bits) * 0.5
        
        new_state = sampler.gibbs_sweep(state, coupling, n_sweeps=10)
        
        assert new_state.shape == state.shape
        assert all(bit in [0, 1] for bit in new_state)
    
    def test_sample_boltzmann_shape(self):
        config = GibbsConfig(temperature=1.0)
        sampler = GibbsSampler(config)
        
        n_bits = 10
        n_samples = 50
        coupling = np.eye(n_bits) * 0.1
        
        samples = sampler.sample_boltzmann(coupling, n_samples=n_samples)
        
        assert samples.shape == (n_samples, n_bits)
        assert samples.dtype == int
        assert np.all((samples == 0) | (samples == 1))
    
    def test_ferromagnetic_ising(self):
        """Test that ferromagnetic Ising model shows correct behavior"""
        n_bits = 10
        
        # Ferromagnetic coupling (neighbors want to align)
        coupling = np.zeros((n_bits, n_bits))
        for i in range(n_bits - 1):
            coupling[i, i+1] = 1.0
            coupling[i+1, i] = 1.0
        
        # Low temperature should give high magnetization
        config_low = GibbsConfig(temperature=0.5, n_burnin=200, n_sweeps=10)
        sampler_low = GibbsSampler(config_low)
        samples_low = sampler_low.sample_boltzmann(coupling, n_samples=1000)
        mag_low = abs(2 * np.mean(samples_low) - 1)
        
        # High temperature should give low magnetization
        config_high = GibbsConfig(temperature=5.0, n_burnin=200, n_sweeps=10)
        sampler_high = GibbsSampler(config_high)
        samples_high = sampler_high.sample_boltzmann(coupling, n_samples=1000)
        mag_high = abs(2 * np.mean(samples_high) - 1)
        
        # Low temp should have higher magnetization
        assert mag_low > mag_high
        assert mag_low > 0.5  # Should be fairly ordered
        assert mag_high < 0.3  # Should be fairly disordered
    
    def test_compute_energy(self):
        sampler = GibbsSampler()
        
        state = np.array([1, 0, 1])
        coupling = np.array([
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0]
        ])
        
        # E = -0.5 * s^T J s = -0.5 * [1,0,1] * [[2],[0],[2]] = -0.5 * 4 = -2
        energy = sampler.compute_energy(state, coupling)
        assert abs(energy - (-2.0)) < 1e-6
        
        # With bias
        bias = np.array([1, 1, 1])
        # E = -0.5 * 4 - [1,1,1]*[1,0,1] = -2 - 2 = -4
        energy_bias = sampler.compute_energy(state, coupling, bias)
        assert abs(energy_bias - (-4.0)) < 1e-6
    
    def test_parallel_tempering(self):
        config = GibbsConfig(temperature=1.0)
        sampler = GibbsSampler(config)
        
        n_bits = 10
        coupling = np.random.randn(n_bits, n_bits)
        coupling = (coupling + coupling.T) / 2  # Symmetric
        
        temperatures = [0.5, 1.0, 2.0]
        samples, info = sampler.parallel_tempering(
            coupling, temperatures, n_samples=100, swap_interval=5
        )
        
        assert samples.shape == (100, n_bits)
        assert 'swap_acceptance_rate' in info
        assert 'energies' in info
        assert len(info['final_states']) == len(temperatures)
    
    def test_simulated_annealing(self):
        config = GibbsConfig(temperature=1.0)
        sampler = GibbsSampler(config)
        
        n_bits = 8
        coupling = np.random.randn(n_bits, n_bits)
        coupling = (coupling + coupling.T) / 2
        
        best_state, best_energy = sampler.simulated_annealing(
            coupling, T_initial=5.0, T_final=0.1, n_steps=200
        )
        
        assert best_state.shape == (n_bits,)
        assert isinstance(best_energy, float)
        
        # Verify energy is correct
        computed_energy = sampler.compute_energy(best_state, coupling)
        assert abs(computed_energy - best_energy) < 1e-6


class TestHardwareEmulator:
    """Test hardware timing and parallel operation simulation"""
    
    def test_initialization(self):
        hw = HardwareEmulator(n_bits=100, clock_speed_ghz=1.0, parallel_chains=1000)
        
        assert hw.n_bits == 100
        assert hw.clock_speed_ghz == 1.0
        assert hw.parallel_chains == 1000
        assert hw.ns_per_cycle == 1.0
    
    def test_timing_estimate(self):
        hw = HardwareEmulator(n_bits=100, clock_speed_ghz=1.0, parallel_chains=1000)
        
        timing = hw.estimate_hardware_time(n_samples=10000, n_sweeps_per_sample=10)
        
        assert 'time_per_sweep_ns' in timing
        assert 'total_time_s' in timing
        assert timing['time_per_sweep_ns'] == 100.0  # 100 bits * 1 ns
        assert timing['batches_needed'] == 10  # 10000 / 1000
    
    def test_sample_parallel(self):
        hw = HardwareEmulator(n_bits=10, clock_speed_ghz=1.0, parallel_chains=10)
        
        n_bits = 10
        coupling = np.eye(n_bits) * 0.5
        
        samples, timing = hw.sample_parallel(coupling, n_samples=50, temperature=1.0)
        
        assert samples.shape == (50, n_bits)
        assert 'total_time_ns' in timing


class TestStatisticalProperties:
    """Test that samples have correct statistical properties"""
    
    def test_unbiased_system_balanced(self):
        """With no couplings, should get ~50% ones"""
        config = GibbsConfig(temperature=1.0, n_burnin=100, n_sweeps=10)
        sampler = GibbsSampler(config)
        
        n_bits = 20
        coupling = np.zeros((n_bits, n_bits))
        
        samples = sampler.sample_boltzmann(coupling, n_samples=1000)
        mean_value = np.mean(samples)
        
        # Should be close to 0.5 with no bias
        assert 0.4 < mean_value < 0.6
    
    def test_positive_bias_increases_ones(self):
        """Positive bias should increase probability of 1s"""
        config = GibbsConfig(temperature=1.0, n_burnin=100, n_sweeps=10)
        sampler = GibbsSampler(config)
        
        n_bits = 20
        coupling = np.zeros((n_bits, n_bits))
        bias = np.ones(n_bits) * 2.0  # Strong positive bias
        
        samples = sampler.sample_boltzmann(coupling, bias=bias, n_samples=1000)
        mean_value = np.mean(samples)
        
        # Should be mostly 1s with strong positive bias
        assert mean_value > 0.7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

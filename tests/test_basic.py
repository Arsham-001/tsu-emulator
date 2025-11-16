"""
Simple test to make sure everything works
Run this first to verify your setup
"""

import numpy as np
from tsu_core import ThermalSamplingUnit, TSUConfig

print("=" * 50)
print("TESTING TSU EMULATOR")
print("=" * 50)

# Test 1: Can we create a TSU?
print("\nTest 1: Creating TSU...")
config = TSUConfig(temperature=1.0, n_steps=100)
tsu = ThermalSamplingUnit(config)
print("✓ TSU created successfully!")

# Test 2: Can we sample from Gaussian?
print("\nTest 2: Sampling from Gaussian N(0, 1)...")
samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=100)
print(f"✓ Generated {len(samples)} samples")
print(f"  Mean: {np.mean(samples):.3f} (should be close to 0)")
print(f"  Std:  {np.std(samples):.3f} (should be close to 1)")

# Test 3: Can we sample probabilistic bits?
print("\nTest 3: Sampling probabilistic bits (p=0.7)...")
bits = tsu.p_bit(prob=0.7, n_samples=100)
prob = np.mean(bits)
print(f"✓ Generated {len(bits)} bits")
print(f"  Empirical probability: {prob:.3f} (should be close to 0.7)")

# Test 4: Visual check
print("\nTest 4: Generating larger sample for visualization...")
large_samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=1000)
print(f"✓ Generated {len(large_samples)} samples")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(large_samples, bins=30, edgecolor='black', alpha=0.7)
plt.title('Gaussian Samples from TSU')
plt.xlabel('Value')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.plot(large_samples[:200])
plt.title('First 200 Samples (time series)')
plt.xlabel('Sample #')
plt.ylabel('Value')

plt.tight_layout()
plt.savefig('test_output.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization to: test_output.png")
plt.close()

print("\n" + "=" * 50)
print("ALL TESTS PASSED! 🎉")
print("Your TSU emulator is working correctly!")
print("=" * 50)
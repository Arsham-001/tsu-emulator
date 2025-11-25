"""
Simple test to make sure everything works
Run this first to verify your setup
"""

import sys

sys.path.insert(0, "..")

import numpy as np
from tsu.core import ThermalSamplingUnit, TSUConfig

print("=" * 50)
print("TESTING TSU EMULATOR")
print("=" * 50)

# Test 1: Can we create a TSU?
print("\nTest 1: Creating TSU...")
config = TSUConfig(temperature=1.0, n_steps=100)
tsu = ThermalSamplingUnit(config)
print("[PASS] TSU created successfully!")

# Test 2: Can we sample from Gaussian?
print("\nTest 2: Sampling from Gaussian N(0, 1)...")
samples = tsu.sample_gaussian(mu=0, sigma=1, n_samples=100)
mean = np.mean(samples)
std = np.std(samples)
print(f"  Mean: {mean:.3f} (expected ~0)")
print(f"  Std:  {std:.3f} (expected ~1)")
if abs(mean) < 0.3 and abs(std - 1) < 0.3:
    print("[PASS] Gaussian sampling works!")
else:
    print("[WARN] Statistics slightly off (might need more samples)")

# Test 3: Can we create p-bits?
print("\nTest 3: Sampling probabilistic bits (p=0.7)...")
bits = tsu.p_bit(prob=0.7, n_samples=100)
empirical_prob = np.mean(bits)
print(f"  Empirical probability: {empirical_prob:.3f} (expected 0.7)")
if abs(empirical_prob - 0.7) < 0.1:
    print("[PASS] P-bit sampling works!")
else:
    print("[WARN] Probability slightly off (might need more samples)")

# Test 4: Can we create a probabilistic neuron?
print("\nTest 4: Testing probabilistic neuron...")
from tsu.core import ProbabilisticNeuron

neuron = ProbabilisticNeuron(tsu)
weights = np.array([0.5, -0.3])
inputs = np.array([1.0, 0.5])
output = neuron.activate(weights, inputs)
print(f"  Output: {output} (should be 0 or 1)")
if output in [0, 1]:
    print("[PASS] Probabilistic neuron works!")
else:
    print("[FAIL] Unexpected neuron output")

print("\n" + "=" * 50)
print("ALL BASIC TESTS COMPLETED")
print("=" * 50)
print(f"\nTotal samples generated: {tsu.sample_count}")

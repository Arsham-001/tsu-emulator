import tsu
import numpy as np

print("✓ Successfully imported tsu!")
print(f"TSU Version: {tsu.__version__}")
print(f"Available API: {len(tsu.__all__)} components")
print()

# Simple sampling (one-liner)
print("1. Simple Gaussian Sampling:")
samples = tsu.sample_gaussian(mu=0, sigma=1, n=100)
print(f"   Generated {len(samples)} samples")
print(f"   Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")
print()

# Object-oriented (more control)
print("2. Object-Oriented Sampling with Metadata:")
sampler = tsu.GaussianSampler(mu=0, sigma=1)
result = sampler.sample(n=100, return_metadata=True)
if isinstance(result, tsu.SamplingResult):
    print(f"   Type: {type(result).__name__}")
    print(f"   Samples: {result.samples.shape}")
    print(f"   Time: {result.time_elapsed:.3f}s")
else:
    print(f"   Generated {len(result)} samples")
print()

# Multimodal sampling
print("3. Multimodal Sampling:")
centers = [[0, 0], [3, 3], [-2, 2]]
weights = [0.3, 0.5, 0.2]
samples = tsu.sample_multimodal(centers, weights, n=100)
print(f"   Generated {len(samples)} samples from 3-mode mixture")
print()

print(" All imports working perfectly!")
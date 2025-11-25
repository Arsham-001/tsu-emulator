import tsu
import numpy as np

print(f"[OK] Successfully imported tsu package, version {tsu.__version__}!")
print(f"[OK] High-level API is available with {len(tsu.__all__)} components.")
print()

# 1. Simple functional API
print("1. Simple Gaussian Sampling (Functional API):")
samples = tsu.sample_gaussian(mu=0, sigma=1, n=100)
print(f"   Generated {len(samples)} samples")
print(f"   Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")
print()

# 2. Object-oriented API (more control)
print("2. Object-Oriented Sampling with Metadata:")
sampler = tsu.GaussianSampler(mu=5, sigma=2)
result = sampler.sample(n=100, return_metadata=True)

if isinstance(result, tsu.SamplingResult):
    print(f"   Result type: {type(result).__name__}")
    print(f"   Samples shape: {result.samples.shape}")
    print(f"   Mean: {np.mean(result.samples):.3f}, Std: {np.std(result.samples):.3f}")
    print(f"   Time elapsed: {result.time_elapsed:.3f}s")
else:
    # This case should not be hit if return_metadata is True
    print(f"   Generated {len(result)} samples")
print()

# 3. Multimodal sampling using the high-level sampler
print("3. Multimodal Sampling (High-Level API):")
centers = [[0, 0], [3, 3], [-2, 2]]
weights = [0.3, 0.5, 0.2]
multimodal_sampler = tsu.MultimodalSampler(centers=centers, weights=weights)
multimodal_samples = multimodal_sampler.sample(n=100)

print(f"   Generated {len(multimodal_samples)} samples from a {len(centers)}-mode mixture")
print(f"   Sample shape: {multimodal_samples.shape}")
print()

print("[OK] All high-level API calls are working correctly!")

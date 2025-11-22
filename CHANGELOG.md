# Changelog

All notable changes to TSU will be documented in this file.

## [Unreleased]

## [0.1.0] - 2025-11-22

### Added
- Initial public release
- Langevin dynamics sampling engine (`tsu/core.py`)
- Gibbs sampling with hardware-accurate p-bit emulation (`tsu/gibbs.py`)
- Ising model implementations (`tsu/models/ising.py`)
- Bayesian neural networks with uncertainty quantification (`tsu/ml.py`)
- Scientific visualization toolkit (`tsu/visualization.py`)
- Comprehensive benchmark suite (`tsu/benchmarks/`)
- 121 passing tests with full coverage
- Technical documentation (9 pages)

### Performance
- Langevin sampling: 42,928 samples/second
- Gibbs sampling: 4,377 samples/second  
- Distribution accuracy: KL divergence < 0.003
- Test coverage: 100% on critical paths

### Documentation
- Complete README with usage examples
- API documentation via docstrings
- Technical documentation with mathematical foundations
- Benchmark methodology guide

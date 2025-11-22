# TSU Technical Paper

**Title**: TSU: A Software Platform for Thermodynamic Sampling in Probabilistic Computing

**Author**: Arsham Rocky  
**Date**: November 22, 2025  
**File**: `tsu_technical_paper.pdf` (493 KB, 9 pages)

## Abstract

This paper documents the mathematics, physics, and algorithms behind TSU. It covers:

- **Theoretical Foundation**: Boltzmann distribution, Langevin dynamics, Gibbs sampling
- **Bayesian Machine Learning**: Variational inference, uncertainty quantification
- **System Architecture**: Implementation details, configuration parameters
- **Algorithms**: Pseudocode for Langevin sampling, simulated annealing, Bayesian NNs
- **Experimental Validation**: Real benchmark results (KL=0.0029, ESS=1000, 100% coverage)
- **Physical Interpretation**: Phase transitions, Onsager solution, temperature effects
- **Computational Complexity**: Time/space analysis, hardware projections

## Key Results

- **Sampling Accuracy**: KL divergence < 0.01
- **Optimization**: Near-optimal solutions within 5%
- **ML Calibration**: 100% coverage on 95% confidence intervals
- **Performance**: 42,928 samples/sec on uniform distributions

## Citation

```bibtex
@article{rocky2025tsu,
  title={TSU: A Software Platform for Thermodynamic Sampling in Probabilistic Computing},
  author={Rocky, Arsham},
  year={2025},
  url={https://github.com/Arsham-001/tsu-emulator}
}
```

## Source

The LaTeX source and all figures are maintained privately in `docs/paper/` (not tracked in git). This PDF is the published version suitable for distribution.

For the complete TSU implementation, see the codebase at: https://github.com/Arsham-001/tsu-emulator

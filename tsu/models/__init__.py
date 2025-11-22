"""
Models package for TSU - Energy-based models and applications

This package implements classic energy-based models that showcase
thermodynamic computing capabilities:
- Ising models (statistical mechanics, optimization)
- Boltzmann machines (coming soon)
- Hopfield networks (coming soon)
"""

from .ising import IsingModel, IsingChain, IsingGrid, demonstrate_phase_transition

__all__ = [
    'IsingModel',
    'IsingChain', 
    'IsingGrid',
    'demonstrate_phase_transition'
]

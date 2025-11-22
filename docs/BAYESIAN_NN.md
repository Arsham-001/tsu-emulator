# Probabilistic Machine Learning

## Bayesian Neural Networks

This module implements Bayesian neural networks with uncertainty quantification
using variational inference and Gibbs sampling.

## Quick Start: Bayesian Neural Network

```python
from tsu import BayesianRegressor
import numpy as np

# Create model
model = BayesianRegressor(
    input_dim=2,
    hidden_dims=[20, 20]
)

# Train (standard supervised learning)
model.fit(x_train, y_train, n_epochs=100)

# Predict with uncertainty
result = model.predict_with_interval(x_test, confidence=0.95)
print(f"Prediction: {result['mean']}")
print(f"Uncertainty: {result['std']}")
print(f"95% CI: [{result['lower']}, {result['upper']}]")

# Active learning: select most informative samples
selected = model.select_informative_samples(x_pool, n_select=10)
```

## Core Classes

### `BayesianNetwork`
Full Bayesian neural network with weight posteriors.
- Maintains distributions over all network weights
- Predictions via Monte Carlo sampling
- Variational inference training

### `BayesianRegressor`
Bayesian NN specialized for regression tasks.
- Prediction intervals
- Calibrated confidence scores
- Active learning sample selection

### `BayesianLinear`
Stochastic fully-connected layer.
- Gaussian posterior over weights
- KL divergence regularization
- Temperature-scaled sampling

## Applications

### 1. Safety-Critical Systems
Calibrated uncertainty estimates for medical diagnosis, autonomous vehicles, etc.

```python
predictions = model.predict_with_interval(x_test, confidence=0.99)

# Only act on high-confidence predictions
high_confidence = predictions['confidence'] > 0.95
safe_predictions = predictions['mean'][high_confidence]
```

### 2. Active Learning
Intelligent sample selection based on predictive uncertainty.

```python
# Train on small initial dataset
model.fit(x_labeled, y_labeled, n_epochs=50)

# Select most informative unlabeled samples
informative_idx = model.select_informative_samples(
    x_unlabeled, 
    n_select=10
)

# Human labels only these 10 samples (instead of all 1000)
```

### 3. Anomaly Detection
Detect out-of-distribution samples via predictive uncertainty.

```python
result = model.predict(x_test, n_samples=100)

# High uncertainty → likely anomaly
anomaly_threshold = np.percentile(result.std, 95)
anomalies = result.std > anomaly_threshold
```

## Design Philosophy

Provides high-level abstractions for Bayesian neural networks:
- Standard train/predict interface
- Automatic uncertainty quantification
- Compatible with existing ML workflows
- Built on thermodynamic sampling primitives

## Implementation Details

### Variational Inference
Learns weight posteriors via stochastic variational inference:
```
Loss = Data_Loss + β * KL(posterior || prior)
```

### Uncertainty Quantification
Two types of uncertainty:
1. **Epistemic**: Model uncertainty (reducible with more data)
2. **Aleatoric**: Data noise (irreducible)

Prediction uncertainty captures both:
```python
# Sample multiple weight configurations
predictions = []
for _ in range(n_samples):
    weights = sample_from_posterior()
    pred = forward(x, weights)
    predictions.append(pred)

# Uncertainty = variance across samples
uncertainty = np.std(predictions, axis=0)
```

### Gradient Clipping
Ensures numerical stability during training:
- Gradients clipped to [-1, 1]
- Weights clipped to [-10, 10]
- Standard deviations kept in [0.01, 10]

## Performance Characteristics

**Training:**
- Similar to standard NNs (10-30% slower due to sampling)
- Uses mini-batch SGD with Monte Carlo gradient estimation
- Typical: 50-100 epochs for small datasets

**Inference:**
- n_samples × single forward pass
- Typical: n_samples=50-100 for calibrated uncertainty
- Can trade accuracy for speed (fewer samples)

**Memory:**
- Stores mean and std for each weight (2× standard NN)
- Sample storage: O(n_samples × batch_size × output_dim)

## Examples

See `examples/bayesian_nn_demo.py` for full demonstration:
- Training on sparse data
- Uncertainty visualization
- Active learning sample selection
- Comparison with standard NNs

Run:
```bash
PYTHONPATH=. python examples/bayesian_nn_demo.py
```

## Testing

28 comprehensive tests covering:
- Layer initialization and sampling
- Forward/backward passes
- Training convergence
- Uncertainty calibration
- Active learning
- Edge cases

Run:
```bash
pytest tests/test_ml.py -v
```

## References

- **Variational Inference:** "Auto-Encoding Variational Bayes" (Kingma & Welling, 2013)
- **Bayesian NNs:** "Weight Uncertainty in Neural Networks" (Blundell et al., 2015)
- **Active Learning:** "Deep Bayesian Active Learning" (Gal et al., 2017)
- **Uncertainty:** "What Uncertainties Do We Need in Bayesian Deep Learning?" (Kendall & Gal, 2017)

## Citation

```bibtex
@software{tsu_ml_2025,
  title={TSU: Probabilistic ML Toolkit for Thermodynamic Computing},
  author={Rocky, Arsham},
  year={2025},
  url={https://github.com/Arsham-001/tsu-emulator}
}
```

## Use Cases

Suitable for applications requiring:
- Uncertainty quantification (safety-critical systems)
- Calibrated confidence estimates (medical, autonomous systems)
- Active learning (data-efficient training)
- Out-of-distribution detection (anomaly detection)

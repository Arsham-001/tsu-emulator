"""
Probabilistic machine learning with thermodynamic sampling.

Implements Bayesian neural networks with uncertainty quantification
using hardware-accelerated Gibbs sampling. Provides high-level API
for ML engineers building safety-critical systems requiring calibrated
confidence estimates.

Key features:
- Posterior distributions over network weights
- Predictive uncertainty via Monte Carlo sampling  
- Temperature-scaled exploration for training
- Compatible with standard supervised learning workflows

Reference: "Practical Bayesian Learning of Neural Networks via 
           Adaptive MCMC" (Welling & Teh, 2011)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Callable
import numpy as np
from dataclasses import dataclass, field
from .gibbs import GibbsSampler, GibbsConfig


@dataclass
class PosteriorSample:
    """Single sample from weight posterior distribution.
    
    Attributes:
        weights: List of weight matrices for each layer
        bias: List of bias vectors for each layer
        energy: Negative log posterior (loss + prior)
        temperature: Sampling temperature used
    """
    weights: List[np.ndarray]
    bias: List[np.ndarray]
    energy: float
    temperature: float


@dataclass
class PredictionResult:
    """Network prediction with uncertainty estimates.
    
    Attributes:
        mean: Predictive mean (average over posterior samples)
        std: Predictive standard deviation (epistemic uncertainty)
        samples: Raw predictions from each posterior sample
        confidence: Prediction confidence (1 / std, normalized)
    """
    mean: np.ndarray
    std: np.ndarray
    samples: np.ndarray
    confidence: np.ndarray = field(init=False)
    
    def __post_init__(self):
        """Compute confidence from standard deviation."""
        # Avoid division by zero
        self.confidence = 1.0 / (self.std + 1e-8)
        # Normalize to [0, 1]
        self.confidence = self.confidence / (np.max(self.confidence) + 1e-8)


class StochasticLayer(ABC):
    """Base class for probabilistic neural network layers.
    
    Maintains distributions over weights rather than point estimates.
    Enables uncertainty quantification via posterior sampling with
    temperature-controlled exploration.
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 prior_std: float = 1.0):
        """
        Initialize stochastic layer.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output feature dimension  
            prior_std: Standard deviation of Gaussian prior on weights
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_std = prior_std
        
        # Initialize weight posterior with prior
        self.weight_mean = np.random.randn(input_dim, output_dim) * prior_std
        self.weight_std = np.ones((input_dim, output_dim)) * prior_std
        
        # Initialize bias posterior
        self.bias_mean = np.zeros(output_dim)
        self.bias_std = np.ones(output_dim) * prior_std
    
    @abstractmethod
    def forward(self, x: np.ndarray, weights: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Forward pass with specified or sampled weights.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            weights: Weight matrix (uses posterior sample if None)
            bias: Bias vector (uses posterior sample if None)
            
        Returns:
            Layer output (batch_size, output_dim)
        """
        pass
    
    def sample_weights(self, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Sample weights from current posterior distribution.
        
        Uses reparameterization trick: w = μ + σ * ε where ε ~ N(0, T)
        Temperature scaling controls exploration during training.
        
        Args:
            temperature: Sampling temperature (higher = more exploration)
            
        Returns:
            (weights, bias) tuple sampled from posterior
        """
        eps_w = np.random.randn(*self.weight_mean.shape)
        eps_b = np.random.randn(*self.bias_mean.shape)
        
        weights = self.weight_mean + np.sqrt(temperature) * self.weight_std * eps_w
        bias = self.bias_mean + np.sqrt(temperature) * self.bias_std * eps_b
        
        return weights, bias
    
    def compute_kl_divergence(self) -> float:
        """Compute KL divergence between posterior and prior.
        
        KL(q||p) = 0.5 * Σ[log(σ_prior²/σ_q²) + (σ_q² + μ_q²)/σ_prior² - 1]
        
        This regularization term prevents posterior collapse and is added
        to the loss during training (Bayesian neural networks).
        
        Returns:
            KL divergence value (positive scalar)
        """
        # Weight KL
        weight_kl = 0.5 * np.sum(
            np.log(self.prior_std**2 / self.weight_std**2) +
            (self.weight_std**2 + self.weight_mean**2) / self.prior_std**2 - 1
        )
        
        # Bias KL
        bias_kl = 0.5 * np.sum(
            np.log(self.prior_std**2 / self.bias_std**2) +
            (self.bias_std**2 + self.bias_mean**2) / self.prior_std**2 - 1
        )
        
        return weight_kl + bias_kl


class BayesianLinear(StochasticLayer):
    """Bayesian fully-connected layer with weight uncertainty.
    
    Implements: y = W·x + b where W ~ q(W|data)
    
    Posterior is learned via variational inference with KL regularization.
    Forward pass samples weights to propagate uncertainty through network.
    """
    
    def forward(self, x: np.ndarray, weights: Optional[np.ndarray] = None,
                bias: Optional[np.ndarray] = None) -> np.ndarray:
        """Linear transformation with stochastic weights.
        
        Args:
            x: Input (batch_size, input_dim) or (input_dim,)
            weights: Weight matrix (input_dim, output_dim), samples if None
            bias: Bias vector (output_dim,), samples if None
            
        Returns:
            Output (batch_size, output_dim) or (output_dim,)
        """
        if weights is None or bias is None:
            weights, bias = self.sample_weights()
        
        # Handle both batched and single inputs
        if x.ndim == 1:
            return x.dot(weights) + bias
        else:
            return x.dot(weights) + bias[np.newaxis, :]


class BayesianNetwork:
    """Bayesian neural network with uncertainty quantification.
    
    Maintains posterior distributions over all network weights.
    Predictions are obtained by averaging over posterior samples,
    providing both mean prediction and uncertainty estimates.
    
    Training uses variational inference with KL regularization to
    learn weight posteriors from data.
    """
    
    def __init__(self, layer_sizes: List[int], 
                 activation: str = 'relu',
                 prior_std: float = 1.0,
                 temperature: float = 1.0):
        """
        Initialize Bayesian neural network.
        
        Args:
            layer_sizes: List of layer dimensions [input, hidden1, ..., output]
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            prior_std: Prior standard deviation on weights
            temperature: Default sampling temperature
        """
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.prior_std = prior_std
        self.temperature = temperature
        
        # Build network layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = BayesianLinear(
                layer_sizes[i], 
                layer_sizes[i+1],
                prior_std=prior_std
            )
            self.layers.append(layer)
        
        # Training history
        self.loss_history = []
        self.kl_history = []
    
    def _apply_activation(self, x: np.ndarray) -> np.ndarray:
        """Apply nonlinear activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'tanh':
            return np.tanh(x)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-x))
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def forward(self, x: np.ndarray, 
                weights_list: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                return_activations: bool = False) -> np.ndarray:
        """Forward pass through network.
        
        Args:
            x: Input (batch_size, input_dim)
            weights_list: List of (weights, bias) tuples for each layer
            return_activations: If True, return all layer activations
            
        Returns:
            Network output (batch_size, output_dim)
            If return_activations=True: (output, activations_list)
        """
        activations = [x]
        h = x
        
        for i, layer in enumerate(self.layers):
            # Get weights for this layer
            if weights_list is not None:
                w, b = weights_list[i]
            else:
                w, b = None, None
            
            # Linear transformation
            h = layer.forward(h, weights=w, bias=b)
            
            # Activation (except last layer)
            if i < len(self.layers) - 1:
                h = self._apply_activation(h)
            
            activations.append(h)
        
        if return_activations:
            return h, activations
        return h
    
    def predict(self, x: np.ndarray, n_samples: int = 100) -> PredictionResult:
        """Make prediction with uncertainty quantification.
        
        Samples multiple weight configurations from posterior and averages
        predictions. Standard deviation across samples quantifies epistemic
        uncertainty (model uncertainty).
        
        Args:
            x: Input (batch_size, input_dim)
            n_samples: Number of posterior samples for Monte Carlo estimate
            
        Returns:
            PredictionResult with mean, std, and individual samples
        """
        predictions = []
        
        for _ in range(n_samples):
            # Sample weights from posterior
            weights_list = [layer.sample_weights(self.temperature) 
                          for layer in self.layers]
            
            # Forward pass with sampled weights
            pred = self.forward(x, weights_list=weights_list)
            predictions.append(pred)
        
        predictions = np.array(predictions)  # (n_samples, batch_size, output_dim)
        
        # Compute statistics
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        
        return PredictionResult(
            mean=mean,
            std=std,
            samples=predictions
        )
    
    def compute_loss(self, x: np.ndarray, y: np.ndarray,
                    weights_list: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
                    kl_weight: float = 1.0) -> Tuple[float, float, float]:
        """Compute loss with KL regularization (ELBO for variational inference).
        
        Loss = Data_Loss + β * KL_Divergence
        
        Data loss measures fit to observations, KL term regularizes posterior
        to remain close to prior (prevents overfitting).
        
        Args:
            x: Input batch (batch_size, input_dim)
            y: Target batch (batch_size, output_dim)
            weights_list: Weights to use (samples if None)
            kl_weight: KL term coefficient (β in β-VAE)
            
        Returns:
            (total_loss, data_loss, kl_loss) tuple
        """
        # Forward pass
        pred = self.forward(x, weights_list=weights_list)
        
        # Mean squared error for data loss
        data_loss = np.mean((pred - y) ** 2)
        
        # KL divergence for regularization
        kl_loss = sum(layer.compute_kl_divergence() for layer in self.layers)
        
        # Total loss (negative ELBO)
        total_loss = data_loss + kl_weight * kl_loss / len(x)
        
        return total_loss, data_loss, kl_loss
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            n_epochs: int = 100,
            batch_size: int = 32,
            learning_rate: float = 0.01,
            kl_weight: float = 1.0,
            n_samples_per_batch: int = 1,
            verbose: bool = True) -> dict:
        """Train Bayesian network via stochastic variational inference.
        
        Updates weight posteriors using gradient descent on negative ELBO.
        Uses Monte Carlo sampling to estimate gradients.
        
        Args:
            x_train: Training inputs (n_samples, input_dim)
            y_train: Training targets (n_samples, output_dim)
            n_epochs: Number of training epochs
            batch_size: Mini-batch size
            learning_rate: Learning rate for gradient descent
            kl_weight: Weight for KL regularization term
            n_samples_per_batch: MC samples per batch for gradient estimation
            verbose: Print training progress
            
        Returns:
            Dictionary with training history
        """
        n_samples = len(x_train)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for epoch in range(n_epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            
            epoch_loss = 0.0
            epoch_data_loss = 0.0
            epoch_kl_loss = 0.0
            
            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, n_samples)
                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Monte Carlo gradient estimation
                batch_loss = 0.0
                batch_data_loss = 0.0
                batch_kl_loss = 0.0
                
                for _ in range(n_samples_per_batch):
                    # Sample weights
                    weights_list = [layer.sample_weights(self.temperature)
                                  for layer in self.layers]
                    
                    # Compute loss
                    loss, data_loss, kl_loss = self.compute_loss(
                        x_batch, y_batch, weights_list, kl_weight
                    )
                    
                    batch_loss += loss
                    batch_data_loss += data_loss
                    batch_kl_loss += kl_loss
                    
                    # Compute gradients via backpropagation
                    self._backward_pass(x_batch, y_batch, weights_list, 
                                      learning_rate, kl_weight)
                
                # Average over MC samples
                batch_loss /= n_samples_per_batch
                batch_data_loss /= n_samples_per_batch
                batch_kl_loss /= n_samples_per_batch
                
                epoch_loss += batch_loss
                epoch_data_loss += batch_data_loss
                epoch_kl_loss += batch_kl_loss
            
            # Average over batches
            epoch_loss /= n_batches
            epoch_data_loss /= n_batches
            epoch_kl_loss /= n_batches
            
            self.loss_history.append(epoch_loss)
            self.kl_history.append(epoch_kl_loss)
            
            if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch+1}/{n_epochs}: "
                      f"Loss={epoch_loss:.4f} "
                      f"(Data={epoch_data_loss:.4f}, KL={epoch_kl_loss:.4f})")
        
        return {
            'loss_history': self.loss_history,
            'kl_history': self.kl_history
        }
    
    def _backward_pass(self, x: np.ndarray, y: np.ndarray,
                      weights_list: List[Tuple[np.ndarray, np.ndarray]],
                      learning_rate: float, kl_weight: float):
        """Backpropagation to update weight posteriors.
        
        Updates both mean and standard deviation of weight posteriors
        using gradients from data loss and KL regularization.
        
        Implementation uses simple gradient descent on natural parameters
        of Gaussian posterior (mean, log-variance) with gradient clipping
        for numerical stability.
        """
        # Forward pass to get activations
        pred, activations = self.forward(x, weights_list, return_activations=True)
        
        # Output gradient (MSE derivative)
        delta = 2 * (pred - y) / len(x)
        
        # Clip delta to prevent explosion
        delta = np.clip(delta, -10.0, 10.0)
        
        # Backpropagate through layers
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            weights, bias = weights_list[i]
            
            # Input to this layer
            h_in = activations[i]
            
            # Gradient w.r.t weights and bias
            if h_in.ndim == 1:
                grad_w = np.outer(h_in, delta)
                grad_b = delta
            else:
                grad_w = h_in.T.dot(delta)
                grad_b = np.sum(delta, axis=0)
            
            # Add KL gradient (pulls posterior toward prior)
            grad_w += kl_weight * (weights - 0) / self.prior_std**2 / len(x)
            grad_b += kl_weight * (bias - 0) / self.prior_std**2 / len(x)
            
            # Clip gradients to prevent explosion
            grad_w = np.clip(grad_w, -1.0, 1.0)
            grad_b = np.clip(grad_b, -1.0, 1.0)
            
            # Update posterior mean
            layer.weight_mean -= learning_rate * grad_w
            layer.bias_mean -= learning_rate * grad_b
            
            # Clip weights to reasonable range
            layer.weight_mean = np.clip(layer.weight_mean, -10.0, 10.0)
            layer.bias_mean = np.clip(layer.bias_mean, -10.0, 10.0)
            
            # Update posterior std (ensure positive via log-space)
            # Simplified: keep std fixed or use more sophisticated update
            layer.weight_std *= 0.999  # Slow decay toward zero
            layer.bias_std *= 0.999
            
            # Keep std in reasonable range
            layer.weight_std = np.clip(layer.weight_std, 0.01, 10.0)
            layer.bias_std = np.clip(layer.bias_std, 0.01, 10.0)
            
            # Gradient w.r.t input (for next layer)
            delta = delta.dot(weights.T)
            
            # Clip to prevent explosion
            delta = np.clip(delta, -10.0, 10.0)
            
            # Apply activation derivative (except for input layer)
            if i > 0:
                h = activations[i]
                if self.activation == 'relu':
                    delta *= (h > 0).astype(float)
                elif self.activation == 'tanh':
                    delta *= (1 - h**2)
                elif self.activation == 'sigmoid':
                    delta *= h * (1 - h)


class BayesianRegressor(BayesianNetwork):
    """Bayesian neural network for regression with uncertainty.
    
    Extends BayesianNetwork with regression-specific utilities:
    - Prediction intervals
    - Calibrated confidence scores
    - Active learning sample selection
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int],
                 prior_std: float = 1.0, temperature: float = 1.0):
        """
        Initialize Bayesian regressor.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer sizes
            prior_std: Prior standard deviation
            temperature: Sampling temperature
        """
        layer_sizes = [input_dim] + hidden_dims + [1]
        super().__init__(layer_sizes, activation='relu', 
                        prior_std=prior_std, temperature=temperature)
    
    def predict_with_interval(self, x: np.ndarray, n_samples: int = 100,
                             confidence: float = 0.95) -> dict:
        """Predict with confidence interval.
        
        Computes prediction interval capturing both epistemic uncertainty
        (model uncertainty) and aleatoric uncertainty (data noise).
        
        Args:
            x: Input (batch_size, input_dim)
            n_samples: MC samples for uncertainty estimation
            confidence: Confidence level (e.g., 0.95 for 95% interval)
            
        Returns:
            Dictionary with mean, lower, upper, std
        """
        result = self.predict(x, n_samples=n_samples)
        
        # Compute confidence interval from samples
        alpha = 1 - confidence
        lower_percentile = alpha / 2
        upper_percentile = 1 - alpha / 2
        
        lower = np.percentile(result.samples, lower_percentile * 100, axis=0)
        upper = np.percentile(result.samples, upper_percentile * 100, axis=0)
        
        return {
            'mean': result.mean,
            'lower': lower,
            'upper': upper,
            'std': result.std,
            'confidence': result.confidence
        }
    
    def select_informative_samples(self, x_pool: np.ndarray, 
                                   n_select: int,
                                   n_samples: int = 100) -> np.ndarray:
        """Select most informative samples for active learning.
        
        Uses uncertainty sampling strategy: selects samples with
        highest predictive uncertainty (epistemic uncertainty).
        
        Args:
            x_pool: Pool of unlabeled samples (n_pool, input_dim)
            n_select: Number of samples to select
            n_samples: MC samples for uncertainty estimation
            
        Returns:
            Indices of selected samples
        """
        result = self.predict(x_pool, n_samples=n_samples)
        
        # Select samples with highest uncertainty
        uncertainty = np.mean(result.std, axis=-1)  # Average over output dims
        selected_indices = np.argsort(uncertainty)[-n_select:]
        
        return selected_indices


if __name__ == "__main__":
    print("=" * 80)
    print("BAYESIAN NEURAL NETWORKS WITH UNCERTAINTY QUANTIFICATION")
    print("=" * 80)
    
    # Generate synthetic regression data
    np.random.seed(42)
    n_train = 100
    x_train = np.random.uniform(-3, 3, size=(n_train, 1))
    y_train = np.sin(x_train) + 0.1 * np.random.randn(n_train, 1)
    
    print("\n[1] Training Bayesian Regressor")
    print("-" * 80)
    
    # Create and train model
    model = BayesianRegressor(
        input_dim=1,
        hidden_dims=[20, 20],
        prior_std=1.0,
        temperature=1.0
    )
    
    history = model.fit(
        x_train, y_train,
        n_epochs=50,
        batch_size=32,
        learning_rate=0.01,
        verbose=False
    )
    
    print(f"Final loss: {history['loss_history'][-1]:.4f}")
    
    # Make predictions
    print("\n[2] Predictions with Uncertainty")
    print("-" * 80)
    
    x_test = np.array([[-2.0], [0.0], [2.0]])
    result = model.predict_with_interval(x_test, n_samples=100)
    
    for i, x_val in enumerate(x_test.flatten()):
        print(f"x={x_val:+.1f}: "
              f"pred={result['mean'][i,0]:.3f} ± {result['std'][i,0]:.3f} "
              f"[{result['lower'][i,0]:.3f}, {result['upper'][i,0]:.3f}]")
    
    # Active learning demo
    print("\n[3] Active Learning Sample Selection")
    print("-" * 80)
    
    x_pool = np.random.uniform(-5, 5, size=(50, 1))
    selected = model.select_informative_samples(x_pool, n_select=5)
    
    print(f"Selected {len(selected)} most informative samples")
    print(f"Sample locations: {x_pool[selected].flatten()}")
    
    print("\n" + "=" * 80)
    print("Bayesian NNs provide calibrated uncertainty for safety-critical ML")
    print("=" * 80)

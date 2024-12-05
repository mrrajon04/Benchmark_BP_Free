import torch
import numpy as np
from scipy.stats import beta
import torch
import torch.nn as nn
class XavierInitialization:
    def __init__(self, mode="normal"):
        self.mode = mode

    def initialize_params(self, qubits):
        fan_in = fan_out = qubits
        if self.mode == "normal":
            params = np.random.normal(0, np.sqrt(2 / (fan_in + fan_out)), size=2 * qubits)
        elif self.mode == "uniform":
            limit = np.sqrt(6 / (fan_in + fan_out))
            params = np.random.uniform(-limit, limit, size=2 * qubits)
        else:
            raise ValueError("Invalid mode for Xavier Initialization.")
        return torch.tensor(params, dtype=torch.float32, requires_grad=True)

class HeInitialization:
    def __init__(self, mode="normal"):
        self.mode = mode

    def initialize_params(self, qubits):
        fan_in = qubits
        if self.mode == "normal":
            params = np.random.normal(0, np.sqrt(2 / fan_in), size=2 * qubits)
        elif self.mode == "uniform":
            limit = np.sqrt(6 / fan_in)
            params = np.random.uniform(-limit, limit, size=2 * qubits)
        else:
            raise ValueError("Invalid mode for He Initialization.")
        return torch.tensor(params, dtype=torch.float32, requires_grad=True)


class UniformNormInitialization:
    def __init__(self):
        pass

    def initialize_params(self, qubits):
        # Generate random parameters in the range [0, 1] and normalize them
        params = np.random.uniform(0, 1, size=2 * qubits)
        params_min, params_max = np.min(params), np.max(params)

        # Normalize the parameters to be within [0, 1)
        normalized_params = (params - params_min) / (params_max - params_min)
        normalized_params[normalized_params <= 0] += 1e-8
        normalized_params[normalized_params >= 1] -= 1e-8

        # Convert the normalized parameters into a PyTorch tensor
        return torch.tensor(normalized_params, dtype=torch.float32, requires_grad=True)


class BetaEbayesInitialization:
    def __init__(self, alpha=2.0, beta_param=5.0):
        """
        Initializes BetaEbayesInitialization with default alpha and beta values.
        The user can adjust these parameters to change the shape of the beta distribution.

        Parameters:
        - alpha (float): The alpha parameter of the beta distribution.
        - beta_param (float): The beta parameter of the beta distribution.
        """
        self.alpha = alpha
        self.beta_param = beta_param

    def initialize_params(self, qubits):
        """
        Generates initialization parameters based on the beta distribution.

        Parameters:
        - qubits (int): The number of qubits, which determines the size of the parameter array.

        Returns:
        - torch.Tensor: A tensor of initialized parameters.
        """
        # Generate parameters using the beta distribution
        params = np.random.beta(self.alpha, self.beta_param, size=2 * qubits)

        # Convert the parameters into a PyTorch tensor
        return torch.tensor(params, dtype=torch.float32, requires_grad=True)


class GaussianInitialization:
    def __init__(self, sigma=0.1, gmm_type="G1"):
        """
        Initializes the GaussianInitialization class with a specified variance and type.

        Parameters:
        - sigma (float): The standard deviation for the Gaussian distribution.
        - gmm_type (str): The type of Gaussian initialization ("G1", "G2", "G3").
        """
        self.sigma = sigma
        self.gmm_type = gmm_type

    def _g1(self, size):
        """Generate samples from G1: N(0, sigma^2)."""
        return np.random.normal(0, self.sigma, size)

    def _g2(self, size):
        """Generate samples from G2: 0.5 * N(-π/2, sigma^2) + 0.5 * N(π/2, sigma^2)."""
        mix = np.random.choice([-np.pi / 2, np.pi / 2], size=size, p=[0.5, 0.5])
        return np.random.normal(mix, self.sigma)

    def _g3(self, size):
        """Generate samples from G3: 0.25 * N(-π, sigma^2) + 0.25 * N(π, sigma^2) + 0.5 * N(0, sigma^2)."""
        mix = np.random.choice([-np.pi, np.pi, 0], size=size, p=[0.25, 0.25, 0.5])
        return np.random.normal(mix, self.sigma)

    def initialize_params(self, qubits):
        """
        Initializes parameters for the quantum circuit using the specified Gaussian Mixture Model.

        Parameters:
        - qubits (int): Number of qubits in the circuit.

        Returns:
        - torch.Tensor: Initialized parameters as a PyTorch tensor.
        """
        size = 2 * qubits  # Number of parameters (Rx and Ry for each qubit)
        if self.gmm_type == "G1":
            params = self._g1(size)
        elif self.gmm_type == "G2":
            params = self._g2(size)
        elif self.gmm_type == "G3":
            params = self._g3(size)
        else:
            raise ValueError("Invalid GMM type. Choose 'G1', 'G2', or 'G3'.")

        return torch.tensor(params, dtype=torch.float32, requires_grad=True)

class CNNInitialization:
    def __init__(self, input_dim=4, num_qubits=4):
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_dim // 2) * 32, num_qubits * 2),
            nn.Tanh(),
        )

    def initialize_params(self):
        dummy_input = torch.randn(1, 1, self.input_dim)
        params = self.model(dummy_input)
        return params.squeeze(0).detach().requires_grad_()


class ModelBasedInitialization:
    def __init__(self, num_qubits=4, layers=5):
        self.num_qubits = num_qubits
        self.layers = layers

    def initialize_params(self, input_data):
        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def encoder(params, x=None):
            for i in range(self.num_qubits):
                qml.RY(x[i], wires=i)
            for l in range(self.layers):
                for i in range(self.num_qubits):
                    qml.RY(params[l][i], wires=i)
                    if i < self.num_qubits - 1:
                        qml.CNOT(wires=[i, i + 1])
            return qml.density_matrix(wires=range(self.num_qubits))

        params_np = np.random.randn(self.layers, self.num_qubits)
        params = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)

        def cost_fn(params):
            rho = encoder(params, input_data)
            return -qml.math.trace(rho @ rho)

        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        for _ in range(100):
            params_np = optimizer.step(cost_fn, params_np)

        params = torch.tensor(params_np.flatten(), dtype=torch.float32, requires_grad=True)
        return params


class OptimizationBasedInitialization:
    def __init__(self, num_qubits=4, layers=5):
        self.num_qubits = num_qubits
        self.layers = layers

    def initialize_params(self):
        params = np.random.uniform(-np.pi, np.pi, size=(self.layers, self.num_qubits))
        return torch.tensor(params.flatten(), dtype=torch.float32, requires_grad=True)

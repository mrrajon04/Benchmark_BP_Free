import torch
import torch.nn as nn
import numpy as np
import pennylane as qml


class CNNInitialization:
    def __init__(self, input_dim=4, num_qubits=4):
        """
        Initializes the CNN-based initialization strategy.

        Parameters:
        - input_dim (int): The size of the input vector for the CNN.
        - num_qubits (int): The number of qubits in the quantum circuit.
        """
        self.input_dim = input_dim
        self.num_qubits = num_qubits
        self.model = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((input_dim // 2) * 32, num_qubits * 2),  # RX and RY per qubit
            nn.Tanh(),
        )

    def initialize_params(self):
        """
        Generate initialization parameters using the CNN.

        Returns:
        - torch.Tensor: Initialized parameters as a PyTorch tensor.
        """
        dummy_input = torch.randn(1, 1, self.input_dim)
        params = self.model(dummy_input)
        return params.squeeze(0).detach().requires_grad_()


class ModelBasedInitialization:
    def __init__(self, num_qubits=4, layers=5):
        """
        Initializes the model-based strategy with subsystem purification.

        Parameters:
        - num_qubits (int): Number of qubits in the quantum circuit.
        - layers (int): Number of layers in the variational encoder.
        """
        self.num_qubits = num_qubits
        self.layers = layers

    def initialize_params(self, input_data):
        """
        Performs subsystem purification for initialization.

        Parameters:
        - input_data (numpy.ndarray): Data for dimensionality reduction.

        Returns:
        - torch.Tensor: Initialized parameters as a PyTorch tensor.
        """
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

        # Initialize parameters with NumPy, then convert to PyTorch tensor
        params_np = np.random.randn(self.layers, self.num_qubits)
        params = torch.tensor(params_np, dtype=torch.float32, requires_grad=True)

        def cost_fn(params):
            rho = encoder(params, input_data)
            return -qml.math.trace(rho @ rho)  # Negative trace of the squared density matrix

        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)

        for _ in range(100):
            params_np = optimizer.step(cost_fn, params_np)

        # Convert optimized NumPy parameters back to PyTorch tensor
        params = torch.tensor(params_np.flatten(), dtype=torch.float32, requires_grad=True)
        return params

class OptimizationBasedInitialization:
    def __init__(self, num_qubits=4, layers=5):
        """
        Initializes the optimization-based strategy with Fourier initialization.

        Parameters:
        - num_qubits (int): Number of qubits in the quantum circuit.
        - layers (int): Number of layers in the quantum circuit.
        """
        self.num_qubits = num_qubits
        self.layers = layers

    def initialize_params(self):
        """
        Generates Fourier-based initialization parameters.

        Returns:
        - torch.Tensor: Initialized parameters as a PyTorch tensor.
        """
        params = np.random.uniform(-np.pi, np.pi, size=(self.layers, self.num_qubits))
        return torch.tensor(params.flatten(), dtype=torch.float32, requires_grad=True)


cnn_init = CNNInitialization(input_dim=32, num_qubits=8)
cnn_params = cnn_init.initialize_params()
print("CNN Initialization Parameters:", cnn_params)
#
# Model-Based Initialization
model_init = ModelBasedInitialization(num_qubits=8, layers=5)
model_params = model_init.initialize_params(np.random.rand(8))
print("Model-Based Initialization Parameters:", model_params)

# Optimization-Based Initialization
# opt_init = OptimizationBasedInitialization(num_qubits=16, layers=5)
# opt_params = opt_init.initialize_params()
# print("Optimization-Based Initialization Parameters:", opt_params)
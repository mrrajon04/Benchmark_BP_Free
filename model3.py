import torch
import torch.nn as nn
import numpy as np
import pennylane as qml
import os
import csv
from torch.utils.data import DataLoader, TensorDataset

csv_dir = "results(new2)/csv"
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)


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

    def initialize_params(self, input_data=None):
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


class QuantumModel:
    def __init__(self, qubits, initialization_strategy):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qubits = qubits
        self.initialization_strategy = initialization_strategy
        self.init_params = initialization_strategy.initialize_params(qubits).to(self.device)

        qml_device = "lightning.qubit" if self.device.type == "cuda" else "default.qubit"
        self.qml_device = qml.device(qml_device, wires=qubits)

        @qml.qnode(self.qml_device, interface="torch")
        def circuit(params, x=None):
            for i in range(self.qubits):
                qml.RX(params[i], wires=i)
                qml.RY(params[i + self.qubits], wires=i)
            for i in range(self.qubits - 1):
                qml.CZ(wires=[i, i + 1])
            return qml.expval(qml.PauliZ(0))

        self.circuit = circuit

    def train(self, data, dName, strategy, qubits, epochs, learning_rate, batch_size=16):
        x_train = torch.tensor(data['x_train'], dtype=torch.float32).to(self.device)
        y_train = torch.tensor(data['y_train'], dtype=torch.float32).to(self.device)

        x_train = (x_train - x_train.mean()) / x_train.std()
        y_train = (y_train - y_train.mean()) / y_train.std()

        dataset = TensorDataset(x_train, y_train)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        strategy_dir = f"{csv_dir}/{strategy}"
        os.makedirs(strategy_dir, exist_ok=True)

        with open(f"{strategy_dir}/({dName}_{qubits}).csv", 'w', newline='') as csvfile:
            fieldnames = ['Epoch', 'Loss', 'Gradient', 'Gradient_Variance']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            optimizer = torch.optim.Adam([self.init_params], lr=learning_rate)
            all_gradients = []

            for epoch in range(epochs):
                epoch_loss = 0
                epoch_gradients = []
                batch_count = 0

                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()

                    def cost_fn():
                        predictions = torch.stack([self.circuit(self.init_params, x=x) for x in batch_x])
                        return torch.mean((predictions - batch_y) ** 2)

                    loss = cost_fn()
                    loss.backward()

                    gradients = self.init_params.grad.view(-1).detach().cpu().numpy()
                    epoch_gradients.append(gradients)
                    all_gradients.append(gradients)

                    optimizer.step()

                    epoch_loss += loss.item()
                    batch_count += 1

                avg_loss = epoch_loss / batch_count
                epoch_gradients = np.array(epoch_gradients)
                gradient_mean = np.mean(np.linalg.norm(epoch_gradients, axis=1))
                gradient_variance = np.var(np.linalg.norm(epoch_gradients, axis=1))

                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}, Gradient = {gradient_mean:.6f}")

                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': avg_loss,
                    'Gradient': gradient_mean,
                })

            all_gradients = np.array(all_gradients)
            overall_gradient_variance = np.var(np.linalg.norm(all_gradients, axis=1))
            writer.writerow({
                'Gradient_Variance': overall_gradient_variance
            })
            print(f"Overall Gradient Variance: {overall_gradient_variance:.6f}")

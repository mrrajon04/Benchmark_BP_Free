import pennylane as qml
import torch
import os
import csv
from torch.utils.data import DataLoader, TensorDataset
import numpy as np  # Ensure NumPy is imported


csv_dir = "results/csv"
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

class QuantumModel:
    def __init__(self, qubits, initialization_strategy):
        # Check for GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.qubits = qubits
        self.initialization_strategy = initialization_strategy
        self.init_params = initialization_strategy.initialize_params(qubits).to(self.device)

        # Use lightning.qubit for GPU or default.qubit for CPU
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

                print(
                    f"Epoch {epoch + 1}: Loss = {avg_loss:.6f}, Gradient = {gradient_mean:.6f}")

                writer.writerow({
                    'Epoch': epoch + 1,
                    'Loss': avg_loss,
                    'Gradient': gradient_mean
                })

            all_gradients = np.array(all_gradients)
            overall_gradient_variance = np.var(np.linalg.norm(all_gradients, axis=1))
            writer.writerow({
                'Gradient_Variance': overall_gradient_variance
            })
            print(f"Overall Gradient Variance: {overall_gradient_variance:.6f}")

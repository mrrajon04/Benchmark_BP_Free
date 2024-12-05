from model import QuantumModel
# from model3 import QuantumModel, CNNInitialization, ModelBasedInitialization, OptimizationBasedInitialization
from load_data import load_iris_data, load_mnist_data, load_medmnist_data
from init_strat import XavierInitialization, HeInitialization, UniformNormInitialization, BetaEbayesInitialization, GaussianInitialization
datas = {
    # "Iris": load_iris_data(),
    "MNIST": load_mnist_data(),
    # "MedMNIST": load_medmnist_data()
}
# Define qubit configurations and initialization strategies
qubit_counts = [16]
initialization_strategies = {
    "Xavier_Normal": XavierInitialization(mode="normal"),
    "Xavier_Uniform": XavierInitialization(mode="uniform"),
    "He_Normal": HeInitialization(mode="normal"),
    "He_Uniform": HeInitialization(mode="uniform"),
    "Uniform_norm": UniformNormInitialization(),
    "BetaEbayesInitialization": BetaEbayesInitialization(),
    "Gaussian_G1": GaussianInitialization(sigma=0.1, gmm_type="G1"),
    "CNNInitialization": UniformNormInitialization(),
    "ModelBasedInitialization": XavierInitialization(mode="uniform"),
    "OptimizationBasedInitialization": HeInitialization(mode="uniform"),
}


# results(old) = []

# Train models for different initializations and qubit counts
for dName, data in datas.items():
    for qubits in qubit_counts:
        for name, strategy in initialization_strategies.items():
            print(f"Training VQC with {qubits} qubits using {name} initialization on {dName} daataset...")
            model = QuantumModel(qubits=qubits, initialization_strategy=strategy)
            # print(data, "shape")
            history = model.train(data, dName, name, qubits, epochs=30, learning_rate=0.001)
            # plateau_scores = calculate_plateau_score(history["gradient_norm"], history["loss"])
            # results(old).append((qubits, name, history))


import matplotlib.pyplot as plt
import numpy as np

# def calculate_plateau_score(gradient_norms, losses, grad_threshold=1e-5, loss_threshold=1e-3):
#     plateau_scores = []
#     for grad_norm, loss_diff in zip(gradient_norms, np.diff(losses)):
#         grad_condition = grad_norm < grad_threshold
#         loss_condition = abs(loss_diff) < loss_threshold
#         score = int(grad_condition) + int(loss_condition)
#         plateau_scores.append(score)
#     return score
#

# def visualize_results(results(old)):
#     for qubits, name, history, plateau_scores in results(old):
#         epochs = range(len(history["loss"]))
#         plt.figure(figsize=(10, 5))
#         plt.subplot(1, 2, 1)
#         plt.plot(epochs, history["loss"], label="Loss")
#         plt.xlabel("Epochs")
#         plt.ylabel("Loss")
#         plt.title(f"{name} Initialization: Loss (Qubits={qubits})")
#         plt.legend()
#
#         plt.subplot(1, 2, 2)
#         plt.plot(epochs, history["gradient_norm"], label="Gradient Norm")
#         plt.xlabel("Epochs")
#         plt.ylabel("Gradient Norm")
#         plt.title(f"{name} Initialization: Gradient Norm (Qubits={qubits})")
#         plt.legend()
#
#         plt.tight_layout()
#         plt.savefig(f"results(old)/{name}_Qubits_{qubits}.png")
#         plt.show()

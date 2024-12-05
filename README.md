# Benchmark_BP_Free
## Overview
Benchmark_BP_Free is a research framework for benchmarking. This repository allows experimentation with various Initialization, Model and Optimization methods and training configurations across different datasets.
---
## Key Features
- **Comprehensive Benchmarking**: Evaluate the effectiveness of multiple BP mitigation techniques.  
- **Customizable Configurations**: Experiment with initialization, model design, and optimization strategies.  
- **Dataset Variety**: Includes datasets of varying complexity (Iris, MNIST, MedMNIST).  
- **Detailed Evaluation Metrics**: Assess training performance using gradient norm, training loss, and classification accuracy.  

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Parameters](#parameters)
- [Sample Results](#sample-results)
- [Licence](#license)

---
---

# Installation

To install the framework from source, follow these steps:

1. Clone the repository:
   
   ```bash
   git clone https://github.com/mrrajon04/Benchmark_BP_Free.git
3. Navigate to the project directory:
```
cd Benchmark_BP_Free
```
# Usage
Train model
This framework supports training on multiple datasets using different initialization methods. Below is an example of training a model with Gaussian initialization on the MNIST dataset for 20 epochs.

Run Training
   After configuring the arguments.yaml file, start training by running:

   ```
   python main.py
   ```
# Repository Configuration

This repository supports customizable configurations, enabling users to experiment with various settings and techniques. Below are the supported configurations:

## Initialization Methods
Choose from the following initialization methods:
- **Beta**
- **Gaussian**
- **He_normal**
- **He_uniform**
- **Xavier_normal**
- **Xavier_uniform**
- **Uniform Norm**
- **Classical CNN-based initialization**

## Model Architectures
- **Variational Encoders**: Easily configurable for your experiments.

## Optimization Techniques
- **Time-Nonlocal Optimization**: Leverage this advanced optimization technique for improved performance.

---


# Sample Results

Performance of various BP mitigation strategies on different datasets:
![Benchmark BP Free Results](https://github.com/mrrajon04/Benchmark_BP_Free/blob/main/results/image.png)

---

These results showcase the effectiveness of different initialization and optimization strategies across datasets.
# Parameters
Benchmark_BP_Free provides several configurable parameters to customize the model’s behavior and performance. Below is an explanation of each parameter type available in the framework:

## Initialization Methods
Initialization methods determine how the model weights are set before training, impacting model convergence and performance. Available methods include:

__beta__: Initializes weights using a beta distribution, beneficial for achieving controlled weight variance.

__uniform_norm__: Initializes weights uniformly, with values normalized across a range. This helps ensure balanced weight distribution across layers.

__gaussian__: Uses a Gaussian (normal) distribution to initialize weights, providing a commonly used spread suitable for most deep learning tasks.

__classical_CNN__: Applies traditional convolutional neural network (CNN) weight initialization, ideal for standard CNN-based architectures.
## Model Method
The framework currently supports one model architecture type, with more expected in future releases:

__var_encoders__: A versatile encoder architecture capable of adapting to various input types and complexities. Suitable for tasks requiring feature encoding.
## Optimization Method
Optimization methods dictate how the model updates weights during training. Benchmark_BP_Free currently includes:

__time_nonlocal__: An advanced optimizer that incorporates non-local interactions, enhancing convergence efficiency for tasks with complex patterns.
## Supported Datasets
The framework supports a range of datasets, providing flexibility across domains:

__Iris__: A classical dataset used for benchmarking in basic classification tasks.

__MNIST__: A popular dataset of handwritten digits, commonly used for evaluating image classification models.

__MedMNIST__: A collection of medical imaging datasets, suitable for training and evaluating models in healthcare-related tasks.
# License
This project is licensed under the MIT License.

This `README.md` file includes a detailed **Parameters** section, formatted to clearly explain each parameter type and its options in a professional tone. Let me know if you’d like further adjustments!


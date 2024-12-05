from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import medmnist
import numpy as np

from medmnist import INFO
from medmnist.dataset import BreastMNIST

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_iris_data():
    iris = load_iris()
    X = iris.data  # Use all features
    y = iris.target  # Use all classes (no binary classification)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.96, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {"x_train": X_train, "y_train": y_train, "x_test": X_test, "y_test": y_test}

def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()])
    mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    data = mnist_dataset.data.numpy()
    labels = mnist_dataset.targets.numpy()

    data = data.reshape(data.shape[0], -1)  # Flatten 28x28 images into vectors of 784 pixels
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.9999, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {"x_train": X_train, "y_train": y_train, "x_test": X_test, "y_test": y_test}
load_mnist_data()
def load_medmnist_data():
    dataset_name = 'breastmnist'
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = BreastMNIST(split='train', transform=transform, download=True)
    test_dataset = BreastMNIST(split='test', transform=transform, download=True)

    train_data = train_dataset.imgs.reshape(train_dataset.imgs.shape[0], -1)
    train_labels = train_dataset.labels

    test_data = test_dataset.imgs.reshape(test_dataset.imgs.shape[0], -1)
    test_labels = test_dataset.labels

    data = np.concatenate((train_data, test_data), axis=0)
    labels = np.concatenate((train_labels, test_labels), axis=0)

    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.965, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return {"x_train": X_train, "y_train": y_train, "x_test": X_test, "y_test": y_test}

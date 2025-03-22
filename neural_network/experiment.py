from model import Model, train
from model_torch import TorchModel, MnistDataset, torch_train
from layer import Layer
from enum import Enum
import numpy as np


class ModelType(Enum):
    CUSTOM = 1
    TORCH = 2


def experiment(
    data_train: tuple[np.ndarray, np.ndarray],
    data_val: tuple[np.ndarray, np.ndarray],
    list_of_layers: list[list[tuple[tuple[int, int]], str]],
    num_epochs: int,
    learning_rates: list[float],
    batch_sizes: list[int],
    model_type: ModelType,
) -> tuple[list[list[float]], list[float], list[float]]:
    X_train, y_train = data_train
    X_val, y_val = data_val
    losses = []
    train_accuracies = []
    val_accuracies = []
    exec_times = []
    for layers, learning_rate, batch_size in zip(
        list_of_layers, learning_rates, batch_sizes
    ):
        if model_type == ModelType.CUSTOM:
            model = Model()
            for layer in layers:
                model.add_module(Layer(layer[0], layer[1]))
            model_losses, time = train(
                model, X_train, y_train, num_epochs, learning_rate, batch_size
            )
        else:
            model = TorchModel(layers)
            dataset = MnistDataset(y_train, X_train)
            model_losses, time = torch_train(
                dataset, model, learning_rate, num_epochs, batch_size
            )
        losses.append(model_losses)
        exec_times.append(time)
        train_accuracy = measure_accuracy(model, X_train, y_train)
        val_accuracy = measure_accuracy(model, X_val, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    return losses, train_accuracies, val_accuracies, exec_times


def evaluate_best_model(
    layers: list[list[tuple[tuple[int, int]], str]],
    data_train: tuple[np.ndarray, np.ndarray],
    data_val: tuple[np.ndarray, np.ndarray],
    data_test: tuple[np.ndarray, np.ndarray],
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[float, float, float]:
    X_train, y_train = data_train
    X_val, y_val = data_val
    X_test, y_test = data_test
    model = Model()
    for layer in layers:
        model.add_module(Layer(layer[0], layer[1]))
    model_losses, time = train(
        model, X_train, y_train, num_epochs, learning_rate, batch_size
    )
    train_accuracy = measure_accuracy(model, X_train, y_train)
    val_accuracy = measure_accuracy(model, X_val, y_val)
    test_accuracy = measure_accuracy(model, X_test, y_test)
    return train_accuracy, val_accuracy, test_accuracy


def measure_accuracy(model: Model, x: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy.item()

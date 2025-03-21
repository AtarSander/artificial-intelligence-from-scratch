from model import Model, train
from layer import Layer
from preprocessing import MnistDataloader
from PIL import Image as im
import matplotlib.pyplot as plt
import numpy as np


def experiment(
    data_train: tuple[np.ndarray, np.ndarray],
    data_val: tuple[np.ndarray, np.ndarray],
    list_of_layers: list[list[tuple[tuple[int, int]], str]],
    num_epochs: int,
    learning_rates: list[float],
    batch_sizes: list[int],
) -> tuple[list[list[float]], list[float], list[float]]:
    X_train, y_train = data_train
    X_val, y_val = data_val
    losses = []
    train_accuracies = []
    val_accuracies = []
    for layers, learning_rate, batch_size in zip(
        list_of_layers, learning_rates, batch_sizes
    ):
        model = Model()
        for layer in layers:
            model.add_module(Layer(layer[0], layer[1]))
        model_losses = train(
            model, X_train, y_train, num_epochs, learning_rate, batch_size
        )
        losses.append(model_losses)
        train_accuracy = measure_accuracy(model, X_train, y_train)
        val_accuracy = measure_accuracy(model, X_val, y_val)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
    return losses, train_accuracies, val_accuracies


def measure_accuracy(model: Model, x: np.ndarray, y: np.ndarray) -> float:
    y_pred = model.predict(x)
    y_pred = np.argmax(y_pred, axis=1)
    accuracy = np.sum(y_pred == y) / len(y)
    return accuracy.item()


def view_image(image: np.ndarray) -> None:
    data = im.fromarray(image)
    data.save("test.png")


def plot_costs(costs):
    x_values_modified = [x for x in range(len(costs[0]))]
    for ind, cost in enumerate(costs):
        plt.plot(
            x_values_modified,
            cost,
            marker="",
            linestyle="-",
            label=f"Model number: {ind}",
        )

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Costs of models in training")
    plt.legend()
    plt.show()

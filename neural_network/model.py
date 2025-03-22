from utils import timer
import numpy as np


class Model:
    def __init__(self, num_classes: int = 10) -> None:
        self.layers = []
        self.num_classes = num_classes

    def add_module(self, layer: tuple[int, int]) -> None:
        self.layers.append(layer)

    def forward_propagation(self, X: np.ndarray) -> np.ndarray:
        A_prev = X
        for layer in self.layers:
            A = layer.forward(A_prev)
            A_prev = A
        AL = A_prev
        return AL

    def calculate_loss(self, y_pred: np.ndarray, y_truth: np.ndarray) -> float:
        epsilon = 1e-8
        y_truth_one_hot = np.zeros((y_truth.size, self.num_classes))
        y_truth_one_hot[np.arange(y_truth.size), y_truth] = 1
        CE = -np.sum(y_truth_one_hot * np.log(y_pred + epsilon), axis=1)
        return np.mean(CE)

    def backward_propagation(self, AL: np.ndarray, Y: np.ndarray) -> None:
        Y_one_hot = np.zeros((Y.size, self.num_classes))
        Y_one_hot[np.arange(Y.size), Y] = 1
        dA_prev = AL - Y_one_hot
        for layer in reversed(self.layers):
            dW, db, dA_prev = layer.backward(dA_prev)
            layer.update_parameters(dW, db, self.learning_rate)

    def step(self, X: np.ndarray, Y: np.ndarray, learning_rate: float) -> float:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        self.learning_rate = learning_rate
        y_pred = self.forward_propagation(X)
        loss = self.calculate_loss(y_pred, Y)
        self.backward_propagation(y_pred, Y)
        return loss

    def predict(self, X: np.ndarray) -> int:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
        y_pred = self.forward_propagation(X)
        return y_pred


@timer
def train(
    model: Model,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    batch_size: int,
) -> list[float]:

    dataset_size = len(X)
    losses = []
    for epoch in range(epochs):
        loss_sum = 0
        for i in range(0, dataset_size, batch_size):
            batch_X = X[i : i + batch_size] / 255.0
            batch_Y = y[i : i + batch_size]
            loss_sum += model.step(batch_X, batch_Y, learning_rate)

        losses.append(loss_sum / (dataset_size / batch_size))
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}, loss: {loss_sum / (dataset_size / batch_size)}")

    return losses

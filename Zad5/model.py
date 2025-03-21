import numpy as np


class Model:
    def __init__(self) -> None:
        self.layers = []

    def add_module(self, layer: tuple[int, int]) -> None:
        self.layers.append(layer)

    def forward_propagation(self, X: np.array) -> np.array:
        A_prev = X
        for i, layer in enumerate(self.layers):
            A = layer.forward(A_prev)
            A_prev = A
        AL = A_prev
        return AL

    def calculate_loss(self, y_pred: np.array, y_truth: np.array) -> float:
        epsilon = 1e-8
        y_truth_one_hot = np.zeros((y_truth.size, np.max(y_truth)+1))
        y_truth_one_hot[np.arange(y_truth.size), y_truth] = 1
        CE = -np.mean(np.multiply(y_truth_one_hot, np.log(y_pred+epsilon)))
        return CE

    def backward_propagation(self, AL: np.array, Y: np.array) -> None:
        Y_one_hot = np.zeros((Y.size, np.max(Y)+1))
        Y_one_hot[np.arange(Y.size), Y] = 1
        dA_prev = AL - Y_one_hot
        for i, layer in enumerate(reversed(self.layers)):
            dW, db, dA_prev = layer.backward(dA_prev)
            layer.update_parameters(dW, db, self.learning_rate)

    def step(self, X: np.array, Y: np.array, learning_rate: float) -> float:
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        self.learning_rate = learning_rate
        y_pred = self.forward_propagation(X)
        loss = self.calculate_loss(y_pred, Y)
        self.backward_propagation(y_pred, Y)
        return loss

    def predict(self, X: np.array) -> int:
        X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
        y_pred = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)


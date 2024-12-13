import numpy as np


class Model:
    def __init__(self):
        self.layers = []

    def add_module(self, layer):
        self.layers.append(layer)

    def forward_propagation(self, X):
        A_prev = X
        for layer in self.layers:
            A = layer.forward(A_prev)
            A_prev = A
        AL = A_prev
        return AL

    def calculate_loss(self, y_pred, y_truth):
        epsilon = 1e-8
        CE = -np.sum(np.multiply(y_truth, np.log(y_pred+epsilon)))
        return CE

    def backward_propagation(self, AL, Y):
        dA_prev = AL - Y
        for layer in reversed(self.layers):
            dW, db, dA_prev = layer.backward(dA_prev)
            layer.update_parameters(dW, db, self.learning_rate)

    def train(self, X, Y, epochs, learning_rate):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)
            loss = self.calculate_loss(y_pred, Y)
            self.backward_propagation(y_pred, Y)
            if epoch % 10 == 0:
                print(f"Loss: {loss}")

    def predict(self, X):
        y_pred = self.forward_propagation(X)
        return y_pred


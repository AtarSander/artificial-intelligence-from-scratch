import numpy as np


class Layer:
    def __init__(self, layer_dims: tuple[int, int], activation: str) -> None:
        self.initialize_parameters(layer_dims)
        self.activation_type = activation.lower()
        self.cache = {}

    def initialize_parameters(self, dimensions: tuple[int, int]) -> None:
        self.weights = np.random.randn(dimensions[1], dimensions[0]) * np.sqrt(
            2 / dimensions[0]
        )
        self.bias = np.zeros((dimensions[1],))

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache["input"] = X
        Z = self.forward_linear(X)
        Z = self.activation(Z)
        return Z

    def forward_linear(self, X: np.ndarray) -> np.ndarray:
        w, b = self.weights, self.bias
        Z = np.dot(X, np.transpose(w)) + b
        return Z

    def activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation_type == "relu":
            Z = self.relu(Z)
        elif self.activation_type == "softmax":
            Z = self.softmax(Z)
        return Z

    def backward(self, d_prev: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.activation_type == "relu":
            dZ = d_prev * self.cache["activation"]
        else:
            dZ = d_prev

        dW = np.dot(np.transpose(self.cache["input"]), dZ)
        db = np.sum(np.transpose(dZ), axis=1, keepdims=False)
        dA_prev = np.dot(dZ, self.weights)
        return dW, db, dA_prev

    def relu(self, Z: np.ndarray) -> np.ndarray:
        self.cache["activation"] = (Z > 0).astype(float)
        return np.maximum(Z, 0)

    def softmax(self, x: np.ndarray) -> np.ndarray:
        shift_x = x - np.max(x, axis=-1, keepdims=True)
        e_x = np.exp(shift_x)
        softmax = e_x / np.sum(e_x, axis=-1, keepdims=True)
        self.cache["activation"] = softmax * (1 - softmax)
        return softmax

    def update_parameters(
        self, grads_w: np.ndarray, grads_b: np.ndarray, learning_rate: float
    ) -> None:
        self.weights = self.weights - learning_rate * np.transpose(grads_w)
        self.bias = self.bias - learning_rate * grads_b

import numpy as np
import pandas as pd

class Model:
    def __init__(self, layer_dims):
        pass

    def initialize_parameters(self, layer_dims):
        np.random.seed(2)
        parameters = {}
        L = len(layer_dims)
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
            parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        return parameters

    def relu(self, Z):
        cache = Z
        A = np.maximum(0, Z)
        return A, cache


    def softmax(self, Z):
        cache = Z
        e_x = np.exp(Z - np.max(Z))
        A = e_x / e_x.sum(axis=0)
        return A, cache


    def linear_forward(self, A, W, b):
        Z = np.matmul(W, A) + b
        cache = (A, W, b)
        return Z, cache


    def linear_activation_forward(self, prev_A, W, b, activation_function):
        Z, linear_cache = self.linear_forward(prev_A, W, b)
        if activation_function == "relu":
            A, activation_cache = self.relu(Z)

        elif activation_function == "softmax":
            A, activation_cache = self.softmax(Z)

        cache = (linear_cache, activation_cache)
        return A, cache


    def forward_propagation(self, X, parameters):
        caches = []
        L = len(parameters) //2
        A = X

        for l in range(1, L):
            A_prev = A
            A, cache = self.linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
            caches.append(cache)
        
        AL, cache = self.linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "soft")
        caches.append(cache)

        return AL, caches


    def cost_function(self, AL, Y):
        m = Y.shape[1]
        cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
        return np.squeeze(cost)


    def relu_backward(self, dA, activ_cache):
        Z = activ_cache
        dZ = np.array(dA, copy=True)
        dZ[Z <= 0] = 0
        return dZ

    def softmax_backward(self, dA, activ_cache):
        Z = activ_cache
        dZ = np.matmul(Z, (1 - Z))
        return dZ


    def linear_backward(self, dZ, cache):
        A_prev, W, b = cache
        m = A_prev.shape[1]
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.matmul(W.T, dZ)
        return dA_prev, dW, db


    def linear_activation_backward(self, dA, cache, activation_function):
        linear_cache, activ_cache = cache
        if activation_function == "relu":
            dZ = self.relu_backward(dA, activ_cache)
        elif activation_function == "softmax":
            dZ = self.softmax_backward(dA, activ_cache)

        dA, dW, db = self.linear_backward(dZ, linear_cache)
        return dA, dW, db


    def backward_propagation(self, AL, Y, caches):
        gradients = {}
        L = len(caches)
        Y = Y.reshape(AL.shape)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        current_cache = caches[L-1]
        dA_prev, dW, db = self.linear_activation_backward(dAL, current_cache, "softmax")
        gradients['dA' + str(L-1)] = dA_prev
        gradients['dW' + str(L)] = dW
        gradients['db' + str(L)] = db

        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev, dW, db = self.linear_activation_backward(gradients['dA'+str(l+1)], current_cache, "relu")
            gradients['dA' + str(l)] = dA_prev
            gradients['dW' + str(l+1)] = dW
            gradients['db' + str(l+1)] = db
        return gradients


    def update_parameters(self, parameters, gradients, learning_rate):
        L = len(parameters) // 2
        parameters = parameters.copy()
        for l in range(L):
            parameters['W'+str(l+1)] = parameters['W'+str(l+1)] - learning_rate * gradients['dW'+str(l+1)]
            parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * gradients['db'+str(l+1)]
        return parameters


    def model(self, X, Y, layer_dims, learning_rate, num_iterations, print_cost):
        parameters = self.initialize_parameters(layer_dims)
        A_prev = X
        costs = []
        for i in range(num_iterations):
            AL, caches = self.forward_propagation(A_prev, parameters)
            cost = self.cost_function(AL, Y)
            costs.append(cost)
            gradients = self.backward_propagation(AL, Y, caches)
            parameters = self.update_parameters(parameters, gradients, learning_rate)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {i}: {cost}\n")
        return parameters, costs


    def predict(self, X, Y, parameters, values):
        AL, _ = self.forward_propagation(X, parameters)
        n = AL.shape[1]
        predictions = np.zeros((1, n))
        for i in range(n):
            index = np.argmax(AL[0, i])
            predictions [0, i] = values[index] 
            accuracy = np.sum(predictions == Y)//n
        return accuracy
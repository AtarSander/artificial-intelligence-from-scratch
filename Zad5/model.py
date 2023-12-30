import numpy as np
import pandas as pd


def initialize_parameters(layer_dims):
    np.random.seed(2)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))


def relu(Z):
    cache = Z
    A = np.maximum(0, Z)
    return A, cache


def softmax(Z):
    cache = Z
    e_x = np.exp(Z - np.max(Z))
    A = e_x / e_x.sum(axis=0)
    return A, cache


def linear_forward(A, W, b):
    Z = np.matmul(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(prev_A, W, b, activation_function):
    Z, linear_cache = linear_forward(prev_A, W, b)
    if activation_function == "relu":
        A, activation_cache = relu(Z)

    elif activation_function == "softmax":
        A, activation_cache = softmax(Z)

    cache = (linear_cache, activation_cache)
    return A, cache


def forward_propagation(X, parameters):
    caches = []
    L = len(parameters) //2
    A = X

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], "relu")
        caches.append(cache)
    
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], "soft")
    caches.append(cache)

    return AL, caches


def cost_function(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    return np.squeeze(cost)




    

        

# def model(X, Y, layer_dims, learning_rate, num_iterations, print_cost):
#     pass


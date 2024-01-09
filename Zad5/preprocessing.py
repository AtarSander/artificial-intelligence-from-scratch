import numpy as np
import gzip
import math
from keras.datasets import mnist


def convert(filename):
    with open(filename, "rb") as f:
        data = f.read()
    return np.frombuffer(gzip.decompress(data), dtype=np.uint8).copy()


def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.reshape((-1, 28 * 28)) / 255.0 * 0.99 + 0.01
    X_test = X_test.reshape((-1, 28 * 28)) / 255.0 * 0.99 + 0.01
    rand = np.arange(60000)
    np.random.shuffle(rand)
    train_no = rand[:50000]
    val_no = np.setdiff1d(rand, train_no)
    X_train, X_dev = X_train[train_no, :], X_train[val_no, :]
    Y_train, Y_dev = Y_train[train_no], Y_train[val_no]

    Y_train_onehot = np.zeros((Y_train.size, Y_train.max() + 1))
    Y_train_onehot[np.arange(Y_train.size), Y_train] = 1

    Y_dev_onehot = np.zeros((Y_dev.size, Y_dev.max() + 1))
    Y_dev_onehot[np.arange(Y_dev.size), Y_dev] = 1

    Y_test_onehot = np.zeros((Y_test.size, Y_test.max() + 1))
    Y_test_onehot[np.arange(Y_test.size), Y_test] = 1
    return X_train, Y_train_onehot, X_dev, Y_dev_onehot, X_test, Y_test_onehot

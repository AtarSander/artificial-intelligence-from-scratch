import numpy as np
from PIL import Image
from keras.datasets import mnist


def load_dataset():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = np.array(
        [np.array(Image.fromarray(img).resize((14, 14))) for img in X_train]
    )
    X_test = np.array(
        [np.array(Image.fromarray(img).resize((14, 14))) for img in X_test]
    )

    X_train = X_train.reshape((-1, 14 * 14)) / 255.0 * 0.99 + 0.01
    X_test = X_test.reshape((-1, 14 * 14)) / 255.0 * 0.99 + 0.01

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

    return (
        X_train.T[:10000, :],
        Y_train_onehot.T[:10000, :],
        X_dev.T[:1000, :],
        Y_dev_onehot.T[:1000, :],
        X_test.T[:2000, :],
        Y_test_onehot.T[:2000, :],
    )

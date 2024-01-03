from treeID3 import Tree
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt


def experiment(X_train, Y_train, X_dev, Y_dev, parameters, target):
    acc_train = []
    acc_dev = []
    accuracy = {}
    depths = np.arange(parameters[0], parameters[1] + 1)
    for depth in depths:
        model = Tree(depth, target)
        model.fit(X_train, Y_train)
        Y_train_pred = model.predict(X_train)
        acc_train.append(accuracy_score(Y_train, Y_train_pred))
        Y_dev_pred = model.predict(X_dev)
        acc_dev.append(accuracy_score(Y_dev, Y_dev_pred))
    accuracy["Maximum depth"] = depths
    accuracy["Train accuracy"] = acc_train
    accuracy["Dev accuracy"] = acc_dev
    return accuracy


def test_results(X_train, Y_train, X_test, Y_test, best_parameter, target):
    model = Tree(best_parameter, target)
    model.fit(X_train, Y_train)
    Y_test_pred = model.predict(X_test)
    depth = [best_parameter]
    accuracy = [accuracy_score(Y_test, Y_test_pred)]
    test_result = {}
    test_result["Depth"] = depth
    test_result["Accuracy"] = accuracy
    return test_result


def plot(results, option1, option2, title):
    plt.plot(results["Maximum depth"], results[option1], label=option1)
    plt.plot(results["Maximum depth"], results[option2], label=option2)
    plt.title(title)
    plt.xlabel("Maximum depth")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left", fontsize="small")
    plt.show

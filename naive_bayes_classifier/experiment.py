from sklearn.model_selection import KFold
from network import NaiveBayesian
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def cross_validation(X, y, num_folds, network):
    train_accuracies = []
    valid_accuracies = []
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        network.fit(X_train, y_train)
        y_pred_train = network.predict(X_train)
        y_pred_val = network.predict(X_test)
        train_accuracy = np.mean(y_pred_train == y_train)
        train_accuracies.append(train_accuracy)
        valid_accuracy = np.mean(y_pred_val == y_test)
        valid_accuracies.append(valid_accuracy)
    mean_train_accuracy = np.mean(train_accuracies)
    mean_val_accuracy = np.mean(valid_accuracies)
    return mean_train_accuracy, mean_val_accuracy


def experiment(dataset, target, splits, mode):
    results = {}
    train_accuracies = []
    val_accuracies = []
    saved_params = []
    test_dataset = []
    network = NaiveBayesian()

    results["Test split"] = [split[0] for split in splits]
    if mode == "cross":
        results["Train split"] = [1 - split[0] for split in splits]
        results["N splits"] = [split[1] for split in splits]
    else:
        results["Train split"] = [(1 - split[0]) * (1 - split[1]) for split in splits]
        results["Val split"] = [split[1] * (1 - split[0]) for split in splits]

    for sample in splits:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.drop(columns=[target]),
            dataset[target],
            test_size=sample[0],
            random_state=42,
        )
        test_dataset.append((X_test, y_test))

        if mode == "cross":
            train_accuracy, val_accuracy = cross_validation(
                X_train, y_train, sample[1], network
            )
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            saved_params.append(network.get_parameters())

        else:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train,
                y_train,
                test_size=sample[1],
                random_state=42,
            )
            network.fit(X_train, y_train)
            predictions = network.predict(X_train)
            train_accuracy = accuracy_score(y_train, predictions)
            predictions = network.predict(X_val)
            val_accuracy = accuracy_score(y_val, predictions)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            saved_params.append(network.get_parameters())

    results["Train accuracy"] = train_accuracies
    results["Validation accuracy"] = val_accuracies
    return results, saved_params, test_dataset

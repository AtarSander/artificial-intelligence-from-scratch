from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    @abstractmethod
    def fit(self, X, y):
        """
        A method that fits the solver to the given data.
        X is the dataset without the class attribute.
        y contains the class attribute for each sample from X.
        It may return anything.
        """
        ...

    def predict(self, X):
        """
        A method that returns predicted class for each row of X
        """


class NaiveBayesian(Solver):
    def __init__(self, threshold=10):
        self.continuous_stats = {}
        self.categorical_probs = {}
        self.threshold = threshold

    def get_parameters(self):
        return self.categorical_probs, self.continuous_stats, self.priori, self.classes

    def set_parameters(self, cat_probs, cont_stats, priori, classes):
        self.continuous_stats = cont_stats
        self.categorical_probs = cat_probs
        self.priori = priori
        self.classes = classes

    def normal_distribution(self, X, mean, std):
        exponent = np.exp(-((X - mean) ** 2 / (2 * std**2)))

        return exponent / (np.sqrt(2 * np.pi) * std)

    def calculate_priori(self, X, y):
        self.priori = {}
        for label in np.unique(y):
            self.priori[label] = len(X[y == label]) / len(X)

    def select_categorical_continous(self, X):
        categorical_features = []
        continuous_features = []
        for feature in X.columns:
            if len(X[feature].unique()) > self.threshold:
                continuous_features.append(feature)
            else:
                categorical_features.append(feature)
        return categorical_features, continuous_features

    def fit(self, X, y):
        self.calculate_priori(X, y)
        categorical_features, continuous_features = self.select_categorical_continous(X)
        self.classes = np.unique(y)

        for feature in categorical_features:
            self.categorical_probs[feature] = {}
            for value in X[feature].unique():
                self.categorical_probs[feature][value] = {}
                for label in np.unique(y):
                    subset = X[y == label]
                    feature_prob = len(subset[subset[feature] == value]) / len(subset)
                    self.categorical_probs[feature][value][label] = feature_prob

        for feature in continuous_features:
            self.continuous_stats[feature] = {}
            for label in np.unique(y):
                subset = X[y == label][feature]
                self.continuous_stats[feature][label] = {
                    "mean": np.mean(subset),
                    "std": np.std(subset),
                }

    def predict(self, X):
        Y_pred = []
        for _, row in X.iterrows():
            class_probs = {}

            for feature in self.categorical_probs:
                for label in self.classes:
                    prob = (
                        self.categorical_probs[feature]
                        .get(row[feature], {})
                        .get(label, 0)
                    )
                    class_probs[label] = class_probs.get(label, 1) * prob

            for feature in self.continuous_stats:
                for label in self.classes:
                    stats = self.continuous_stats[feature][label]
                    prob = self.normal_distribution(
                        row[feature], stats["mean"], stats["std"]
                    )
                    class_probs[label] = class_probs.get(label, 1) * prob

            for label in self.classes:
                class_probs[label] *= self.priori[label]

            predicted_class = max(class_probs, key=class_probs.get)
            Y_pred.append(predicted_class)

        return Y_pred

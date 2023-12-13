from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from TreeNode import TreeNode


class Solver(ABC):
    """A solver. Parameters may be passed during initialization."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

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


class Tree(Solver):
    def __init__(self, max_depth, target):
        self.max_depth = max_depth
        self.target = target
        self.root = None

    def get_parameters(self):
        dict = {}
        dict["Maximum_depth"] = self.max_depth
        return dict

    def fit(self, X, y):
        dataset = pd.concat([X, y], axis=1)
        self.root = self.build_tree(dataset, 0)

    def build_tree(self, dataset, current_depth):
        if (
            self.check_leaf(dataset)
            or current_depth >= self.max_depth
            or dataset.shape[1] == 1
        ):
            leaf_value = self.classify(dataset)
            return TreeNode(value=leaf_value)

        d = self.max_inf_gain(dataset)
        labels = {}
        for value in dataset[d].unique():
            subdataset = dataset[dataset[d] == value]
            branch = self.build_tree(subdataset.drop(columns=d), current_depth + 1)
            labels[value] = branch

        return TreeNode(feature=d, labels=labels)

    def check_leaf(self, dataset):
        if len(np.unique(dataset[self.target])) <= 1:
            return True
        return False

    def classify(self, dataset):
        Y = dataset[self.target]
        classes, count_classes = np.unique(Y, return_counts=True)
        index = count_classes.argmax()
        return classes[index]

    def max_inf_gain(self, dataset):
        Y = dataset[self.target]
        X = dataset.drop(columns=self.target)
        max_feature = None
        max_entropy = -2
        for feature in X.columns:
            all_count = len(dataset[feature])
            all_values = Y.value_counts()
            values_count = (
                dataset.groupby(feature)[self.target]
                .value_counts()
                .unstack(fill_value=0)
            )
            all_entropy = self.entropy(all_values)
            part_entropies = []
            for _, counts in values_count.iterrows():
                div_val = sum(counts) / all_count
                part_entropies.append(div_val * self.entropy(counts))
            entropy = all_entropy - sum(part_entropies)
            if entropy > max_entropy:
                max_entropy = entropy
                max_feature = feature

        return max_feature

    def entropy(self, values):
        entropy = 0
        all = sum(values)
        for count in values:
            if count != 0:
                entropy -= (count / all) * (np.log2(count / all))
        return entropy

    def predict(self, X):
        predictions = []
        for _, row in X.iterrows():
            x = pd.DataFrame(row).transpose()
            predictions.append(self.make_prediction(x, self.root))
        return predictions

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value

        row_values = x.values[0]
        key = row_values[x.columns.get_loc(tree.feature)]
        if key in tree.labels.keys():
            return self.make_prediction(x, tree.labels[key])

        closest_key = min(tree.labels, key=lambda a: abs(a - key))
        return self.make_prediction(x, tree.labels[closest_key])

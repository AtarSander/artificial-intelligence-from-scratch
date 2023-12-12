from abc import ABC, abstractmethod
import numpy as np
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
    def __init__(self, min_split, max_depth, target):
        self.min_split = min_split
        self.max_depth = max_depth
        self.target = target
        self.root = None

    def get_parameters(self):
        dict = {}
        dict["Minimal_split"] = self.min_split
        dict["Maximum depth"] = self.max_depth
        return dict

    def fit(self, X, y):
        pass

    def build_tree(self, dataset, current_depth):
        if self.check_leaf(dataset) or current_depth < self.max_depth:
            leaf_value = self.classify(dataset)
            return TreeNode(value=leaf_value)

        d = self.max_inf_gain(self, dataset)
        for j in range(len(dataset[d].unique()) - 1):
            return self.build_tree(self, dataset.drop(columns=d), current_depth=current_depth-1)

    def check_leaf(self, dataset):
        X = dataset.drop(columns=self.target)
        if len(np.unique(X)) == 1:
            return True
        return False

    def classify(self, dataset):
        Y = dataset[:, self.target]
        classes, count_classes = np.unique(Y, return_counts=True)
        index = count_classes.argmax()
        return classes[index]

    def max_inf_gain(self, dataset):
        Y = dataset[:, self.target]
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
            all_entropy = self.entropy(all_count, all_values)
            part_entropies = []
            for _, counts in values_count.iterrows():
                part_entropies.append(sum(counts)/all_count * self.entropy(counts))
            entropy = all_entropy - sum(part_entropies)
            if entropy > max_entropy:
                max_entropy = entropy
                max_feature = feature

        return max_feature

    def entropy(self, values):
        entropy = 0
        all = sum(values)
        for _, count in values:
            entropy -= count/all * np.log2(count/all)
        return entropy


    def predict(self, X):
        pass

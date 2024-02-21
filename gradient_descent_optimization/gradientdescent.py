from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, x0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solution.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """


class GradientDescent(Solver):
    """
    Calculates gradient descent for given function, starting points and hyperparams

    :param step_size: learning rate
    :type: float

    :param iterations: how many times to iterate
    :type: int
    """
    def __init__(self, step_size, iterations):
        self.step_size = step_size
        self.iterations = iterations
        self.x_values = []
        self.y_values = []

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        dict = {}
        dict["Learning_rate"] = self.step_size
        dict["Iterations"] = self.iterations
        dict["X_values"] = self.x_values
        dict["Y_values"] = self.y_values
        return dict

    def set_learning_rate(self, new_step_size):
        """Sets new learning rate"""
        self.step_size = new_step_size

    def set_iterations(self, new_iter_num):
        """Sets new iteration number"""
        self.iterations = new_iter_num

    def solve(self, problem, x0, funct):
        """
        Calculates gradient descent and saves gradient descent steps values

        :param problem: gradient function
        :type: funct

        :param x0: starting point
        :type: np.array

        :param funct: function of which gradient is calculated
        :type: funct

        :return: final x,y values
        :type: tuple(np.array)
        """
        n_iter = self.iterations

        while n_iter > 0:
            self.x_values.append(x0)
            self.y_values.append(funct(x0))
            n_iter -= 1
            d = problem(x0)
            x0 = np.subtract(x0, d * self.step_size)
        return x0, funct(x0)

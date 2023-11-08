from abc import ABC, abstractmethod
import numpy as np


class Solver(ABC):
    """A solver. It may be initialized with some hyperparameters."""

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

    @abstractmethod
    def solve(self, problem, pop0, *args, **kwargs):
        """
        A method that solves the given problem for given initial solutions.
        It may accept or require additional parameters.
        Returns the solution and may return additional info.
        """
        ...


class Genetic(Solver):
    def get_parameters(self):
        pass

    def solve(self, problem, pop0, t_max, pc, pm):
        t = 0
        p = np.array([])
        p[0] = initialize(pop0)
        o = grade(problem, p)
        x_best, o_best = find_best(p, o)
        while t < t_max:
            s = selection(p[t])
            m = cross_mutate(s)

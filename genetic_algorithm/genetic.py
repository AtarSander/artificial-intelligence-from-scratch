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
    def __init__(self, pop_n=10, t_max=100, pc=0.95, pm=0.05):
        self.update_params(pop_n, t_max, pc, pm)
        self.best_grades = []
        self.best_solutions = []

    def update_params(self, pop_n, t_max, pc, pm):
        self.pop_n = pop_n
        self.t_max = t_max
        self.pc = pc
        self.pm = pm

    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        dict = {}
        dict["Individuals_number"] = self.pop_n
        dict["Iteration_number"] = self.t_max
        dict["Crossover_rate"] = self.pc
        dict["Mutation_rate"] = self.pm
        return dict

    def get_best_grades(self):
        return self.best_grades

    def get_best_solution(self):
        return self.best_solutions

    def solve(self, problem, pop_n, t_max, pc, pm, dim=200):
        self.update_params(pop_n, t_max, pc, pm)
        t = 0
        p = [0] * (t_max+1)
        g = [0] * (t_max+1)
        x = [0] * t_max
        c = [0] * t_max
        p[0] = self.initialize(pop_n, dim)
        g[0] = self.grade(problem, p[0])
        x_best, g_best = self.find_best(p[0], g[0])

        while t < t_max:
            s = self.selection(p[t], g[t])
            m = self.cross_mutate(s, pc, pm)
            g[t+1] = self.grade(problem, m)
            x[t], c[t] = self.find_best(m, g[t+1])
            if c[t] > g_best:
                x_best = x[t]
                g_best = c[t]
            p[t+1] = m
            t += 1

        self.best_grades = c
        self.best_solutions = x
        return x_best, g_best

    def initialize(self, pop_n, dim):
        population = []
        for _ in range(pop_n):
            random_array = np.random.randint(2, size=dim)
            population.append(random_array)
        return np.array(population)

    def grade(self, target_funct, pop, C=3000):
        grades = []
        for individual in pop:
            grades.append(target_funct(individual)+C)
        return np.array(grades)

    def find_best(self, pop, grades, C=3000):
        g_best = np.max(grades)
        x_best = pop[np.argmax(grades)]
        return x_best, g_best-C

    def selection(self, pop, g):
        prob = []
        selected = []
        prob_sum = np.sum(g)
        for target_val in g:
            prob.append(target_val/prob_sum)
        indices = np.random.choice(len(pop), size=len(prob), p=prob)
        selected = [pop[i] for i in indices]

        return selected

    def cross_mutate(self, s, pc, pm):
        m = []
        if len(s) % 2 == 1:
            s.append(s[0])
        s = np.array(s)
        for i in range(0, len(s), 2):
            cross_rand = np.random.rand()
            child1 = s[i]
            child2 = s[i+1]

            if pc > cross_rand:
                parent1 = s[i]
                parent2 = s[i+1]
                split = np.random.randint(len(s[i]))
                child1 = np.concatenate((parent1[:split], parent2[split:]),
                                        axis=0)
                child2 = np.concatenate((parent2[:split], parent1[split:]),
                                        axis=0)

            for j in range(len(s[i])):
                mut_rand = np.random.rand()
                if pm > mut_rand:
                    child1[j] = child1[j] ^ 1
                mut_rand = np.random.rand()
                if pm > mut_rand:
                    child2[j] = child2[j] ^ 1

            m.append(child1)
            m.append(child2)
        return np.array(m)

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
        """Returns a dictionary of hyperparameters"""
        dict = {}
        dict["Iteration_number"] = self.step_size
        dict["Crossover_rate"] = self.iterations
        dict["Mutation_rate"] = self.x_values
        dict["Individuals_number"] = self.y_values
        return dict

    def solve(self, problem, pop_n, t_max, pc, pm, dim=200):
        t = 0
        p = [0]*t_max
        o = [0]*t_max
        p[0] = self.initialize(pop_n, dim)
        o[0] = self.grade(problem, p[0])
        x_best, o_best = self.find_best(p[0], o[0])
        while t < t_max-1:
            s = self.selection(p[t], o[t])
            m = self.cross_mutate(s, pc, pm)
            o[t+1] = self.grade(problem, m)
            x_t, o_t = self.find_best(m, o[t+1])
            if o_t > o_best:
                x_best = x_t
                o_best = o_t
            p[t+1] = m
            t += 1
        return x_best, o_best

    def initialize(self, pop_n, dim):
        population = []
        for _ in range(pop_n):
            random_array = np.random.randint(2, size=dim)
            population.append(random_array)
        return np.array(population)

    def grade(self, target_funct, pop):
        g = []
        for i in pop:
            g.append(target_funct(i)+3000)
        return np.array(g)

    def find_best(self, pop, o):
        o_best = np.max(o)
        x_best = pop[np.argmax(o)]
        return x_best, o_best

    def selection(self, pop, o):
        prob = []
        s = []
        prob_sum = np.sum(o)
        for target_val in o:
            prob.append(target_val/prob_sum)
        # for _ in pop:
        #     value = np.random.choice(pop, p=prob)
        #     s.append(value)
        indices = np.random.choice(len(pop), size=len(prob), p=prob)

        # Select arrays based on the chosen indices
        s = [pop[i] for i in indices]

        return np.array(s)

    def cross_mutate(self, s, pc, pm):
        m = []
        for i in range(0, len(s), 2):
            cross_rand = np.random.rand()
            child1 = s[i]
            child2 = s[i+1]

            if pc > cross_rand:
                parent1 = s[i]
                parent2 = s[i+1]
                split = np.random.randint(len(s[i]))
                child1 = np.concatenate((parent1[:split], parent2[split:]), axis=0)
                child2 = np.concatenate((parent2[:split], parent1[split:]), axis=0)
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

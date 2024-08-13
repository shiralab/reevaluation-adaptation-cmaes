#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..objective_function.base import *
import numpy as np


# public symbols
__all__ = ['initial_setting_for_gaussian', 'Sphere', 'Rosenbrock', 'Ellipsoid', 'kTablet', 'Cigar',
           'Bohachevsky', 'Ackley', 'Schaffer', 'Rastrigin']


def initial_setting_for_gaussian(func_instance, random=True):
    """
    Return random initial vector within the range or constant initial vector.

    :type func_instance: object
    :type random: bool
    :return: initial mean vector
    :rtype: array_like, shape=(d), dtype=float
    :return: initial sigma
    :rtype: float
    """
    if isinstance(func_instance, Sphere) or isinstance(func_instance, Ellipsoid) or isinstance(func_instance, kTablet)\
            or isinstance(func_instance, Cigar) or isinstance(func_instance, Rastrigin):
        a, b = 1., 5.
    elif isinstance(func_instance, Rosenbrock):
        a, b = -2., 2.
    elif isinstance(func_instance, Bohachevsky):
        a, b = 1., 15.
    elif isinstance(func_instance, Ackley):
        a, b = 1., 30.
    elif isinstance(func_instance, Schaffer):
        a, b = 10., 100.
    elif isinstance(func_instance, Griewank):
        a, b = 10., 600.
    else:
        a, b = 1., 5.

    return (b - a) * np.random.rand(func_instance.d) + a if random else (b + a) / 2. * np.ones(func_instance.d), (b - a) / 2.


class Sphere(ObjectiveFunction):
    """
    Sphere function: :math:`f(x) = \\sum_{i=1}^d x_i^2`
    """
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Sphere, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        evals = (X**2).sum(axis=1)
        return evals


class Rosenbrock(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        """
        Rosenbrock function
        """
        super(Rosenbrock, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        evals = np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1.)**2, axis=1)
        return evals


class Ellipsoid(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Ellipsoid, self).__init__(target_eval, max_eval)
        self.d = d
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))

    def evaluation(self, X):
        tmp = X * self.coefficient
        evals = (tmp**2).sum(axis=1)
        return evals


class kTablet(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, k=None, target_eval=1e-10, max_eval=1e4):
        super(kTablet, self).__init__(target_eval, max_eval)
        self.d = d
        self.k = int(d / 4) if k is None else k

    def evaluation(self, X):
        evals = (X[:, :self.k]**2).sum(axis=1) + 100 * 100 * (X[:, self.k:]**2).sum(axis=1)
        return evals


class Cigar(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Cigar, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        evals = (X[:, 0]**2) + 1000000 * (X[:, 1:]**2).sum(axis=1)
        return evals


class Bohachevsky(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Bohachevsky, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        evals = (X[:, :-1]**2 + 2. * X[:, 1:]**2 - 0.3 * np.cos(3. * np.pi * X[:, :-1])
                 - 0.4 * np.cos(4. * np.pi * X[:, 1:]) + 0.7).sum(axis=1)
        return evals


class Ackley(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Ackley, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        tmp1 = (X**2).sum(axis=1)
        tmp2 = np.cos(2. * np.pi * X).sum(axis=1)
        evals = 20. - 20. * np.exp(-0.2 * np.sqrt(tmp1 / self.d)) + np.exp(1.0) - np.exp(tmp2 / self.d)
        return evals


class Schaffer(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Schaffer, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        tmp1 = X[:, :-1]**2 + X[:, 1:]**2
        tmp2 = np.sin(50. * tmp1**(0.1))
        evals = np.sum(tmp1**(0.25) * (tmp2**2 + 1.), axis=1)
        return evals


class Rastrigin(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Rastrigin, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        evals = (X ** 2).sum(axis=1) - 10. * np.cos(2. * np.pi * X).sum(axis=1) + 10. * self.d
        return evals


class Griewank(ObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, target_eval=1e-10, max_eval=1e4):
        super(Griewank, self).__init__(target_eval, max_eval)
        self.d = d

    def evaluation(self, X):
        term_one = (X ** 2).sum(axis=1) / 4000.0
        term_two = np.prod([np.cos(X[:, i] / np.sqrt(i + 1)) for i in range(self.d)], axis=0)
        evals = term_one - term_two + 1.0
        return evals


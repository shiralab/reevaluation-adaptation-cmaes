#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..objective_function.base import *
from abc import ABCMeta, abstractmethod
import numpy as np

# public symbols
__all__ = ['NoiseWrapper', 'GaussianNoise']

class NoiseWrapper(ObjectiveFunction):
    
    def __init__(self, objective_function, noise_function, target_values_min=None, target_values_max=None):
        super().__init__(objective_function.target_eval, objective_function.max_eval)
        self.objective_function = objective_function
        self.noise_function = noise_function
        self.d = objective_function.d
        
        self.target_values_min = target_values_min if target_values_min is not None else 1e-3
        self.target_values_max = target_values_max if target_values_max is not None else 1e6
        self.target_values_num = 500
        self.target_values = np.geomspace(self.target_values_min, self.target_values_max, self.target_values_num)
        
        self.best_reached_target_num = 0
        self.save_log = True

    def __call__(self, X):
        self.eval_count += len(X)
        evals = self.objective_function(X)
        self._update_best_eval(evals)

        return self.noise_function(X, evals)
    
    def set_target_values(self, target_max = None, target_min = None):
        if target_max is not None:
            self.target_values_max = target_max 
        if target_min is not None:
            self.target_values_min = target_min
        self.target_values = np.geomspace(self.target_values_min, self.target_values_max, self.target_values_num)
        
    
    @staticmethod
    def info_header():
        return ObjectiveFunction.info_header() + ["reached_target_num"]

    def info_list(self):
        return super().info_list() + ['%d' % self.best_reached_target_num]
    
    def request_save_log(self):
        if self.save_log:
            self.save_log = False
            return True
        else:
            return False
    
    def evaluation(self, X):
        return self.objective_function.evaluation(X)

    def compute_reached_target(self):
        if self.objective_function.minimization_problem:
            reached_target_num = int(np.sum(self.mean_best_eval < self.target_values)) 
        else:
            reached_target_num = int(np.sum(self.mean_best_eval > self.target_values)) 
        
        if self.best_reached_target_num < reached_target_num:
            self.save_log = True
            self.best_reached_target_num = reached_target_num
    
    def _update_mean_best_eval(self, mean_eval):
        self.mean_best_eval = self.get_better(mean_eval, self.mean_best_eval)
        self.compute_reached_target()
    
    def is_success(self):
        if self.is_better_eq(self.mean_best_eval, self.target_eval):
            return True
        else:
            return False
    
    def verbose_display(self):
        return ' EvalCount: %d' % self.eval_count + ' MeanBestEval: {}'.format(self.mean_best_eval)


class AdditiveGaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, evals):
        return evals + np.random.randn(len(X)) * self.sigma


class MultiplicativeUniformNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, evals):
        u = (np.random.rand(len(X)) - 0.5) * 2
        return evals * (1 + u * self.sigma)
    

class MultiplicativeGaussianNoise(object):
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, X, evals):
        return evals * (1 + np.random.randn(len(X)) * self.sigma)
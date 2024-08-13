#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

# public symbols
__all__ = ['BaseOptimizer']


class BaseOptimizer(object):
    """
    Base class for information geometric util
    """
    
    twice_eval_num = 0
    num_reeval = 1
    sampler = None

    @abstractmethod
    def sampling_model(self):
        pass

    @abstractmethod
    def update(self, X, evals):
        """
        Abstract method for parameter updating.
        """
        pass

    def terminate_condition(self):
        """
        Check terminate condition.

        :return bool: terminate condition is satisfied or not
        """
        return False

    def verbose_display(self):
        """
        Return verbose display string.

        :return str: string for verbose display
        """
        return ''

    def log_header(self):
        """
        Return log header list.

        :return: header info list for log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return log string list.

        :return: log string list
        :rtype string list:
        """
        return []

    def run(self, sampler, logger=None, verbose=True, log_interval=1):
        """
        Running script of Information Geometric Optimization (IGO)
        """

        self.sampler = sampler
        f = sampler.f
        if logger is not None:
            logger.write_csv(['Ite'] + ['mean_eval'] + f.info_header() + self.log_header() + sampler.log_header())

        ite = 0

        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1

            # sampling and evaluation
            X = self.sampling_model().sampling(sampler.lam)
            X_tile = np.tile(X, (self.num_reeval, 1))
            evals = sampler.f(X_tile).reshape((self.num_reeval, -1))
            
            if self.num_reeval == 1:
                if self.twice_eval_num > 0:
                    evals_second = np.full(evals.shape, np.nan)
                    X_second_tile = np.tile(X[:self.twice_eval_num], (self.num_reeval, 1))
                    evals_second[:, :self.twice_eval_num] = sampler.f(X_second_tile).reshape((self.num_reeval, -1))
                    evals = np.vstack((evals, evals_second))
                else:
                    evals = evals[0,:]
            
            # parameter update
            self.update(X, evals)

            # for PSA
            sampler.lam = self.lam
            
            # evaluate mean vector
            true_eval_mean = sampler.f.evaluation(self.model.m[None,:])[0]
            sampler.f._update_mean_best_eval(true_eval_mean)

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display())
            if logger is not None and (
                ite % log_interval == 0 or 
                sampler.f.request_save_log() or
                sampler.f.terminate_condition() or 
                self.terminate_condition()
            ):
                logger.write_csv([str(ite)] + ['%e' % true_eval_mean] + f.info_list() + self.log() + sampler.log())

        return [f.eval_count, f.best_eval, f.is_success()]

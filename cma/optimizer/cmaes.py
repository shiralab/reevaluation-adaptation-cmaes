#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg

from ..util.weight import CMAWeight
from ..optimizer.base_optimizer import BaseOptimizer
from ..util.model import GaussianSigmaC

# public symbols
__all__ = ['CMAParam', 'CMAES']


class CMAParam(object):
    """
    Default parameters for CMA-ES.
    """
    @staticmethod
    def pop_size(dim):
        return 4 + int(np.floor(3 * np.log(dim)))

    @staticmethod
    def mu_eff(lam, weights=None):
        if weights is None and lam < 4:
            weights = CMAWeight(4).w
        if weights is None:
            weights = CMAWeight(lam).w
        w_1 = np.absolute(weights).sum()
        return w_1**2 / weights.dot(weights)

    @staticmethod
    def c_1(dim, mueff):
        return 2.0 / ((dim + 1.3) * (dim + 1.3) + mueff)

    @staticmethod
    def c_mu(dim, mueff, c1=0., alpha_mu=2.):
        return np.minimum(1. - c1, alpha_mu * (mueff - 2. + 1./mueff) / ((dim + 2.)**2 + alpha_mu * mueff / 2.))

    @staticmethod
    def c_c(dim, mueff):
        return (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

    @staticmethod
    def c_sigma(dim, mueff):
        return (mueff + 2.0) / (dim + mueff + 5.0)

    @staticmethod
    def damping(dim, mueff):
        return 1.0 + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + CMAParam.c_sigma(dim, mueff)

    @staticmethod
    def chi_d(dim):
        return np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))  # ||N(0,I)||


class CMAES(BaseOptimizer):
    def __init__(self, d, weight_func, m=None, C=None, sigma=1., minimal_eigenval=1e-30,
                 lam=None, c_m=1., c_1=None, c_mu=None, c_c=None, c_sigma=None, damping=None, alpha_mu=2.):
        self.model = GaussianSigmaC(d, m=m, C=C, sigma=sigma, minimal_eigenval=minimal_eigenval)
        self.weight_func = weight_func
        self.lam = lam if lam is not None else CMAParam.pop_size(d)

        # CMA parameters
        self.mu_eff = CMAParam.mu_eff(self.lam)
        self.c_1 = CMAParam.c_1(d, self.mu_eff) if c_1 is None else c_1
        self.c_mu = CMAParam.c_mu(d, self.mu_eff, c1=self.c_1, alpha_mu=alpha_mu) if c_mu is None else c_mu
        self.c_c = CMAParam.c_c(d, self.mu_eff) if c_c is None else c_c
        self.c_sigma = CMAParam.c_sigma(d, self.mu_eff) if c_sigma is None else c_sigma
        self.damping = CMAParam.damping(d, self.mu_eff) if damping is None else damping
        self.chi_d = CMAParam.chi_d(d)
        self.c_m = c_m

        # evolution path
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen_count = 0

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        self.gen_count += 1

        # natural gradient
        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        WYT = weights * Y.T
        m_grad = self.model.sigma * WYT.sum(axis=1)
        C_grad = np.dot(WYT, Y) - weights.sum() * self.model.C

        hsig = 1.
        if self.c_1 != 0. or self.damping != np.inf:
            # evolution path
            self.ps = (1.0 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * np.dot(self.model.invSqrtC, self.c_m * m_grad / self.model.sigma)
            hsig = 1. if scipy.linalg.norm(self.ps) / (np.sqrt(1. - (1. - self.c_sigma) ** (2 * self.gen_count))) < (1.4 + 2. / (self.model.d + 1.)) * self.chi_d else 0.
            self.pc = (1. - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * self.c_m * m_grad / self.model.sigma

        if self.damping != np.inf:
            # CSA
            self.model.sigma = self.model.sigma * np.exp(self.c_sigma / self.damping * (scipy.linalg.norm(self.ps) / self.chi_d - 1.))
        # mean vector update
        self.model.m = self.model.m + self.c_m * m_grad
        # covariance matrix update
        self.model.C = self.model.C + (1.-hsig)*self.c_1*self.c_c*(2.-self.c_c)*self.model.C + self.c_1 * (np.outer(self.pc, self.pc) - self.model.C) + self.c_mu * C_grad

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display()

    def log_header(self):
        return self.model.log_header()

    def log(self):
        return self.model.log()





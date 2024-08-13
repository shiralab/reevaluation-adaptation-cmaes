#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg
from .cmaes import CMAES

"""
    This code is based on the source code of LRA-CMA-ES.
    https://github.com/nomuramasahir0/cma-learning-rate-adaptation
"""
class LRACMAES(CMAES):
    def __init__(self, d, weight_func, **kwargs):

        CMAES.__init__(self, d, weight_func, **kwargs)
        
        self.d = d
        
        self.alpha = 1.4
        self.beta_mean = 0.1
        self.beta_Sigma = 0.03
        self.gamma = 0.1

        self.Emean = np.zeros([self.d, 1])
        self.ESigma = np.zeros([self.d * self.d, 1])
        self.Vmean = 0
        self.VSigma = 0
        self.eta_mean = 1.0
        self.eta_Sigma = 1.0
        
    def compute_updated_parameter(self, X, evals, is_update_evolution_path=False):
        # natural gradient
        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        WYT = weights * Y.T
        m_grad = self.model.sigma * WYT.sum(axis=1)
        C_grad = np.dot(WYT, Y) - weights.sum() * self.model.C
        
        # evolution path
        updated_ps = (1. - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2. - self.c_sigma) * self.mu_eff) * np.dot(self.model.invSqrtC, m_grad / self.model.sigma)
        
        if (np.linalg.norm(updated_ps) ** 2) / (1 - (1 - self.c_sigma) ** (2 * (self.gen_count + 1))) < (2.0 + 4.0 / (self.d + 1)) * self.d:
            hsig = 1
        else:
            hsig = 0
        
        updated_pc = (1. - self.c_c) * self.pc + np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * m_grad / self.model.sigma
            
        # CSA
        updated_sigma = self.model.sigma * np.exp(min(1.0, self.c_sigma / self.damping * (scipy.linalg.norm(updated_ps) / self.chi_d - 1.)))
        
        # mean vector update
        updated_m = self.model.m + self.c_m * m_grad
        
        # covariance matrix update
        updated_C = self.model.C + (1.-hsig)*self.c_1*self.c_c*(2.-self.c_c)*self.model.C + self.c_1 * (np.outer(updated_pc, updated_pc) - self.model.C) + self.c_mu * C_grad
        
        if is_update_evolution_path:
            self.ps = updated_ps
            self.pc = updated_pc
        
        return updated_m, updated_C, updated_sigma
        
    def update(self, X, evals):
        self.gen_count += 1

        # save parameters before update
        old_mean = np.copy(self.model.m)
        old_Sigma = np.copy((self.model.sigma ** 2) * self.model.C)
        old_inv_sqrtSigma = np.copy(self.model.invSqrtC / self.model.sigma)

        updated_m, updated_C, updated_sigma = self.compute_updated_parameter(X, evals, is_update_evolution_path=True)

        # local coordinate
        Deltamean = updated_m - old_mean
        DeltaSigma = (updated_sigma ** 2) * updated_C - old_Sigma
        
        locDeltamean = old_inv_sqrtSigma.dot(Deltamean).reshape(self.d, 1)
        locDeltaSigma = (
            old_inv_sqrtSigma.dot(DeltaSigma.dot(old_inv_sqrtSigma))
        ).reshape(self.d * self.d, 1) / np.sqrt(2)

        # moving average E and V
        self.Emean = (1 - self.beta_mean) * self.Emean + self.beta_mean * locDeltamean
        self.ESigma = (1 - self.beta_Sigma) * self.ESigma + self.beta_Sigma * locDeltaSigma
        
        self.Vmean = (1 - self.beta_mean) * self.Vmean + self.beta_mean * (
            np.linalg.norm(locDeltamean) ** 2
        )
        self.VSigma = (1 - self.beta_Sigma) * self.VSigma + self.beta_Sigma * (
            np.linalg.norm(locDeltaSigma) ** 2
        )
        
        # estimate SNR
        sqnormEmean = np.linalg.norm(self.Emean) ** 2
        hatSNRmean = (sqnormEmean - (self.beta_mean / (2 - self.beta_mean)) * self.Vmean) / (
            self.Vmean - sqnormEmean
        )

        sqnormESigma = np.linalg.norm(self.ESigma) ** 2
        hatSNRSigma = (sqnormESigma - (self.beta_Sigma / (2 - self.beta_Sigma)) * self.VSigma) / (
            self.VSigma - sqnormESigma
        )
        
        before_eta_mean = self.eta_mean
        relativeSNRmean = np.clip((hatSNRmean / self.alpha / self.eta_mean) - 1, -1, 1)
        self.eta_mean = self.eta_mean * np.exp(
            min(self.gamma * self.eta_mean, self.beta_mean) * relativeSNRmean
        )
        relativeSNRSigma = np.clip((hatSNRSigma / self.alpha / self.eta_Sigma) - 1, -1, 1)
        self.eta_Sigma = self.eta_Sigma * np.exp(
            min(self.gamma * self.eta_Sigma, self.beta_Sigma) * relativeSNRSigma
        )
        # cap
        self.eta_mean = min(self.eta_mean, 1.0) 
        self.eta_Sigma = min(self.eta_Sigma, 1.0)
        
        self.model.m = old_mean + self.eta_mean * Deltamean
        Sigma = old_Sigma + self.eta_Sigma * DeltaSigma
        
        # decompose Sigma to sigma and C
        eigs, _ = np.linalg.eigh(Sigma)
        logeigsum = sum([np.log(e) for e in eigs])
        self.model.sigma = np.exp(logeigsum / 2.0 / self.d)
        
        if self.model.sigma == 0:
            print("sigma is 0, so the optimization ends. ")
            return
        self.model.C = Sigma / (self.model.sigma ** 2)
        
        # step-size correction
        self.model.sigma *= before_eta_mean / self.eta_mean
    
    def log_header(self):
        return self.model.log_header() + ['eta_mean'] + ['eta_Sigma'] 

    def log(self):
        return self.model.log() + ['%e' % self.eta_mean] + ['%e' % self.eta_Sigma]




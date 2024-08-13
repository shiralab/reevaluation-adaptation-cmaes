#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import rankdata
import scipy.linalg

from abc import abstractmethod
from .lra_cmaes import LRACMAES
from scipy.stats import norm

def fpr(x):
    return int(np.floor(x) + int( np.random.rand() < (x - np.floor(x)) ))


class RACMAES(LRACMAES):
    '''
    Adaptation of Num. of Reevaluation using SNR
    - if eval_num == 1:
        - set next_eval_num = 2 for each reeval_zero_ite_max iteration
        - if the ranking of solutions changes, set eval_num to 2.
    - otherwise:
        - compute the update direction using next_eval_num reevaluation
        - compute the update direction using half of next_eval_num reevaluation
        - compute SNRs of update directions
        - If SNRs differ significantly, increase eval_num
    '''

    def __init__(self, d, weight_func, **kwargs):
        
        LRACMAES.__init__(self, d, weight_func, **kwargs)
        
        
        self.Emean_half1 = np.zeros([self.d, 1])
        self.Vmean_half1 = 0
        self.ESigma_half1 = np.zeros([self.d * self.d, 1])
        self.VSigma_half1 = 0
        
        self.Emean_half2 = np.zeros([self.d, 1])
        self.Vmean_half2 = 0
        self.ESigma_half2 = np.zeros([self.d * self.d, 1])
        self.VSigma_half2 = 0
        
        self.Vmean_cov = 0
        self.VSigma_cov = 0
        
        self.target_corr_base = 0.8
        self.eval_num_min = 1.2
        self.eval_num = self.eval_num_min
        
        self.target_corr_lra_coef = 1
        
    def compute_updated_parameter(self, X, evals, is_update_evolution_path=False):
        # natural gradient
        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        WYT = weights * Y.T
        m_grad = self.model.sigma * WYT.sum(axis=1)
        C_grad = np.dot(WYT, Y) - weights.sum() * self.model.C
        
        # evolution path
        updated_ps = (1. - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2. - self.c_sigma) * self.mu_eff) * np.dot(self.model.invSqrtC, self.c_m * m_grad / self.model.sigma)
        
        if (np.linalg.norm(updated_ps) ** 2) / (1 - (1 - self.c_sigma) ** (2 * (self.gen_count + 1))) < (2.0 + 4.0 / (self.d + 1)) * self.d:
            hsig = 1
        else:
            hsig = 0
        
        updated_pc = (1. - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * self.c_m * m_grad / self.model.sigma
            
        # CSA
        updated_sigma = self.model.sigma * np.exp(self.c_sigma / self.damping * (scipy.linalg.norm(updated_ps) / self.chi_d - 1.))
        
        # mean vector update
        updated_m = self.model.m + self.c_m * m_grad
        
        # covariance matrix update
        updated_C = self.model.C + (1.-hsig)*self.c_1*self.c_c*(2.-self.c_c)*self.model.C + self.c_1 * (np.outer(updated_pc, updated_pc) - self.model.C) + self.c_mu * C_grad
        
        if is_update_evolution_path:
            self.ps = updated_ps
            self.pc = updated_pc
        
        return updated_m, updated_C, updated_sigma

    def compute_target_corr(self):
        exponent = np.ones(2)
        exponent *= ((np.log(self.eval_num) - np.log(self.eval_num_min)) + 1)

        p = np.min([self.eval_num - 1, 1])
        
        return self.target_corr_base ** (p * exponent)
    
    def update_cov_accumulation(self, Deltamean_half1, DeltaSigma_half1, Deltamean_half2, DeltaSigma_half2):

        coef_mean = self.beta_mean #* self.eta_mean # / self.eval_num
        coef_Sigma = self.beta_Sigma #* self.eta_Sigma # / self.eval_num
        
        old_inv_sqrtSigma = np.copy(self.model.invSqrtC / self.model.sigma)
        
        locDeltamean_half1 = old_inv_sqrtSigma.dot(Deltamean_half1).reshape(self.d, 1)
        locDeltaSigma_half1 = (
            old_inv_sqrtSigma.dot(DeltaSigma_half1.dot(old_inv_sqrtSigma))
        ).reshape(self.d * self.d, 1) / np.sqrt(2)
        
        locDeltamean_half2 = old_inv_sqrtSigma.dot(Deltamean_half2).reshape(self.d, 1)
        locDeltaSigma_half2 = (
            old_inv_sqrtSigma.dot(DeltaSigma_half2.dot(old_inv_sqrtSigma))
        ).reshape(self.d * self.d, 1) / np.sqrt(2)

        # moving average E and V
        self.Emean_half1 = (1 - coef_mean) * self.Emean_half1 + coef_mean * locDeltamean_half1
        self.ESigma_half1 = (1 - coef_Sigma) * self.ESigma_half1 + coef_Sigma * locDeltaSigma_half1
        
        self.Emean_half2 = (1 - coef_mean) * self.Emean_half2 + coef_mean * locDeltamean_half2
        self.ESigma_half2 = (1 - coef_Sigma) * self.ESigma_half2 + coef_Sigma * locDeltaSigma_half2
        
        self.Vmean_half1 = (1 - coef_mean) * self.Vmean_half1 + coef_mean * (
            np.linalg.norm(locDeltamean_half1) ** 2
        )
        self.VSigma_half1 = (1 - coef_Sigma) * self.VSigma_half1 + coef_Sigma * (
            np.linalg.norm(locDeltaSigma_half1) ** 2
        )
        self.Vmean_half2 = (1 - coef_mean) * self.Vmean_half2 + coef_mean * (
            np.linalg.norm(locDeltamean_half2) ** 2
        )
        self.VSigma_half2 = (1 - coef_Sigma) * self.VSigma_half2 + coef_Sigma * (
            np.linalg.norm(locDeltaSigma_half2) ** 2
        )
        
        self.Vmean_cov = (1 - coef_mean) * self.Vmean_cov + coef_mean * (
            np.dot(locDeltamean_half2.T, locDeltamean_half1)
        )
        self.VSigma_cov = (1 - coef_Sigma) * self.VSigma_cov + coef_Sigma * (
            np.dot(locDeltaSigma_half2.T, locDeltaSigma_half1)
        )
    
    def compute_corr(self):
        
        coef_mean = self.beta_mean #* self.eta_mean # / self.eval_num
        coef_Sigma = self.beta_Sigma #* self.eta_Sigma # / self.eval_num
        
        sqnormEmean_half1 = np.linalg.norm(self.Emean_half1) ** 2
        sqnormEmean_half2 = np.linalg.norm(self.Emean_half2) ** 2
        sqnormEmean_cov = np.dot(self.Emean_half1.T, self.Emean_half2)
        
        mean_cov_half1 = (self.Vmean_half1 - sqnormEmean_half1) / (1. - coef_mean / (2 - coef_mean))
        mean_cov_half2 = (self.Vmean_half2 - sqnormEmean_half2) / (1. - coef_mean / (2 - coef_mean))
        mean_cov = (self.Vmean_cov - sqnormEmean_cov) / (1. - coef_mean / (2 - coef_mean))

        sqnormESigma_half1 = np.linalg.norm(self.ESigma_half1) ** 2
        sqnormESigma_half2 = np.linalg.norm(self.ESigma_half2) ** 2
        sqnormESigma_cov = np.dot(self.ESigma_half1.T, self.ESigma_half2)
        
        Sigma_cov_half1 = (self.VSigma_half1 - sqnormESigma_half1) / (1. - coef_Sigma / (2 - coef_Sigma))
        Sigma_cov_half2 = (self.VSigma_half2 - sqnormESigma_half2) / (1. - coef_Sigma / (2 - coef_Sigma))
        Sigma_cov = (self.VSigma_cov - sqnormESigma_cov) / (1. - coef_Sigma / (2 - coef_Sigma))
        
        corr_mean = mean_cov / np.sqrt(mean_cov_half1 * mean_cov_half2)
        corr_Sigma = Sigma_cov / np.sqrt(Sigma_cov_half1 * Sigma_cov_half2)
        
        return float(corr_mean), float(corr_Sigma)
    
    def update_accumulation(self, Emean, ESigma, Vmean, VSigma, Deltamean, DeltaSigma):
        
        old_inv_sqrtSigma = np.copy(self.model.invSqrtC / self.model.sigma)
        
        locDeltamean = old_inv_sqrtSigma.dot(Deltamean).reshape(self.d, 1)
        locDeltaSigma = (
            old_inv_sqrtSigma.dot(DeltaSigma.dot(old_inv_sqrtSigma))
        ).reshape(self.d * self.d, 1) / np.sqrt(2)

        # moving average E and V
        Emean = (1 - self.beta_mean) * Emean + self.beta_mean * locDeltamean
        ESigma = (1 - self.beta_Sigma) * ESigma + self.beta_Sigma * locDeltaSigma
        
        Vmean = (1 - self.beta_mean) * Vmean + self.beta_mean * (
            np.linalg.norm(locDeltamean) ** 2
        )
        VSigma = (1 - self.beta_Sigma) * VSigma + self.beta_Sigma * (
            np.linalg.norm(locDeltaSigma) ** 2
        )
        
        return Emean, ESigma, Vmean, VSigma
    
    def compute_snr(self, Emean, ESigma, Vmean, VSigma):
        sqnormEmean = np.linalg.norm(Emean) ** 2
        hatSNRmean = (sqnormEmean - (self.beta_mean / (2 - self.beta_mean)) * Vmean) / (
            Vmean - sqnormEmean
        )

        sqnormESigma = np.linalg.norm(ESigma) ** 2
        hatSNRSigma = (sqnormESigma - (self.beta_Sigma / (2 - self.beta_Sigma)) * VSigma) / (
            VSigma - sqnormESigma
        )
        
        return hatSNRmean, hatSNRSigma
    
    def compute_cov_snr(self):

        sqnormEmean_cov = np.dot(self.Emean_half1.T, self.Emean_half2)
        
        mean_cov = (self.Vmean_cov - sqnormEmean_cov) / (1. - self.beta_mean / (2 - self.beta_mean))
        mean_signal = (self.Vmean_cov - sqnormEmean_cov * (2 - self.beta_mean) / self.beta_mean) / (2 - 2 * self.beta_mean) / self.beta_mean

        sqnormESigma_cov = np.dot(self.ESigma_half1.T, self.ESigma_half2)
        
        Sigma_cov = (self.VSigma_cov - sqnormESigma_cov) / (1. - self.beta_Sigma / (2 - self.beta_Sigma))
        Sigma_signal = (self.VSigma_cov - sqnormESigma_cov * (2 - self.beta_Sigma) / self.beta_Sigma) / (2 - 2 * self.beta_Sigma) / self.beta_Sigma
        
        cov_snr_mean = mean_signal / mean_cov 
        cov_snr_Sigma = Sigma_signal / Sigma_cov
        
        return float(cov_snr_mean), float(cov_snr_Sigma)

    def run(self, sampler, logger=None, verbose=True, log_interval=1):

        f = sampler.f
        if logger is not None:
            logger.write_csv(['Ite'] + ['mean_eval'] + f.info_header() + self.log_header() + sampler.log_header())

        ite = 0

        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1
            
            eval_num = fpr(self.eval_num)

            # sampling and evaluation
            X = self.sampling_model().sampling(sampler.lam)
            X_tile = np.tile(X, (eval_num, 1))
            evals_set = sampler.f(X_tile).reshape((eval_num,-1))

            # parameter update
            self.update(X, evals_set)
            
            # evaluate mean vector
            true_eval_mean = sampler.f.evaluation(self.model.m[None,:])[0]
            sampler.f._update_mean_best_eval(true_eval_mean)

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display() + " N_eval:" + str(eval_num))
            if logger is not None and (
                ite % log_interval == 0 or 
                ite < log_interval or
                sampler.f.request_save_log() or
                sampler.f.terminate_condition() or 
                self.terminate_condition()
            ):
                logger.write_csv([str(ite)] + ['%e' % true_eval_mean] + f.info_list() + self.log() + sampler.log())

        return [f.eval_count, f.best_eval, f.is_success()]

    def update(self, X, evals):
        self.gen_count += 1
        
        # save parameters before update
        old_mean = np.copy(self.model.m)
        old_Sigma = np.copy((self.model.sigma ** 2) * self.model.C)
            
        if evals.shape[0] == 1: 
            # if num of reevaluations is one
            even_evals = np.vstack([evals, evals[None, -1]])
        else:
            even_evals = evals
            
        even_evals_len = (len(even_evals) // 2) * 2
        half_evals1 = np.mean(even_evals[:even_evals_len][::2, :], axis=0)
        half_evals2 = np.mean(even_evals[:even_evals_len][1::2, :], axis=0)
        
        # compute update directions
        updated_m_half1, updated_C_half1, updated_sigma_half1 = \
            self.compute_updated_parameter(X, half_evals1)
        updated_m_half2, updated_C_half2, updated_sigma_half2 = \
            self.compute_updated_parameter(X, half_evals2)
            
        updated_m, updated_C, updated_sigma \
            = self.compute_updated_parameter(X, np.mean(evals, axis=0), is_update_evolution_path=True)
        
        # START: Adaptation of num of reevaluation
        Deltamean_half1 = updated_m_half1 - old_mean
        DeltaSigma_half1 = (updated_sigma_half1 ** 2) * updated_C_half1 - old_Sigma
        Deltamean_half2 = updated_m_half2 - old_mean
        DeltaSigma_half2 = (updated_sigma_half2 ** 2) * updated_C_half2 - old_Sigma
        
        self.update_cov_accumulation(Deltamean_half1, DeltaSigma_half1, Deltamean_half2, DeltaSigma_half2)
        
        corr_mean, corr_Sigma = self.compute_corr()
        target_corr = self.compute_target_corr()
        min_rel_corr = np.min([corr_mean / target_corr[0], corr_Sigma / target_corr[1]])
        relative_corr = np.clip((min_rel_corr - 1), -1, 1) * self.gamma 
        
        self.eval_num = self.eval_num * np.exp(- relative_corr)
        self.eval_num = max(self.eval_num, self.eval_num_min)

        # END: Adaptation of num of reevaluation
        
        # START: Learning rate adaptation
        Deltamean = updated_m - old_mean
        DeltaSigma = (updated_sigma ** 2) * updated_C - old_Sigma
        
        self.Emean, self.ESigma, self.Vmean, self.VSigma = self.update_accumulation(
            self.Emean, self.ESigma, self.Vmean, self.VSigma, Deltamean, DeltaSigma
        )  

        before_eta_mean = self.eta_mean
        hatSNRmean, hatSNRSigma = self.compute_snr(self.Emean, self.ESigma, self.Vmean, self.VSigma)
        
        dumping_for_lra = np.log(self.eval_num) - np.log(self.eval_num_min) + 1
        
        relativeSNRmean = np.clip((hatSNRmean / self.alpha / self.eta_mean ) - 1, -1, 1) / dumping_for_lra
        self.eta_mean = self.eta_mean * np.exp(
            min(self.gamma * self.eta_mean, self.beta_mean) * relativeSNRmean
        )
        
        relativeSNRSigma = np.clip((hatSNRSigma / self.alpha / self.eta_Sigma ) - 1, -1, 1) / dumping_for_lra
        self.eta_Sigma = self.eta_Sigma * np.exp(
            min(self.gamma * self.eta_Sigma, self.beta_Sigma) * relativeSNRSigma
        )
                
        # cap
        self.eta_mean = min(self.eta_mean, 1.0) 
        self.eta_Sigma = min(self.eta_Sigma, 1.0)
        
        self.model.m = old_mean + self.eta_mean * Deltamean
        Sigma = old_Sigma + self.eta_Sigma * DeltaSigma
        
        logeigsum = np.log(np.linalg.det(Sigma))
        self.model.sigma = np.exp(logeigsum / 2.0 / self.d)
        
        if self.model.sigma == 0:
            print("sigma is 0, so the optimization ends. ")
            return
        self.model.C = Sigma / (self.model.sigma ** 2)
        
        # step-size correction
        self.model.sigma *= before_eta_mean / self.eta_mean

        #END: Learning rate adaptation
        
    def log_header(self):
        return self.model.log_header() + ['eta_mean'] + ['eta_Sigma'] + ['eval_num'] + ['corr_mean'] + ['corr_Sigma'] + ['corr_mean_target'] + ['corr_Sigma_target']

    def log(self):
        corr_mean, corr_Sigma = self.compute_corr()
        target_corr = self.compute_target_corr()
        return self.model.log() + ['%e' % self.eta_mean] + ['%e' % self.eta_Sigma] + ['%e' % self.eval_num] \
                + ['%e' % corr_mean] + ['%e' % corr_Sigma] + ['%e' % target_corr[0]] + ['%e' % target_corr[1]]
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

# public symbols
__all__ = ['DefaultSampler', 'ImportanceMixingSampler']


class DefaultSampler(object):
    def __init__(self, f, lam):
        self.f = f
        self.lam = lam

    def __call__(self, model):
        X = model.sampling(self.lam)
        evals = self.f(X)
        return X, evals

    def verbose_display(self):
        return ''

    def log_header(self):
        return []

    def log(self):
        return []


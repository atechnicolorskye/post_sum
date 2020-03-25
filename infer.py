import numpy as np
import torch

import pdb

class InferRanndomWalk(object):
    # random walk, obtain posterior for beta from 1 to T
    def __init__(self):
        pass

    def infer(self, X):
        '''
        Taken from pg 109 of Mike West's Bayesian Forecasting and Dynamic Models. Cross-referenced with MLAPP.
        '''
        # get self.dim, T and x_0
        self.T, self.dim  = X.shape[0] - 1, X.shape[1]
        self.x0 = X[0, :].reshape(-1, 1)
        self.diag_x0 = np.diag(X[0, :])

        # assign identity once and for all
        self.identity = np.identity(self.dim)

        # init posteriors
        self.betas_mu = np.zeros((self.T, self.dim))
        self.betas_cov = np.zeros((self.T, self.dim, self.dim))

        # set initial cov to identity
        self.betas_cov[0] = self.identity

        # infer
        for i in range(1, self.T):
            R_t = self.betas_cov[i-1]
            Q_t = self.diag_x0 * R_t * self.diag_x0 + self.identity
            f_t = np.matmul(self.diag_x0, self.betas_mu[i-1])
            e_t = X[i] - f_t
            A_t = R_t * self.diag_x0 / Q_t
            self.betas_mu[i] = self.betas_mu[i-1] + np.matmul(A_t, e_t)
            self.betas_cov[i] = R_t - A_t * A_t.T * Q_t

        return  self.betas_mu, self.betas_cov

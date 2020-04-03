import numpy as np
import matplotlib.pyplot as plt
import pystan

import gzip
import os
import pdb
import pickle

seed = 1

np.random.seed(seed)
threads = 4
os.environ['STAN_NUM_THREADS'] = str(threads)

n_T = 100
n_F = 5
n_Y = 10
n_D = 5000
n_W = 1000
n_C = 4
AD = 0.95

Y = np.zeros((n_T, n_Y))
B = np.zeros((n_T, n_Y, n_F))
log_F_sigma = np.zeros((n_T, n_F))

# y hyperparameters
y_sigma = 1./ np.random.gamma(1, 3) * np.ones(n_Y)

# f hyperparameters
log_f_sigma_ = np.random.normal(-1, 0.25, n_F)
# diag chol
chol_f_sigma = np.diag(np.abs(np.random.normal(0, 0.1, n_F)))
# full chol
# chol_f_sigma = np.random.normal(0, 0.25, n_F * n_F).reshape(n_F, n_F)
# row, col = np.diag_indices(n_F)
# chol_f_sigma[row, col] = np.abs(chol_f_sigma[row, col])

# B
discount = 0.95
base_order = 0.05
bias_order = 0.5
p_connect = 0.3
n_connect = np.int(0.3 * n_Y * n_F)
add = (2 * np.random.binomial(1, 0.5, n_connect) - 1) * (bias_order + np.abs(np.random.standard_normal(n_connect)))
B_ = base_order * np.random.standard_normal(n_Y * n_F)
B_[:n_connect] += add
B_ = np.random.permutation(B_).reshape(n_Y, n_F)
row, col = np.triu_indices(n_Y, 0, n_F)
B_[row, col] = 0
np.fill_diagonal(B_, 1)

# Initialise
# log_F_sigma[0] = np.random.multivariate_normal(log_f_sigma_, chol_f_sigma ** 2)
log_F_sigma[0] = log_f_sigma_ + chol_f_sigma @ np.random.standard_normal(n_F) 
B[0] = B_ + discount * base_order * np.tril(np.random.standard_normal(n_Y * n_F).reshape(n_Y, n_F), k=-1)
Y[0] = B[0] @ np.random.multivariate_normal(np.zeros(n_F), np.diag(np.exp(2 * log_F_sigma[0]))) + y_sigma * np.random.standard_normal(n_Y)

for i in range(2, n_T):
    # log_F_sigma[i] = np.random.multivariate_normal(log_F_sigma[i], chol_f_sigma ** 2)    
    log_F_sigma[i] = log_F_sigma[i-1] + chol_f_sigma @ np.random.standard_normal(n_F)
    B[i] = B[i-1] + discount * base_order * np.tril(np.random.standard_normal(n_Y * n_F).reshape(n_Y, n_F), k=-1)
    Y[i] = B[i] @ np.random.multivariate_normal(np.zeros(n_F), np.diag(np.exp(2 * log_F_sigma[i]))) + y_sigma * np.random.standard_normal(n_Y)

dat = {
    'P': n_Y,
    'D': n_F,
    'TS': n_T,
    'disc': discount,
    # 'fac_mu': np.zeros(n_F),
    'y': Y
    }

extra_compile_args = ['-pthread', '-DSTAN_THREADS']
model = pystan.StanModel(file='infer.stan', extra_compile_args=extra_compile_args)
fit = model.sampling(data=dat, iter=n_D, warmup=n_W, seed=seed, chains=n_C, verbose=True, control={'adapt_delta':AD}, n_jobs=threads)
with gzip.open('pystan_fit_{}_{}_{}_{}_{}.gz'.format(n_D, n_W, seed, n_C, AD), 'wb') as f:
    pickle.dump({'model' : model, 'fit' : fit}, f)
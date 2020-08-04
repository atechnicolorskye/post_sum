import numpy as np
import matplotlib.pyplot as plt
import pystan

np.set_printoptions(precision=5, suppress=True)

import gzip
import os
import pdb
import pickle

seed = 1

np.random.seed(seed)

n_N = 500
n_F = 5
n_Y = 10
n_D = 2000
n_W = 1000
n_C = 4
AD = 0.95

Y = np.zeros((n_N, n_Y))
B = np.zeros((n_Y, n_F))
log_F_sigma = np.zeros((n_F))

# y hyperparameters
log_y_sigma = np.random.normal(0, 1) * np.ones(n_Y)

# f hyperparameters
# log_f_sigma = np.random.normal(0, 0.25, n_F)
# diag chol
# chol_log_f_sigma = np.diag(np.abs(np.random.normal(0, 0.1, n_F)))
# full chol
chol_log_f_sigma = np.random.normal(0, 1, n_F * n_F).reshape(n_F, n_F)
row, col = np.diag_indices(n_F)
chol_log_f_sigma[row, col] = np.abs(chol_log_f_sigma[row, col])

# B
base_order = 1
# base_order = 0.05
# bias_order = 0.5
# p_connect = 0.3
# n_connect = np.int(0.3 * n_Y * n_F)
# add = (2 * np.random.binomial(1, 0.5, n_connect) - 1) * (bias_order + np.abs(np.random.standard_normal(n_connect)))
B_ = base_order * np.random.standard_normal(n_Y * n_F)
# B_[:n_connect] += add
B_ = np.random.permutation(B_).reshape(n_Y, n_F)
row, col = np.triu_indices(n_Y, 0, n_F)
B_[row, col] = 0
np.fill_diagonal(B_, 1)

# Initialise
# log_F_sigma[0] = np.random.multivariate_normal(log_f_sigma_, chol_log_f_sigma ** 2)
log_F_sigma = np.zeros(n_F) # chol_log_f_sigma @ np.random.standard_normal(n_F) 
B = B_ # + base_order * np.tril(np.random.standard_normal(n_Y * n_F).reshape(n_Y, n_F), k=-1)

for i in range(1, n_N):
    Y[i] = B @ np.random.multivariate_normal(np.zeros(n_F), np.diag(np.exp(2 * log_F_sigma))) + np.exp(log_y_sigma) * np.random.standard_normal(n_Y)

dat = {
    'P': n_Y,
    'F': n_F,
    'N': n_N,
    # 'fac_mu': np.zeros(n_F),
    'y': Y
    }

model = pystan.StanModel(file='infer.stan')
fit = model.sampling(data=dat, iter=n_D, warmup=n_W, seed=seed, chains=n_C, control={'adapt_delta':AD})
# with gzip.open('pystan_non_tv_fit_{}_{}_{}_{}_{}.gz'.format(n_D, n_W, seed, n_C, AD), 'wb') as f:
#     pickle.dump({'model' : model, 'fit' : fit}, f)
res = fit.extract(pars=['beta', 'fac'])
plt.scatter(B, np.mean(res['beta'], axis=0))
plt.show()
pdb.set_trace()
print(B - np.mean(res['beta'], axis=0))
print('log_y_sigma', log_y_sigma)
print('log_F_sigma', log_F_sigma)
print(fit.stansummary(pars=['log_y_sd', 'L_lower'])) 
Y_hat = np.einsum('ijk,ilk->ilj', res['beta'], res['fac'])
print(np.mean(Y - Y_hat, axis=(0, 1)))


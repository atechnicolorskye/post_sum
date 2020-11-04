import numpy as np
import pandas as pd
from numpy.random import multivariate_normal as mvnorm
from numpy.linalg import norm

import time

from regain.covariance import CVXInequalityTimeGraphicalLasso, TimeGraphicalLasso

seed = 0
np.random.seed(seed)

# Standardised Fama French 5 to industry portfolio 30
# Import data
data = pd.read_pickle("../ff5_30_nonsmooth_standard_4000_draws.pkl")

# # Restrict to 100 time points
X = data[10:110].transpose(2, 1, 0)
X_cov = np.einsum('ijkl,jmkl->imkl', np.expand_dims(X, 1), np.expand_dims(X, 0))

tic = time.perf_counter()
tgl = CVXInequalityTimeGraphicalLasso(max_iter=1e4, loss='LL', c_level=0.2, theta=0.5, psi="laplacian", tol=1e-4)
emp_inv_score, baseline_score, fit_score, pre_cvx = tgl.fit_cov(X_cov).eval_cov_pre() 
toc = time.perf_counter()
print('Running Time :{}'.format(toc - tic))

np.save("mosek_sol_ff5_30_nonsmooth_standard_alpha_0.2_laplacian.npy", pre_cvx)

tic = time.perf_counter()
tgl = CVXInequalityTimeGraphicalLasso(max_iter=1e4, loss='LL', c_level=0.2, theta=0.5, psi="l2", tol=1e-4)
emp_inv_score, baseline_score, fit_score, pre_cvx = tgl.fit_cov(X_cov).eval_cov_pre() 
toc = time.perf_counter()
print('Running Time :{}'.format(toc - tic))

np.save("mosek_sol_ff5_30_nonsmooth_standard_alpha_0.2_l2.npy", pre_cvx)

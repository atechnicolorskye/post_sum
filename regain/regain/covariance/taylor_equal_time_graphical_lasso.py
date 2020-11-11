# BSD 3-Clause License

# Copyright (c) 2017, Federico T.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""Sparse inverse covariance selection over time via ADMM.

More information can be found in the paper linked at:
https://arxiv.org/abs/1703.01958
"""
from __future__ import division

from itertools import compress 

import warnings
import copy

import numpy as np
from scipy import linalg
from six.moves import map, range, zip
from sklearn.covariance import empirical_covariance, log_likelihood
from sklearn.utils.extmath import squared_norm
from sklearn.utils.validation import check_X_y

from regain.covariance.graphical_lasso_ import (
    GraphicalLasso, neg_logl, dtrace)
from regain.norm import l1_od_norm
from regain.prox import soft_thresholding_od
from regain.update_rules import update_rho
from regain.utils import convergence, error_norm_time
from regain.validation import check_norm_prox

from scipy.optimize import minimize_scalar
from functools import partial

import pdb

def loss_gen(loss, S, K):
    T, p, _, = S.shape
    losses = np.array([loss(S[i], K[i]) for i in range(T)])
    return losses


def penalty_objective(Z_0, Z_1, Z_2, psi, theta):
    """Penalty-only objective function for time-varying graphical LASSO."""
    return theta * sum(map(l1_od_norm, Z_0)) + (1 - theta) * sum(map(psi, Z_2 - Z_1))


def rbf_weights(T, bandwidth, mult):
    """RBF Weights"""
    weights = np.zeros((T, T))
    for i in range(T):
        time_diff = np.arange(-i, T-i)
        weights[i] = np.exp(-time_diff ** 2 / bandwidth) * (mult - 1) + 1
    return weights


def lin_weights(T, bandwidth, mult):
    """Linear Weights"""
    weights = np.zeros((T, T))
    for i in range(T):
        time_diff = np.arange(-i, T-i)
        weights[i] = np.exp(-np.abs(time_diff) / bandwidth) * (mult - 1) + 1
    return weights


def taylor_equal_time_graphical_lasso(
    S, K_init, max_iter, loss, C, theta, rho, mult, 
    weights, m, eps, psi, gamma, tol, rtol, verbose, 
    return_history, return_n_iter, mode, compute_objective, 
    stop_at, stop_when, update_rho_options
    ):
    """Equality constrained time-varying graphical LASSO solver.

    Solves the following problem via ADMM:
        min sum_{i=1}^T ||K_i||_{od,1} + beta sum_{i=2}^T Psi(K_i - K_{i-1})
        s.t. objective = c_i for i = 1, ..., T

    where S_i = (1/n_i) X_i^T X_i is the empirical covariance of data
    matrix X (training observations by features).

    Parameters
    ----------
    emp_cov : ndarray, shape (n_features, n_features)
        Empirical covariance of data.
    alpha, beta : float, optional
        Regularisation parameter.
    rho : float, optional
        Augmented Lagrangian parameter.
    max_iter : int, optional
        Maximum number of iterations.
    n_samples : ndarray
        Number of samples available for each time point.
    gamma: float, optional
        Kernel parameter when psi is chosen to be 'kernel'.
    tol : float, optional
        Absolute tolerance for convergence.
    rtol : float, optional
        Relative tolerance for convergence.
    return_history : bool, optional
        Return the history of computed values.
    return_n_iter : bool, optional
        Return the number of iteration before convergence.
    verbose : bool, default False
        Print info at each iteration.
    update_rho_options : dict, optional
        Arguments for the rho update.
        See regain.update_rules.update_rho function for more information.
    compute_objective : bool, default True
        Choose to compute the objective value.
    init : {'empirical', 'zero', ndarray}
        Choose how to initialize the precision matrix, with the inverse
        empirical covariance, zero matrix or precomputed.

    Returns
    -------
    K : numpy.array, 3-dimensional (T x d x d)
        Solution to the problem for each time t=1...T .
    history : list
        If return_history, then also a structure that contains the
        objective value, the primal and dual residual norms, and tolerances
        for the primal and dual residual norms at each iteration.

    """
    psi, prox_psi, psi_node_penalty = check_norm_prox(psi)

    if loss == 'LL':
        loss_function = neg_logl
    else:
        loss_function = dtrace

    T = S.shape[0]
    S_flat = S.copy().reshape(T, S.shape[1] * S.shape[2])
    I_flat = np.diagflat(S.shape[1]).ravel()

    Z_0 = K_init
    Z_0_flat = K_init.copy().reshape(T, S.shape[1] * S.shape[2])
    Z_1 = Z_0.copy()[:-1] 
    Z_2 = Z_0.copy()[1:]  

    u = np.zeros(T)
    # U_0 = np.zeros_like(Z_0) 
    U_1 = np.zeros_like(Z_1)
    U_2 = np.zeros_like(Z_2)

    Z_0_old = Z_0.copy()
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for one less matrix for t = 0 and t = T
    divisor = np.full(T, 2, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    if weights is not None:
        rho = rho * np.ones(T)    
        weights = rbf_weights(T, weights, mult)
        con_obj = {}
        for t in range(T):
            con_obj[t] = []

    con_obj_mean = []
    con_obj_max = []
    out_obj = []

    loss_res = np.zeros(T)

    checks = [
        convergence(
            obj=penalty_objective(Z_0, Z_1, Z_2, psi, theta))
    ]

    loss_init = loss_gen(loss_function, S, Z_0_old)
    loss_diff = C - loss_init

    C_  = C - loss_diff

    for iteration_ in range(max_iter):
        if iteration_ == 0:
            loss_res_old = -loss_diff
            g = np.zeros(T)
            nabla = np.zeros_like(Z_0_flat)
            trace_nabla_Z_0_old =  g.copy()
            trace_nabla_nabla = g.copy()
        else:
            if loss_function.__name__ == 'neg_logl':
                nabla = np.array([S_t - np.linalg.inv(Z_0_old_t).ravel() for (S_t, Z_0_old_t) in zip(S_flat, Z_0_old)])
            elif loss_function.__name__ == 'dtrace': 
                nabla = np.array([(2 * Z_0_old_t.ravel() @ S_t - I) for (S_t, Z_0_old_t) in zip(S_flat, Z_0_old)])
            trace_nabla_Z_0_old = np.array([nabla_t @ Z_0_old_t.ravel() for (nabla_t, Z_0_old_t) in zip(nabla, Z_0_old)])
            g = trace_nabla_Z_0_old - loss_res_old - u
            trace_nabla_nabla =  np.einsum('ij,ij->i', nabla, nabla)
        
        A_p = Z_0_old
        if weights is not None:
            A_p[:-1] += rho[:-1, None, None] * (Z_1 - U_1)
            A_p[1:] += rho[1:, None, None] * (Z_2 - U_2)
        else:
            A_p[:-1] += rho * (Z_1 - U_1)
            A_p[1:] += rho * (Z_2 - U_2)

        def _Z_0(x, t, weights):
            if weights is not None:
                Z_0_t_ = (A_p_t + x * g[t] * nabla[t] - ((x * nabla_t_T_A_p_t + x ** 2 * g[t] * trace_nabla_nabla[t]) * nabla[t])  / (divisor[t] * rho[t] + 1 + x * trace_nabla_nabla[t])).reshape(S.shape[1], S.shape[2])
                return soft_thresholding_od(0.5 * (Z_0_t_ + Z_0_t_.transpose(1,0)) / (rho[t] * divisor[t] + 1), lamda=theta / (rho[t] * divisor[t] + 1))
            else:
                Z_0_t_ = (A_p_t + x * g[t] * nabla[t] - ((x * nabla_t_T_A_p_t + x ** 2 * g[t] * trace_nabla_nabla[t]) * nabla[t])  / (divisor[t] * rho + 1 + x * trace_nabla_nabla[t])).reshape(S.shape[1], S.shape[2])
                return soft_thresholding_od(0.5 * (Z_0_t_ + Z_0_t_.transpose(1,0)) / (rho * divisor[t] + 1), lamda=theta / (rho * divisor[t] + 1))
            

        def _f(x, loss_function, t, S_t, _Z_0, weights, c_t, loss_res_old_t, nabla_t, trace_nabla_Z_0_old_t):
            _Z_0_t = _Z_0(x, t, weights)
            loss_res_new = loss_function(S_t, _Z_0_t) - c_t
            return loss_res_new ** 2 + (loss_res_new - loss_res_old_t - nabla_t @ _Z_0_t.ravel() + trace_nabla_Z_0_old_t) ** 2

        
        for t in range(T):
            if weights is not None:
                Z_0[t] = soft_thresholding_od(0.5 * (A_p[t] + A_p[t].transpose(1,0)) / (rho[t] * divisor[t] + 1), lamda=theta / (rho[t] * divisor[t] + 1))
                con_obj[t].append(loss_res[t])    
            else:
                Z_0[t] = soft_thresholding_od(0.5 * (A_p[t] + A_p[t].transpose(1,0)) / (rho * divisor[t] + 1), lamda=theta / (rho * divisor[t] + 1))
            loss_res[t] = loss_function(S[t], Z_0[t]) - C[t]
            A_p_t = A_p[t].ravel()
            nabla_t_T_A_p_t = nabla[t] @ A_p_t
            out = minimize_scalar(partial(_f, t=t, loss_function=loss_function, S_t=S[t], _Z_0=_Z_0, weights=weights, c_t=C[t], loss_res_old_t=loss_res_old[t], nabla_t=nabla[t], trace_nabla_Z_0_old_t=trace_nabla_Z_0_old[t]))
            Z_0[t] = _Z_0(out.x, t, weights)
            loss_res[t] = loss_function(S[t], Z_0[t]) - C[t]
            u[t] += loss_res[t]    
            if weights is not None:
                con_obj[t][-1] = loss_res[t]
                if len(con_obj[t]) > m and np.mean(con_obj[t][-m:-int(m/2)]) < np.mean(con_obj[t][-int(m/2):]) and loss_res[t] > eps:
                    if con_obj[t][-2] < con_obj[t][-1] and loss_res[t] > eps:
                        rho *= weights[t]
                        u /= weights[t]
                        U_1 /= weights[t][:-1, None, None]
                        U_2 /= weights[t][1:, None, None]
                        con_obj[t] = []
                        print('Mult', iteration_, t, rho[t])      
            
        # other Zs
        A_1 = Z_0[:-1] + U_1
        A_2 = Z_0[1:] + U_2
        if not psi_node_penalty:
            if weights is not None:
                prox_e = prox_psi(A_2 - A_1, lamda=2. * (1 - theta) / rho[t])
                Z_1 = .5 * (A_1 + A_2 - prox_e)
                Z_2 = .5 * (A_1 + A_2 + prox_e)
            else:
                prox_e = prox_psi(A_2 - A_1, lamda=2. * (1 - theta) / rho)
                Z_1 = .5 * (A_1 + A_2 - prox_e)
                Z_2 = .5 * (A_1 + A_2 + prox_e)
        else:
            if weights is not None:
                Z_1, Z_2 = prox_psi(
                    np.concatenate((A_1, A_2), axis=1), lamda=.5 * (1 - theta) / rho[t],
                    rho=rho[t], tol=tol, rtol=rtol, max_iter=max_iter)
            else:
                Z_1, Z_2 = prox_psi(
                    np.concatenate((A_1, A_2), axis=1), lamda=.5 * (1 - theta) / rho,
                    rho=rho, tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        con_obj_mean.append(np.mean(loss_res) ** 2)
        con_obj_max.append(np.max(loss_res))

        U_1 += Z_0[:-1] - Z_1
        U_2 += Z_0[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
            squared_norm(loss_res) + # squared_norm(Z_0_res) + 
            squared_norm(Z_0[:-1] - Z_1) + squared_norm(Z_0[1:] - Z_2)
            )

        dual_con_res = loss_res - loss_res_old
        dual_con_res += (trace_nabla_Z_0_old - np.array([np.sum(nabla_t * Z_0_t.flatten()) for (nabla_t, Z_0_t) in zip(nabla, Z_0)]))

        loss_res_old = loss_res.copy()

        if weights is not None:
            snorm = np.sqrt(
                    squared_norm((rho * dual_con_res)[:, None, None] * nabla) + 
                    squared_norm(rho[:-1, None, None] * (Z_1 - Z_1_old)) + squared_norm(rho[1:, None, None] * (Z_2 - Z_2_old))
                )
            e_dual = np.sqrt(loss_res.size + 2 * Z_1.size) * tol + rtol * np.sqrt(
                    squared_norm(rho * u) +
                    squared_norm(rho[:-1, None, None] * U_1) + 
                    squared_norm(rho[1:, None, None] * U_2)
                )     
        else:
            snorm = rho * np.sqrt(
                    squared_norm(dual_con_res[:, None, None] * nabla) + 
                    squared_norm(Z_1 - Z_1_old) + squared_norm(Z_2 - Z_2_old)
                )
            e_dual = np.sqrt(loss_res.size + 2 * Z_1.size) * tol + rtol * rho * np.sqrt(
                    squared_norm(u) +
                    squared_norm(U_1) + 
                    squared_norm(U_2)
                )

        obj = penalty_objective(Z_0, Z_1, Z_2, psi, theta)

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            # e_pri=np.sqrt(loss_res.size + Z_0.size + 2 * Z_1.size) * tol + rtol * 
            e_pri=np.sqrt(loss_res.size + 2 * Z_1.size) * tol + rtol * 
                (
                max(np.sqrt(squared_norm(loss_res + C)), np.sqrt(squared_norm(C))) + 
                # max(np.sqrt(squared_norm(Z_0)), np.sqrt(squared_norm(Z_0_old))) +
                max(np.sqrt(squared_norm(Z_1)), np.sqrt(squared_norm(Z_0[:-1]))) + 
                max(np.sqrt(squared_norm(Z_2)), np.sqrt(squared_norm(Z_0[1:])))
                ),
            e_dual=e_dual
        )

        Z_0_old = Z_0.copy()
        Z_1_old = Z_1.copy()
        Z_2_old = Z_2.copy()

        if verbose:
            print(
                "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
                "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

        out_obj.append(penalty_objective(Z_0, Z_0[:-1], Z_0[1:], psi, theta))
        if not iteration_ % 100:
            print(iteration_)
            print(np.max(loss_res), np.mean(loss_res))
            print(out_obj[-1])
        checks.append(check)

        if stop_at is not None:
            if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
                break

        if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
            break

        if weights is None:
            if len(con_obj_mean) > m:
                if np.mean(con_obj_mean[-m:-int(m/2)]) < np.mean(con_obj_mean[-int(m/2):]) and np.max(loss_res) > eps:
                # or np.mean(con_obj_max[-100:-50]) < np.mean(con_obj_max[-50:])) # np.mean(loss_res) > 0.25:
                    print("Rho Mult", mult * rho, iteration_, np.mean(loss_res), con_obj_max[-1])
                    # loss_diff /= 5            
                    # C_ = C - loss_diff           
                    # resscale scaled dual variables
                    rho = mult * rho
                    u /= mult
                    U_1 /= mult
                    U_2 /= mult
                    con_obj_mean = []
                    con_obj_max = []
    else:
        warnings.warn("Objective did not converge.")

    print(iteration_, out_obj[-1])
    # print(out_obj)
    print(check.rnorm, check.e_pri)
    print(check.snorm, check.e_dual)

    covariance_ = np.array([linalg.pinvh(x) for x in Z_0])
    return_list = [Z_0, covariance_]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    return return_list


class TaylorEqualTimeGraphicalLasso(GraphicalLasso):
    """Sparse inverse covariance estimation with an l1-penalized estimator.

    Parameters
    ----------
    alpha : positive float, default 0.01
        Regularization parameter for precision matrix. The higher alpha,
        the more regularization, the sparser the inverse covariance.

    beta : positive float, default 1
        Regularization parameter to constrain precision matrices in time.
        The higher beta, the more regularization,
        and consecutive precision matrices in time are more similar.

    psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
        Type of norm to enforce for consecutive precision matrices in time.

    rho : positive float, default 1
        Augmented Lagrangian parameter.

    over_relax : positive float, deafult 1
        Over-relaxation parameter (typically between 1.0 and 1.8).

    tol : positive float, default 1e-4
        Absolute tolerance to declare convergence.

    rtol : positive float, default 1e-4
        Relative tolerance to declare convergence.

    max_iter : integer, default 100
        The maximum number of iterations.

    verbose : boolean, default False
        If verbose is True, the objective function, rnorm and snorm are
        printed at each iteration.

    assume_centered : boolean, default False
        If True, data are not centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data are centered before computation.

    time_on_axis : {'first', 'last'}, default 'first'
        If data have time as the last dimension, set this to 'last'.
        Useful to use scikit-learn functions as train_test_split.

    update_rho_options : dict, default None
        Options for the update of rho. See `update_rho` function for details.

    compute_objective : boolean, default True
        Choose if compute the objective function during iterations
        (only useful if `verbose=True`).

    init : {'empirical', 'zeros', ndarray}, default 'empirical'
        How to initialise the inverse covariance matrix. Default is take
        the empirical covariance and inverting it.

    Attributes
    ----------
    covariance_ : array-like, shape (n_times, n_features, n_features)
        Estimated covariance matrix

    precision_ : array-like, shape (n_times, n_features, n_features)
        Estimated precision matrix.

    n_iter_ : int
        Number of iterations run.

    """
    def __init__(
            self, max_iter=1000, loss='LL', c_level=None, theta=0.5,
            rho=1e2, mult=2, weights=None, m=100, eps=2, psi='laplacian', 
            gamma=None, tol=1e-4, rtol=1e-4, mode='admm', 
            verbose=False, assume_centered=False, return_history=False, 
            update_rho_options=None, compute_objective=True, stop_at=None, 
            stop_when=1e-4, suppress_warn_list=False):
        super().__init__(
            alpha=1., rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
            verbose=verbose, assume_centered=assume_centered, mode=mode,
            update_rho_options=update_rho_options,
            compute_objective=compute_objective)
        self.max_iter = max_iter
        self.loss = loss
        self.c_level = c_level
        self.theta = theta
        self.rho = rho
        self.mult = mult
        self.weights = weights
        self.m = m
        self.eps = eps
        self.psi = psi
        self.gamma = gamma
        self.return_history = return_history
        self.stop_at = stop_at
        self.stop_when = stop_when
        self.suppress_warn_list = suppress_warn_list


    def get_observed_precision(self):
        """Get the observed precision matrix.

        Returns
        -------
        precision_ : array-like,
            The precision matrix associated to the current covariance object.

        """
        return self.get_precision()


    def _fit(self, emp_cov):
        """Fit the InequalityTimeGraphicalLasso model to X.

        Parameters
        ----------
        emp_cov : ndarray, shape (n_time, n_features, n_features)
            Empirical covariance of data.

        """
        out = taylor_equal_time_graphical_lasso(
              emp_cov, self.emp_inv, max_iter=self.max_iter, loss=self.loss, C=self.C, 
              theta=self.theta, rho=self.rho, mult=self.mult, weights=self.weights, m=self.m, 
              eps=self.eps, psi=self.psi, gamma=self.gamma, tol=self.tol, rtol=self.rtol, 
              verbose=self.verbose, return_history=self.return_history, return_n_iter=True,  
              mode=self.mode, compute_objective=self.compute_objective, stop_at=self.stop_at,
              stop_when=self.stop_when, update_rho_options=self.update_rho_options)
        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.n_iter_ = out
        return self


    def fit_obs(self, X, y):
        """Fit the InequalityTimeGraphicalLasso model to observations.

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each sample.

        """
        # Covariance does not make sense for a single feature
        
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        emp_cov = []
        emp_inv_score = []
        sam_inv_score = []

    
        for i in self.classes_:
            emp_cov_i = empirical_covariance(X[y == i], assume_centered=self.assume_centered)
            emp_inv_i = np.linalg.inv(emp_cov_i)
            _, log_det = np.linalg.slogdet(emp_inv_i)
            emp_cov.append(emp_cov_i)
            emp_inv_score.append(log_det - np.trace(emp_cov_i @ emp_inv_i))
            sam_inv_score.append(log_det - np.array([np.trace((X[y == i][[j], :].T @ X[y == i][[j], :]) @ emp_inv_i) for j in range(int(n_samples))]))

        self.emp_inv_score = np.array(emp_inv_score)
        self.sam_inv_score = np.array(sam_inv_score)

        self.constrained_to = None

        return self._fit(np.array(emp_cov), n_samples)


    def fit_cov(self, X):
        """Fit the InequalityTimeGraphicalLasso model to covariances.

        Parameters
        ----------
        X : ndarray, shape = (n_dimensions, n_dimensions, n_samples, time_steps)
        
        """
        n_dimensions, _, n_samples, time_steps = X.shape
        
        self.emp_cov = []
        self.emp_inv = []
        self.emp_inv_score = []
        self.sam_inv_score = []
        self.C = []

        for i in range(time_steps):
            self.emp_cov.append(np.mean(X[:, :, :, i], 2))
            self.emp_inv.append(np.linalg.inv(self.emp_cov[i]))
            if self.loss == 'LL':
                self.emp_inv_score.append(neg_logl(self.emp_cov[i], self.emp_inv[i]))
                self.sam_inv_score.append(np.array([neg_logl(X[:, :, j, i], self.emp_inv[i]) for j in range(n_samples)]))
            else:
                self.emp_inv_score.append(dtrace(self.emp_cov[i], self.emp_inv[i]))
                self.sam_inv_score.append(np.array([dtrace(X[:, :, j, i], self.emp_inv[i]) for j in range(n_samples)]))
            self.C.append(np.quantile(self.sam_inv_score[i], 1 - self.c_level, 0))

        self.emp_cov = np.array(self.emp_cov)
        self.emp_inv = np.array(self.emp_inv)

        self.emp_inv_score = np.array(self.emp_inv_score)
        self.sam_inv_score = np.array(self.sam_inv_score)

        return self._fit(self.emp_cov)


    def score(self, X, y):
        """Computes the log-likelihood of a Gaussian data set with
        `self.covariance_` as an estimator of its covariance matrix.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test data of which we compute the likelihood, where n_samples is
            the number of samples and n_features is the number of features.
            X is assumed to be drawn from the same distribution than
            the data used in fit (including centering).

        y :  array-like, shape = (n_samples,)
            Class of samples.

        Returns
        -------
        res : float
            The likelihood of the data set with `self.covariance_` as an
            estimator of its covariance matrix.

        """
        # Covariance does not make sense for a single feature
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        # compute empirical covariance of the test set
        test_cov = np.array(
            [
                empirical_covariance(
                    X[y == cl] - self.location_[i], assume_centered=True)
                for i, cl in enumerate(self.classes_)
            ])

        res = sum(
            X[y == cl].shape[0] * log_likelihood(S, K) for S, K, cl in zip(
                test_cov, self.get_observed_precision(), self.classes_))

        return res


    def eval_obs_pre(self, X, y):
        """Evaluate the log likelihood of estimated precisions compared to the inverse sample covariance at each time step

        Parameters
        ----------
        X : ndarray, shape = (n_samples * n_times, n_dimensions)
            Data matrix.
        y : ndarray, shape = (n_times,)
            Indicate the temporal belonging of each sample.
        """
        
        X, y = check_X_y(
            X, y, accept_sparse=False, dtype=np.float64, order="C",
            ensure_min_features=2, estimator=self)

        n_dimensions = X.shape[1]
        self.classes_, n_samples = np.unique(y, return_counts=True)
        n_times = self.classes_.size

        # n_samples = np.array([x.shape[0] for x in X])
        if self.assume_centered:
            self.location_ = np.zeros((n_times, n_dimensions))
        else:
            self.location_ = np.array(
                [X[y == cl].mean(0) for cl in self.classes_])

        precisions = self.get_observed_precision()

        emp_pre_score = []
        sam_pre_score = []
        slack = []

        for i in range(precisions.shape[0]):
            emp_cov = empirical_covariance(X[y == i] - self.location_[i], assume_centered=True)
            precision = precisions[i, :, :]
            # slack.append(-self.constrained_to[i] + logl(emp_cov, precision))
            _, log_det = np.linalg.slogdet(precision)
            emp_pre_score.append(log_det - np.trace(emp_cov @ precision))
            sam_pre_score.append(log_det - np.array([np.trace((X[y == i][[j], :].T @ X[y == i][[j], :]) @ precision) for j in range(int(n_samples))]))

        return self.emp_inv_score - np.array(emp_pre_score), self.sam_inv_score - np.array(sam_pre_score), precision


    def eval_cov_pre(self):
        """Evaluate the log likelihood of estimated precisions compared to the inverse sample covariance at each time step

        Parameters
        ----------
        X : ndarray, shape = (n_dimensions, n_dimensions, n_samples, time_steps)
        """
        
        precisions = self.precision_

        fit_score = []

        for i in range(precisions.shape[0]):
            precision = precisions[i]
            if self.loss == 'LL':
                fit_score.append(neg_logl(self.emp_cov[i], precision))
            else:
                fit_score.append(dtrace(self.emp_cov[i], precision))

        return self.emp_inv_score, self.C, fit_score, precisions 

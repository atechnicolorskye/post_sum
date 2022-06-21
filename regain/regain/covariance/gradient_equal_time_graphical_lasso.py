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
from sklearn.utils.extmath import squared_norm, fast_logdet
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
    for t in range(T):
        time_diff = np.arange(-t, T-t)
        weights[t] = np.exp(-time_diff ** 2 / bandwidth) * (mult - 1) + 1
    return weights


def exp_weights(T, bandwidth, mult):
    """Exponential Weights"""
    weights = np.zeros((T, T))
    for t in range(T):
        time_diff = np.arange(-t, T-t)
        weights[t] = np.exp(-np.abs(time_diff) / bandwidth) * (mult - 1) + 1
    return weights


def exp_weights_C(T, C, mult):
    """Exponential Weights"""
    weights = np.zeros((T, T))
    C = np.array(C)
    for t in range(T):
        c_diff = C - C[t]
        bandwidth = np.abs(np.min(c_diff))
        c_mask = c_diff < 0
        c_mask[t] = 1
        weights[t] = c_mask * np.exp(-np.abs(c_diff) / bandwidth) * (mult - 1) + 1
    return weights


def lin_weights(T, bandwidth, mult):
    """Linear Weights"""
    weights = np.zeros((T, T))
    for t in range(T):
        time_diff = np.arange(-t, T-t)
        weights[t] = (bandwidth - np.abs(time_diff)) / bandwidth * (mult - 1) + 1
    return weights


def gradient_equal_time_graphical_lasso(
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
        loss_func = neg_logl
    else:
        loss_func = dtrace

    T = S.shape[0]
    I = np.eye(S.shape[1])
    
    Z_0 = K_init
    Z_1 = Z_0.copy()[:-1] 
    Z_2 = Z_0.copy()[1:]  

    U_1 = np.zeros_like(Z_1)
    U_2 = np.zeros_like(Z_2)

    Z_0_old = Z_0.copy()
    Z_1_old = np.zeros_like(Z_1)
    Z_2_old = np.zeros_like(Z_2)

    # divisor for consensus variables, accounting for one less matrix for t = 0 and t = T
    divisor = np.full(T, 2, dtype=float)
    divisor[0] -= 1
    divisor[-1] -= 1

    rho = rho * np.ones(T)    
    if weights[0] is not None:
        if weights[0] == 'rbf':
            weights = rbf_weights(T, weights[1], mult)
        elif weights[0] == 'exp':
            weights = exp_weights(T, weights[1], mult)
        elif weights[0] == 'lin':
            weights = lin_weights(T, weights[1], mult)
        con_obj = {}
        for t in range(T):
            con_obj[t] = []
    
    # loss residuals
    loss_res = np.zeros(T)
    loss_init = loss_gen(loss_func, S, Z_0_old)
    loss_res_old = loss_init - C

    # loss_diff = C - loss_init
    # C_  = C - loss_diff

    con_obj_mean = []
    con_obj_max = []
    out_obj = []
    
    checks = [
        convergence(
            obj=penalty_objective(Z_0, Z_1, Z_2, psi, theta))
    ]

    # soft-thresholded gradient modified Z_0_t
    def _Z_0(x, A_t, nabla_t, rho_t, divisor_t):
        _A_t = A_t - x * nabla_t 
        return soft_thresholding_od(_A_t, lamda=theta / (rho_t * divisor_t))            


    # # soft-thresholded gradient modified Z_0
    # def _Z_0(x, A_t, nabla, rho, divisor):
    #     _A_t = A_t - x * nabla 
    #     return soft_thresholding_od(_A_t, lamda=theta / (rho * divisor)[:, None, None])            


    # constrained optimisation via line search
    def _f(x, _Z_0, A_t, nabla_t, rho_t, divisor_t, loss_func, S_t, c_t, loss_res_pre_t, Z_0_t):
        if x == 0:
            # if loss_res_pre_t == np.inf:
            #     return np.inf
            # else:
            return 0
            # return (1 - 1e-4) * (loss_res_pre_t ** 2) + 1e-4 * l1_od_norm(Z_0_t)
            # return np.inf

        _Z_0_t = _Z_0(x, A_t, nabla_t, rho_t, divisor_t)
        loss_res_t = loss_func(S_t, _Z_0_t) - c_t
        
        # if loss_res_t == np.inf:
        #     return np.inf
        # if loss_res_pre_t == np.inf:
        #     loss_res_pre_t = 0

        # if (loss_res_t ** 2 - loss_res_pre_t ** 2 - 2 * x *(loss_res_t * np.sum(_nabla_Z_0_t * (_Z_0_t - Z_0_t)) + theta * (l1_od_norm(_Z_0_t) - l1_od_norm(Z_0_t))) > 0):
            # return np.inf
        # else:
        return 0.99 * ((loss_res_t ** 2) - (loss_res_pre_t ** 2)) + 0.01 * (l1_od_norm(_Z_0_t) - l1_od_norm(Z_0_t)) # + np.sum(_Z_0_t - Z_0_t)

         # (np.sum(abs(_Z_0_t - np.abs(np.diag(_Z_0_t)))) - np.sum(abs(Z_0_t - np.abs(np.diag(Z_0_t)))))
        # return 0.8 * ((loss_res_t ** 2) - (loss_res_pre_t ** 2)) + 0.2 * (l1_od_norm(_Z_0_t) - l1_od_norm(Z_0_t)) # (np.sum(abs(_Z_0_t - np.abs(np.diag(_Z_0_t)))) - np.sum(abs(Z_0_t - np.abs(np.diag(Z_0_t)))))
        # return 1 / x * ((loss_res_t ** 2) - (loss_res_pre_t ** 2) + l1_od_norm(_Z_0_t) - l1_od_norm(Z_0_t)) # (np.sum(abs(_Z_0_t - np.abs(np.diag(_Z_0_t)))) - np.sum(abs(Z_0_t - np.abs(np.diag(Z_0_t)))))
        # return ((loss_res_t ** 2) - (loss_res_pre_t ** 2) + l1_od_norm(_Z_0_t) - l1_od_norm(Z_0_t))
        # return (1 - 1e-4) * (loss_res_t ** 2) + 1e-4 * l1_od_norm(Z_0_t)
        # return (loss_res_t ** 2 - (loss_res_pre_t ** 2)) + (loss_res_t - loss_res_pre_t - np.sum(nabla_t * (_Z_0_t - _Z_0(0, A_t, nabla_t, rho_t, divisor_t)))) ** 2

    
    # # max constrained optimisation via line search
    # def _f(x, _Z_0, A, nabla, rho, divisor, loss_func, S, C, loss_res_pre, Z_0):
    #     _Z_0 = _x`Z_0(x, A, nabla, rho, divisor)
    #     loss_res = loss_gen(loss_func, S, _Z_0) - C
    #     return np.max(loss_res ** 2) + np.max((loss_res - loss_res_pre - np.sum(nabla * (_Z_0 - Z_0), (1, 2))) ** 2)

    # def backtracking_line_search(Z_0_t_pre, loss_func, nabla_A_t, S_t, c_t, rho_t, divisor_t, nabla_Z_0_t_pre, loss_res_pre_t, a, b, max_iter=50):
    #     count = 0
    #     t = 1

    #     nabla_Z_0_t_pre_l2_norm = np.sum(nabla_Z_0_t_pre * nabla_A_t)

    #     while count < max_iter:
    #         Z_0_t_ = soft_thresholding_od(Z_0_t_pre - t * nabla_A_t, lamda=theta / (rho_t * divisor_t))
    #         Alpha_, Q = np.linalg.eigh(Z_0_t_)
    #         Alpha[Alpha < 0] = 0
    #         Z_0_t =    
    #         if loss_func(S_t, soft_thresholding_od(Z_0_t_pre - t * nabla_A_t, lamda=theta / (rho_t * divisor_t))) - c_t > loss_res_pre_t - a * t * nabla_Z_0_t_pre_l2_norm:
    #             count += 1
    #             t *= b
    #         else:
    #             break

    #     Z_0_t = soft_thresholding_od(Z_0_t_pre - t * nabla_A_t, lamda=theta / (rho_t * divisor_t))

    #     return Z_0_t, loss_func(S_t, Z_0_t) - c_t


    for iteration_ in range(max_iter):
        # update Z_0        
        A = np.zeros_like(Z_0)
        A[:-1] += Z_1 - U_1
        A[1:] += Z_2 - U_2
        A += A.transpose(0, 2, 1)
        A /= 2. 
        A /= divisor[:, None, None]

        Z_0_pre = soft_thresholding_od(A, lamda=theta / (rho * divisor)[:, None, None])
        loss_res_pre = loss_gen(loss_func, S, Z_0_pre) - C

        
        # out = minimize_scalar(
        #         partial(_f, _Z_0=_Z_0, A=A, nabla=nabla, rho=rho, 
        #                 divisor=divisor, loss_func=loss_func, S=S,   
        #                 C=C, loss_res_pre=loss_res_pre, Z_0=Z_0_pre)
        #         )
        # Z_0 = _Z_0(out.x, A, nabla, rho, divisor)
        # loss_res = loss_gen(loss_func, S, Z_0_pre) - C

        col = []

        for t in range(T):
            # if loss_res_pre[t] > 0:
            #     Z_0_pre[t] = soft_thresholding_od(A[t], lamda=theta / (rho[t] * divisor[t]))
            #     loss_res_pre[t] = loss_func(S[t], Z_0_pre[t]) - C[t]
            if loss_res_pre[t] == np.inf:
                if (loss_func(S[t], A[t]) - C[t]) == np.inf:
                    Alpha, Q = np.linalg.eigh(A[t])
                    Alpha[Alpha < 0] = 0
                    A[t] = (Q * Alpha) @ Q.T
                    Z_0_pre[t] = soft_thresholding_od(A[t], lamda=theta / (rho[t] * divisor[t]))
                    # soft_thresholding_od(A, lamda=theta / (rho * divisor)[:, None, None])
                    loss_res_pre[t] = loss_func(S[t], Z_0_pre[t]) - C[t]
            if loss_func.__name__ == 'neg_logl':            
                nabla_t = S[t] - np.linalg.inv(A[t])
            elif loss_func.__name__ == 'dtrace': 
                nabla_t = 2 * A[t] @ S[t] - I

                
            out = minimize_scalar(
                    partial(_f, _Z_0=_Z_0, A_t=A[t], nabla_t=nabla_t, rho_t=rho[t], 
                            divisor_t=divisor[t], loss_func=loss_func, S_t=S[t],   
                            c_t=C[t], loss_res_pre_t=loss_res_pre[t], Z_0_t=Z_0_pre[t]),
                    )
            Z_0[t] = _Z_0(out.x, A[t], nabla_t, rho[t], divisor[t])
            loss_res[t] = loss_func(S[t], Z_0[t]) - C[t]
            # if loss_func.__name__ == 'neg_logl':
            #     nabla_Z_t = S[t] - np.linalg.inv(Z_0_pre[t])
            #     nabla_A_t = S[t] - np.linalg.inv(A[t])
            # elif loss_func.__name__ == 'dtrace': 
            #     nabla_t = 2 * Z_0_pre[t] @ S[t] - I
            # Z_0[t], loss_res[t] = backtracking_line_search(Z_0_t_pre=Z_0_pre[t], loss_func=loss_func, nabla_A_t=nabla_A_t,  S_t=S[t], c_t=C[t], 
            #                                                 rho_t=rho[t], divisor_t=divisor[t], nabla_Z_0_t_pre=nabla_Z_t, loss_res_pre_t=loss_res_pre[t], a=0.49, b=0.5)
            if loss_res[t] == np.inf:
                pdb.set_trace()
            if weights[0] is not None:
                con_obj[t].append(loss_res[t] ** 2)    
                if len(con_obj[t]) > m and np.mean(con_obj[t][-m:-int(m/2)]) < np.mean(con_obj[t][-int(m/2):]) and loss_res[t] > eps:
                    col.append(t)
            # else:
            #     Z_0[t] = Z_0_pre[t]
            #     loss_res[t] = loss_res_pre[t]

        # update Z_1, Z_2
        A_1 = Z_0[:-1] + U_1
        A_2 = Z_0[1:] + U_2
        if not psi_node_penalty:
            A_add = A_2 + A_1
            A_sub = A_2 - A_1
            prox_e_1 = prox_psi(A_sub, lamda=2. * (1 - theta) / rho[:-1, None, None])
            prox_e_2 = prox_psi(A_sub, lamda=2. * (1 - theta) / rho[1:, None, None])
            Z_1 = .5 * (A_add - prox_e_1)
            Z_2 = .5 * (A_add + prox_e_2)
        # TODO: Fix for rho vector
        # else:
        #     if weights is not None:
        #         Z_1, Z_2 = prox_psi(
        #             np.concatenate((A_1, A_2), axis=1), lamda=.5 * (1 - theta) / rho[t],
        #             rho=rho[t], tol=tol, rtol=rtol, max_iter=max_iter)

        # update residuals
        con_obj_mean.append(np.mean(loss_res) ** 2)
        # con_obj_mean.append(np.max(loss_res) ** 2)
        con_obj_max.append(np.max(loss_res))

        U_1 += Z_0[:-1] - Z_1
        U_2 += Z_0[1:] - Z_2

        # diagnostics, reporting, termination checks
        rnorm = np.sqrt(
                    # squared_norm(loss_res) + 
                    squared_norm(Z_0[:-1] - Z_1) + 
                    squared_norm(Z_0[1:] - Z_2)
                )

        loss_res_old = loss_res.copy()

        snorm = np.sqrt(
                    squared_norm(rho[:-1, None, None] * (Z_1 - Z_1_old)) + 
                    squared_norm(rho[1:, None, None] * (Z_2 - Z_2_old))
                )
        e_dual = np.sqrt(2 * Z_1.size) * tol + rtol * np.sqrt(
                    squared_norm(rho[:-1, None, None] * U_1) + 
                    squared_norm(rho[1:, None, None] * U_2)
                )     

        obj = penalty_objective(Z_0, Z_1, Z_2, psi, theta)

        check = convergence(
            obj=obj,
            rnorm=rnorm,
            snorm=snorm,
            e_pri=np.sqrt(loss_res.size + 2 * Z_1.size) * tol + rtol * 
                (
                    # max(np.sqrt(squared_norm(loss_res + C)), np.sqrt(squared_norm(C))) + 
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

        if weights[0] is None:
            if len(con_obj_mean) > m:
                if np.mean(con_obj_mean[-m:-int(m/2)]) < np.mean(con_obj_mean[-int(m/2):]) and np.max(loss_res) > eps:
                # or np.mean(con_obj_max[-100:-50]) < np.mean(con_obj_max[-50:])) # np.mean(loss_res) > 0.25:
                    print("Rho Mult", mult * rho[0], iteration_, np.mean(loss_res), con_obj_max[-1])
                    # loss_diff /= 5            
                    # C_ = C - loss_diff           
                    # resscale scaled dual variables
                    rho = mult * rho
                    U_1 /= mult
                    U_2 /= mult
                    con_obj_mean = []
                    con_obj_max = []
        else:
            for t in col:
                rho *= weights[t]
                U_1 /= weights[t][:-1, None, None]
                U_2 /= weights[t][1:, None, None]
                con_obj[t] = []
                print('Mult', iteration_, t, rho[t])      
        
    else:
        warnings.warn("Objective did not converge.")

    print(iteration_, out_obj[-1])
    print(check.rnorm, check.e_pri)
    print(check.snorm, check.e_dual)

    covariance_ = np.array([linalg.pinvh(x) for x in Z_0])
    return_list = [Z_0, covariance_, out_obj]
    if return_history:
        return_list.append(checks)
    if return_n_iter:
        return_list.append(iteration_ + 1)
    return return_list


class GradientEqualTimeGraphicalLasso(GraphicalLasso):
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
        out = gradient_equal_time_graphical_lasso(
              emp_cov, self.emp_inv, max_iter=self.max_iter, loss=self.loss, C=self.C, 
              theta=self.theta, rho=self.rho, mult=self.mult, weights=self.weights, m=self.m, 
              eps=self.eps, psi=self.psi, gamma=self.gamma, tol=self.tol, rtol=self.rtol, 
              verbose=self.verbose, return_history=self.return_history, return_n_iter=True,  
              mode=self.mode, compute_objective=self.compute_objective, stop_at=self.stop_at,
              stop_when=self.stop_when, update_rho_options=self.update_rho_options)
        if self.return_history:
            self.precision_, self.covariance_, self.history_, self.n_iter_ = out
        else:
            self.precision_, self.covariance_, self.obj, self.n_iter_ = out
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

# # BSD 3-Clause License

# # Copyright (c) 2017, Federico T.
# # All rights reserved.

# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:

# # * Redistributions of source code must retain the above copyright notice, this
# #   list of conditions and the following disclaimer.

# # * Redistributions in binary form must reproduce the above copyright notice,
# #   this list of conditions and the following disclaimer in the documentation
# #   and/or other materials provided with the distribution.

# # * Neither the name of the copyright holder nor the names of its
# #   contributors may be used to endorse or promote products derived from
# #   this software without specific prior written permission.

# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# # AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# # FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# # DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# # SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# # CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# # OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# """Sparse inverse covariance selection over time via ADMM.

# More information can be found in the paper linked at:
# https://arxiv.org/abs/1703.01958
# """
# from __future__ import division

# from itertools import compress 

# import warnings
# import copy

# import numpy as np
# from scipy import linalg
# from six.moves import map, range, zip
# from sklearn.covariance import empirical_covariance, log_likelihood
# from sklearn.utils.extmath import squared_norm, fast_logdet
# from sklearn.utils.validation import check_X_y

# from regain.covariance.graphical_lasso_ import (
#     GraphicalLasso, neg_logl, dtrace)
# from regain.norm import l1_od_norm
# from regain.prox import soft_thresholding_od
# from regain.update_rules import update_rho
# from regain.utils import convergence, error_norm_time
# from regain.validation import check_norm_prox

# from scipy.optimize import minimize_scalar
# from functools import partial

# import pdb

# def loss_gen(loss, S, K):
#     T, p, _, = S.shape
#     losses = np.array([loss(S[i], K[i]) for i in range(T)])
#     return losses


# def penalty_objective(Z_0, Z_1, Z_2, psi, theta):
#     """Penalty-only objective function for time-varying graphical LASSO."""
#     return theta * sum(map(l1_od_norm, Z_0)) + (1 - theta) * sum(map(psi, Z_2 - Z_1))


# def rbf_weights(T, bandwidth, mult):
#     """RBF Weights"""
#     weights = np.zeros((T, T))
#     for t in range(T):
#         time_diff = np.arange(-t, T-t)
#         weights[t] = np.exp(-time_diff ** 2 / bandwidth) * (mult - 1) + 1
#     return weights


# def exp_weights(T, bandwidth, mult):
#     """Exponential Weights"""
#     weights = np.zeros((T, T))
#     for t in range(T):
#         time_diff = np.arange(-t, T-t)
#         weights[t] = np.exp(-np.abs(time_diff) / bandwidth) * (mult - 1) + 1
#     return weights


# def exp_weights_C(T, C, mult):
#     """Exponential Weights"""
#     weights = np.zeros((T, T))
#     C = np.array(C)
#     for t in range(T):
#         c_diff = C - C[t]
#         bandwidth = np.abs(np.min(c_diff))
#         c_mask = c_diff < 0
#         c_mask[t] = 1
#         weights[t] = c_mask * np.exp(-np.abs(c_diff) / bandwidth) * (mult - 1) + 1
#     return weights


# def lin_weights(T, bandwidth, mult):
#     """Linear Weights"""
#     weights = np.zeros((T, T))
#     for t in range(T):
#         time_diff = np.arange(-t, T-t)
#         weights[t] = (bandwidth - np.abs(time_diff)) / bandwidth * (mult - 1) + 1
#     return weights


# def gradient_equal_time_graphical_lasso(
#     S, K_init, max_iter, loss, C, theta, rho, mult, 
#     weights, m, eps, psi, gamma, tol, rtol, verbose, 
#     return_history, return_n_iter, mode, compute_objective, 
#     stop_at, stop_when, update_rho_options
#     ):
#     """Equality constrained time-varying graphical LASSO solver.

#     Solves the following problem via ADMM:
#         min sum_{i=1}^T ||K_i||_{od,1} + beta sum_{i=2}^T Psi(K_i - K_{i-1})
#         s.t. objective = c_i for i = 1, ..., T

#     where S_i = (1/n_i) X_i^T X_i is the empirical covariance of data
#     matrix X (training observations by features).

#     Parameters
#     ----------
#     emp_cov : ndarray, shape (n_features, n_features)
#         Empirical covariance of data.
#     alpha, beta : float, optional
#         Regularisation parameter.
#     rho : float, optional
#         Augmented Lagrangian parameter.
#     max_iter : int, optional
#         Maximum number of iterations.
#     n_samples : ndarray
#         Number of samples available for each time point.
#     gamma: float, optional
#         Kernel parameter when psi is chosen to be 'kernel'.
#     tol : float, optional
#         Absolute tolerance for convergence.
#     rtol : float, optional
#         Relative tolerance for convergence.
#     return_history : bool, optional
#         Return the history of computed values.
#     return_n_iter : bool, optional
#         Return the number of iteration before convergence.
#     verbose : bool, default False
#         Print info at each iteration.
#     update_rho_options : dict, optional
#         Arguments for the rho update.
#         See regain.update_rules.update_rho function for more information.
#     compute_objective : bool, default True
#         Choose to compute the objective value.
#     init : {'empirical', 'zero', ndarray}
#         Choose how to initialize the precision matrix, with the inverse
#         empirical covariance, zero matrix or precomputed.

#     Returns
#     -------
#     K : numpy.array, 3-dimensional (T x d x d)
#         Solution to the problem for each time t=1...T .
#     history : list
#         If return_history, then also a structure that contains the
#         objective value, the primal and dual residual norms, and tolerances
#         for the primal and dual residual norms at each iteration.

#     """
#     psi, prox_psi, psi_node_penalty = check_norm_prox(psi)

#     if loss == 'LL':
#         loss_func = neg_logl
#     else:
#         loss_func = dtrace

#     T = S.shape[0]
#     I = np.eye(S.shape[1])
    
#     Z_0 = K_init
#     Z_1 = Z_0.copy()[:-1] 
#     Z_2 = Z_0.copy()[1:]  

#     U_1 = np.zeros_like(Z_1)
#     U_2 = np.zeros_like(Z_2)

#     Z_0_old = Z_0.copy()
#     Z_1_old = np.zeros_like(Z_1)
#     Z_2_old = np.zeros_like(Z_2)

#     # divisor for consensus variables, accounting for one less matrix for t = 0 and t = T
#     divisor = np.full(T, 2, dtype=float)
#     divisor[0] -= 1
#     divisor[-1] -= 1

#     rho = rho * np.ones(T)    
#     if weights[0] is not None:
#         if weights[0] == 'rbf':
#             weights = rbf_weights(T, weights[1], mult)
#         elif weights[0] == 'exp':
#             weights = exp_weights(T, weights[1], mult)
#         elif weights[0] == 'lin':
#             weights = lin_weights(T, weights[1], mult)
#         con_obj = {}
#         for t in range(T):
#             con_obj[t] = []
    
#     # loss residuals
#     loss_res = np.zeros(T)
#     loss_init = loss_gen(loss_func, S, Z_0_old)
#     loss_res_old = loss_init - C

#     # loss_diff = C - loss_init
#     # C_  = C - loss_diff

#     con_obj_mean = []
#     con_obj_max = []
#     out_obj = []
    
#     checks = [
#         convergence(
#             obj=penalty_objective(Z_0, Z_1, Z_2, psi, theta))
#     ]

#     # soft-thresholded gradient modified Z_0_t
#     def _Z_0(x, A_t, nabla_t, rho_t, divisor_t):
#         _A_t = A_t - x * nabla_t 
#         return soft_thresholding_od(_A_t, lamda=theta / (rho_t * divisor_t))            


#     # # soft-thresholded gradient modified Z_0
#     # def _Z_0(x, A_t, nabla, rho, divisor):
#     #     _A_t = A_t - x * nabla 
#     #     return soft_thresholding_od(_A_t, lamda=theta / (rho * divisor)[:, None, None])            


#     # constrained optimisation via line search
#     def _f(x, _Z_0, A_t, nabla_t, rho_t, divisor_t, loss_func, S_t, c_t, loss_res_pre_t, Z_0_pre_t):
#         _Z_0_t = _Z_0(x, A_t, nabla_t, rho_t, divisor_t)
        
#         loss_res_t = loss_func(S_t, _Z_0_t) - c_t 
#         if loss_res_t == np.inf:
#             return np.inf
#         if loss_res_pre_t == np.inf:
#             loss_res_pre_t = 0
        
#         _nabla_Z_0_t = S_t - np.linalg.inv(_Z_0_t)

#         return 0.5 * (loss_res_t ** 2 - loss_res_pre_t ** 2) - x * (loss_res_t * np.sum(_nabla_Z_0_t * (_Z_0_t - Z_0_pre_t)) + theta * (l1_od_norm(_Z_0_t) - l1_od_norm(A_t))) # + 1e-3 * (l1_od_norm(_Z_0_t) - l1_od_norm(A_t))
#         # return (loss_res_t * np.sum(_nabla_Z_0_t * (_Z_0_t - A_t)) + theta * (l1_od_norm(A_t) - l1_od_norm(_Z_0_t))) # + 0.25 * (loss_res_t - loss_res_pre_t - np.sum(nabla_t * (_Z_0_t - Z_0_t))) ** 2


#     def _g(step_size_t, mu, _Z_0, A_t, nabla_t, rho_t, divisor_t, loss_func, S_t, c_t, loss_res_A_t):
#         _Z_0_t = _Z_0(step_size_t, A_t, nabla_t, rho_t, divisor_t)
#         _nabla_Z_0_t = S_t - np.linalg.inv(_Z_0_t)
#         loss_res_t = loss_func(S_t, _Z_0_t) - c_t 
        
#         pdb.set_trace()

#         while (0.5 * loss_res_t ** 2 - 0.5 * loss_res_A_t ** 2 - (loss_res_t * np.sum(_nabla_Z_0_t * gradient) + theta * (l1_od_norm(_Z_0_t) - l1_od_norm(A_t))) > 0):
#             step_size_t = mu * step_size_t
#             _Z_0_t = A_t + step_size_t * gradient
#             loss_res_t = loss_func(S_t, _Z_0_t) - c_t 

#         return step_size_t, gradient

    
#     # # max constrained optimisation via line search
#     # def _f(x, _Z_0, A, nabla, rho, divisor, loss_func, S, C, loss_res_pre, Z_0):
#     #     _Z_0 = _Z_0(x, A, nabla, rho, divisor)
#     #     loss_res = loss_gen(loss_func, S, _Z_0) - C
#     #     return np.max(loss_res ** 2) + np.max((loss_res - loss_res_pre - np.sum(nabla * (_Z_0 - Z_0), (1, 2))) ** 2)


#     for iteration_ in range(max_iter):
#         # update Z_0        
#         A = np.zeros_like(Z_0)
#         A[:-1] += Z_1 - U_1
#         A[1:] += Z_2 - U_2
#         A += A.transpose(0, 2, 1)
#         A /= 2. 
#         A /= divisor[:, None, None]

#         Z_0_pre = soft_thresholding_od(A, lamda=theta / (rho * divisor)[:, None, None])
#         loss_res_A = loss_gen(loss_func, S, A) - C
#         loss_res_pre = loss_gen(loss_func, S, Z_0_pre) - C

#         # if np.inf in loss_res_A:
#         #     pdb.set_trace()
        
#         if loss_func.__name__ == 'neg_logl':
#             # inv_A = np.array([np.linalg.inv(A_t) for A_t in A])
#             # pdb.set_trace()
#             nabla = np.array([S_t - np.linalg.inv(A_t) for (S_t, A_t) in zip(S, A)])

#             # nabla = np.array([S_t - np.linalg.inv(Z_0_t) for (S_t, Z_0_t) in zip(S, Z_0_old)])
#         elif loss_func.__name__ == 'dtrace': 
#             nabla = np.array([(2 * A_t @ S_t - I) for (S_t, A_t) in zip(S, A)])
#             # nabla = np.array([(2 * Z_0_t @ S_t - I) for (S_t, Z_0_t) in zip(S, Z_0_pre)])

#         # out = minimize_scalar(
#         #         partial(_f, _Z_0=_Z_0, A=A, nabla=nabla, rho=rho, 
#         #                 divisor=divisor, loss_func=loss_func, S=S,   
#         #                 C=C, loss_res_pre=loss_res_pre, Z_0=Z_0_pre)
#         #         )
#         # Z_0 = _Z_0(out.x, A, nabla, rho, divisor)
#         # loss_res = loss_gen(loss_func, S, Z_0_pre) - C

#         col = []

#         for t in range(T):
#             if loss_res_pre[t] > 0:
#                 # step_size_t = np.sum(nabla[t] * nabla[t]) / (np.sum(nabla[t] * inv_A[t]) ** 2)
#                 out = minimize_scalar(
#                         partial(_f, _Z_0=_Z_0, A_t=A[t], nabla_t=nabla[t], rho_t=rho[t], 
#                                 divisor_t=divisor[t], loss_func=loss_func, S_t=S[t],   
#                                 c_t=C[t], loss_res_pre_t=loss_res_pre[t], Z_0_pre_t=Z_0_pre[t]
#                                 ),
#                         # bounds = (-5, 5),
#                         # method = 'bounded'
#                         )
#                 # out = _g(step_size_t=step_size_t, mu=0.75, _Z_0=_Z_0, A_t=A[t], nabla_t=nabla[t], 
#                 #         rho_t=rho[t], divisor_t=divisor[t], loss_func=loss_func, S_t=S[t],   
#                 #         c_t=C[t], loss_res_A_t=loss_res_A[t])

#                 Z_0_t = _Z_0(out.x, A[t], nabla[t], rho[t], divisor[t])
#                 # Z_0_t = _Z_0(out, A[t], nabla[t], rho[t], divisor[t])
#                 loss_res_t  = loss_func(S[t], Z_0_t) - C[t]
#                 # if loss_res[t] != np.inf:
#                 #     Z_0[t] = Z_0_t
#                 #     loss_res[t] = loss_res_t
            
#                 if weights[0] is not None:
#                     con_obj[t].append(loss_res[t] ** 2)    
#                     if len(con_obj[t]) > m and np.mean(con_obj[t][-m:-int(m/2)]) < np.mean(con_obj[t][-int(m/2):]) and loss_res[t] > eps:
#                         col.append(t)
#             else:
#                 Z_0[t] = Z_0_pre[t]
#                 loss_res[t] = loss_res_pre[t]

#         # pdb.set_trace()

#         # update Z_1, Z_2
#         A_1 = Z_0[:-1] + U_1
#         A_2 = Z_0[1:] + U_2
#         if not psi_node_penalty:
#             A_add = A_2 + A_1
#             A_sub = A_2 - A_1
#             prox_e_1 = prox_psi(A_sub, lamda=2. * (1 - theta) / rho[:-1, None, None])
#             prox_e_2 = prox_psi(A_sub, lamda=2. * (1 - theta) / rho[1:, None, None])
#             Z_1 = .5 * (A_add - prox_e_1)
#             Z_2 = .5 * (A_add + prox_e_2)
#         # TODO: Fix for rho vector
#         # else:
#         #     if weights is not None:
#         #         Z_1, Z_2 = prox_psi(
#         #             np.concatenate((A_1, A_2), axis=1), lamda=.5 * (1 - theta) / rho[t],
#         #             rho=rho[t], tol=tol, rtol=rtol, max_iter=max_iter)

#         # update residuals
#         con_obj_mean.append(np.abs(np.mean(loss_res)))
#         # con_obj_mean.append(np.max(loss_res) ** 2)
#         con_obj_max.append(np.max(loss_res))

#         U_1 += Z_0[:-1] - Z_1
#         U_2 += Z_0[1:] - Z_2

#         # diagnostics, reporting, termination checks
#         rnorm = np.sqrt(
#                     squared_norm(loss_res) + 
#                     squared_norm(Z_0[:-1] - Z_1) + 
#                     squared_norm(Z_0[1:] - Z_2)
#                 )

#         loss_res_old = loss_res.copy()

#         snorm = np.sqrt(
#                     squared_norm(rho[:-1, None, None] * (Z_1 - Z_1_old)) + 
#                     squared_norm(rho[1:, None, None] * (Z_2 - Z_2_old))
#                 )
#         e_dual = np.sqrt(2 * Z_1.size) * tol + rtol * np.sqrt(
#                     squared_norm(rho[:-1, None, None] * U_1) + 
#                     squared_norm(rho[1:, None, None] * U_2)
#                 )     

#         obj = penalty_objective(Z_0, Z_1, Z_2, psi, theta)

#         check = convergence(
#             obj=obj,
#             rnorm=rnorm,
#             snorm=snorm,
#             e_pri=np.sqrt(loss_res.size + 2 * Z_1.size) * tol + rtol * 
#                 (
#                     max(np.sqrt(squared_norm(loss_res + C)), np.sqrt(squared_norm(C))) + 
#                     max(np.sqrt(squared_norm(Z_1)), np.sqrt(squared_norm(Z_0[:-1]))) + 
#                     max(np.sqrt(squared_norm(Z_2)), np.sqrt(squared_norm(Z_0[1:])))
#                 ),
#             e_dual=e_dual
#         )

#         Z_0_old = Z_0.copy()
#         Z_1_old = Z_1.copy()
#         Z_2_old = Z_2.copy()

#         if verbose:
#             print(
#                 "obj: %.4f, rnorm: %.4f, snorm: %.4f,"
#                 "eps_pri: %.4f, eps_dual: %.4f" % check[:5])

#         out_obj.append(penalty_objective(Z_0, Z_0[:-1], Z_0[1:], psi, theta))
#         if not iteration_ % 100:
#             print(iteration_)
#             print(np.max(loss_res), np.mean(loss_res))
#             print(out_obj[-1])
        
#         checks.append(check)

#         if stop_at is not None:
#             if abs(check.obj - stop_at) / abs(stop_at) < stop_when:
#                 break

#         if check.rnorm <= check.e_pri and check.snorm <= check.e_dual:
#             break

#         if weights[0] is None:
#             if len(con_obj_mean) > m:
#                 if np.mean(con_obj_mean[-m:-int(m/2)]) < np.mean(con_obj_mean[-int(m/2):]) and np.max(loss_res) > eps:
#                 # or np.mean(con_obj_max[-100:-50]) < np.mean(con_obj_max[-50:])) # np.mean(loss_res) > 0.25:
#                     print("Rho Mult", mult * rho[0], iteration_, np.mean(loss_res), con_obj_max[-1])
#                     # loss_diff /= 5            
#                     # C_ = C - loss_diff           
#                     # resscale scaled dual variables
#                     rho = mult * rho
#                     U_1 /= mult
#                     U_2 /= mult
#                     con_obj_mean = []
#                     con_obj_max = []
#         else:
#             for t in col:
#                 rho *= weights[t]
#                 U_1 /= weights[t][:-1, None, None]
#                 U_2 /= weights[t][1:, None, None]
#                 con_obj[t] = []
#                 print('Mult', iteration_, t, rho[t])      
        
#     else:
#         warnings.warn("Objective did not converge.")

#     print(iteration_, out_obj[-1])
#     print(check.rnorm, check.e_pri)
#     print(check.snorm, check.e_dual)

#     covariance_ = np.array([linalg.pinvh(x) for x in Z_0])
#     return_list = [Z_0, covariance_]
#     if return_history:
#         return_list.append(checks)
#     if return_n_iter:
#         return_list.append(iteration_ + 1)
#     return return_list


# class GradientEqualTimeGraphicalLasso(GraphicalLasso):
#     """Sparse inverse covariance estimation with an l1-penalized estimator.

#     Parameters
#     ----------
#     alpha : positive float, default 0.01
#         Regularization parameter for precision matrix. The higher alpha,
#         the more regularization, the sparser the inverse covariance.

#     beta : positive float, default 1
#         Regularization parameter to constrain precision matrices in time.
#         The higher beta, the more regularization,
#         and consecutive precision matrices in time are more similar.

#     psi : {'laplacian', 'l1', 'l2', 'linf', 'node'}, default 'laplacian'
#         Type of norm to enforce for consecutive precision matrices in time.

#     rho : positive float, default 1
#         Augmented Lagrangian parameter.

#     over_relax : positive float, deafult 1
#         Over-relaxation parameter (typically between 1.0 and 1.8).

#     tol : positive float, default 1e-4
#         Absolute tolerance to declare convergence.

#     rtol : positive float, default 1e-4
#         Relative tolerance to declare convergence.

#     max_iter : integer, default 100
#         The maximum number of iterations.

#     verbose : boolean, default False
#         If verbose is True, the objective function, rnorm and snorm are
#         printed at each iteration.

#     assume_centered : boolean, default False
#         If True, data are not centered before computation.
#         Useful when working with data whose mean is almost, but not exactly
#         zero.
#         If False, data are centered before computation.

#     time_on_axis : {'first', 'last'}, default 'first'
#         If data have time as the last dimension, set this to 'last'.
#         Useful to use scikit-learn functions as train_test_split.

#     update_rho_options : dict, default None
#         Options for the update of rho. See `update_rho` function for details.

#     compute_objective : boolean, default True
#         Choose if compute the objective function during iterations
#         (only useful if `verbose=True`).

#     init : {'empirical', 'zeros', ndarray}, default 'empirical'
#         How to initialise the inverse covariance matrix. Default is take
#         the empirical covariance and inverting it.

#     Attributes
#     ----------
#     covariance_ : array-like, shape (n_times, n_features, n_features)
#         Estimated covariance matrix

#     precision_ : array-like, shape (n_times, n_features, n_features)
#         Estimated precision matrix.

#     n_iter_ : int
#         Number of iterations run.

#     """
#     def __init__(
#             self, max_iter=1000, loss='LL', c_level=None, theta=0.5,
#             rho=1e2, mult=2, weights=None, m=100, eps=2, psi='laplacian', 
#             gamma=None, tol=1e-4, rtol=1e-4, mode='admm', 
#             verbose=False, assume_centered=False, return_history=False, 
#             update_rho_options=None, compute_objective=True, stop_at=None, 
#             stop_when=1e-4, suppress_warn_list=False):
#         super().__init__(
#             alpha=1., rho=rho, tol=tol, rtol=rtol, max_iter=max_iter,
#             verbose=verbose, assume_centered=assume_centered, mode=mode,
#             update_rho_options=update_rho_options,
#             compute_objective=compute_objective)
#         self.max_iter = max_iter
#         self.loss = loss
#         self.c_level = c_level
#         self.theta = theta
#         self.rho = rho
#         self.mult = mult
#         self.weights = weights
#         self.m = m
#         self.eps = eps
#         self.psi = psi
#         self.gamma = gamma
#         self.return_history = return_history
#         self.stop_at = stop_at
#         self.stop_when = stop_when
#         self.suppress_warn_list = suppress_warn_list


#     def get_observed_precision(self):
#         """Get the observed precision matrix.

#         Returns
#         -------
#         precision_ : array-like,
#             The precision matrix associated to the current covariance object.

#         """
#         return self.get_precision()


#     def _fit(self, emp_cov):
#         """Fit the InequalityTimeGraphicalLasso model to X.

#         Parameters
#         ----------
#         emp_cov : ndarray, shape (n_time, n_features, n_features)
#             Empirical covariance of data.

#         """
#         out = gradient_equal_time_graphical_lasso(
#               emp_cov, self.emp_inv, max_iter=self.max_iter, loss=self.loss, C=self.C, 
#               theta=self.theta, rho=self.rho, mult=self.mult, weights=self.weights, m=self.m, 
#               eps=self.eps, psi=self.psi, gamma=self.gamma, tol=self.tol, rtol=self.rtol, 
#               verbose=self.verbose, return_history=self.return_history, return_n_iter=True,  
#               mode=self.mode, compute_objective=self.compute_objective, stop_at=self.stop_at,
#               stop_when=self.stop_when, update_rho_options=self.update_rho_options)
#         if self.return_history:
#             self.precision_, self.covariance_, self.history_, self.n_iter_ = out
#         else:
#             self.precision_, self.covariance_, self.n_iter_ = out
#         return self


#     def fit_obs(self, X, y):
#         """Fit the InequalityTimeGraphicalLasso model to observations.

#         Parameters
#         ----------
#         X : ndarray, shape = (n_samples * n_times, n_dimensions)
#             Data matrix.
#         y : ndarray, shape = (n_times,)
#             Indicate the temporal belonging of each sample.

#         """
#         # Covariance does not make sense for a single feature
        
#         X, y = check_X_y(
#             X, y, accept_sparse=False, dtype=np.float64, order="C",
#             ensure_min_features=2, estimator=self)

#         n_dimensions = X.shape[1]
#         self.classes_, n_samples = np.unique(y, return_counts=True)
#         n_times = self.classes_.size

#         # n_samples = np.array([x.shape[0] for x in X])
#         if self.assume_centered:
#             self.location_ = np.zeros((n_times, n_dimensions))
#         else:
#             self.location_ = np.array(
#                 [X[y == cl].mean(0) for cl in self.classes_])

#         emp_cov = []
#         emp_inv_score = []
#         sam_inv_score = []

    
#         for i in self.classes_:
#             emp_cov_i = empirical_covariance(X[y == i], assume_centered=self.assume_centered)
#             emp_inv_i = np.linalg.inv(emp_cov_i)
#             _, log_det = np.linalg.slogdet(emp_inv_i)
#             emp_cov.append(emp_cov_i)
#             emp_inv_score.append(log_det - np.trace(emp_cov_i @ emp_inv_i))
#             sam_inv_score.append(log_det - np.array([np.trace((X[y == i][[j], :].T @ X[y == i][[j], :]) @ emp_inv_i) for j in range(int(n_samples))]))

#         self.emp_inv_score = np.array(emp_inv_score)
#         self.sam_inv_score = np.array(sam_inv_score)

#         self.constrained_to = None

#         return self._fit(np.array(emp_cov), n_samples)


#     def fit_cov(self, X):
#         """Fit the InequalityTimeGraphicalLasso model to covariances.

#         Parameters
#         ----------
#         X : ndarray, shape = (n_dimensions, n_dimensions, n_samples, time_steps)
        
#         """
#         n_dimensions, _, n_samples, time_steps = X.shape
        
#         self.emp_cov = []
#         self.emp_inv = []
#         self.emp_inv_score = []
#         self.sam_inv_score = []
#         self.C = []

#         for i in range(time_steps):
#             self.emp_cov.append(np.mean(X[:, :, :, i], 2))
#             self.emp_inv.append(np.linalg.inv(self.emp_cov[i]))
#             if self.loss == 'LL':
#                 self.emp_inv_score.append(neg_logl(self.emp_cov[i], self.emp_inv[i]))
#                 self.sam_inv_score.append(np.array([neg_logl(X[:, :, j, i], self.emp_inv[i]) for j in range(n_samples)]))
#             else:
#                 self.emp_inv_score.append(dtrace(self.emp_cov[i], self.emp_inv[i]))
#                 self.sam_inv_score.append(np.array([dtrace(X[:, :, j, i], self.emp_inv[i]) for j in range(n_samples)]))
#             self.C.append(np.quantile(self.sam_inv_score[i], 1 - self.c_level, 0))

#         self.emp_cov = np.array(self.emp_cov)
#         self.emp_inv = np.array(self.emp_inv)

#         self.emp_inv_score = np.array(self.emp_inv_score)
#         self.sam_inv_score = np.array(self.sam_inv_score)

#         return self._fit(self.emp_cov)


#     def score(self, X, y):
#         """Computes the log-likelihood of a Gaussian data set with
#         `self.covariance_` as an estimator of its covariance matrix.

#         Parameters
#         ----------
#         X : array-like, shape = (n_samples, n_features)
#             Test data of which we compute the likelihood, where n_samples is
#             the number of samples and n_features is the number of features.
#             X is assumed to be drawn from the same distribution than
#             the data used in fit (including centering).

#         y :  array-like, shape = (n_samples,)
#             Class of samples.

#         Returns
#         -------
#         res : float
#             The likelihood of the data set with `self.covariance_` as an
#             estimator of its covariance matrix.

#         """
#         # Covariance does not make sense for a single feature
#         X, y = check_X_y(
#             X, y, accept_sparse=False, dtype=np.float64, order="C",
#             ensure_min_features=2, estimator=self)

#         # compute empirical covariance of the test set
#         test_cov = np.array(
#             [
#                 empirical_covariance(
#                     X[y == cl] - self.location_[i], assume_centered=True)
#                 for i, cl in enumerate(self.classes_)
#             ])

#         res = sum(
#             X[y == cl].shape[0] * log_likelihood(S, K) for S, K, cl in zip(
#                 test_cov, self.get_observed_precision(), self.classes_))

#         return res


#     def eval_obs_pre(self, X, y):
#         """Evaluate the log likelihood of estimated precisions compared to the inverse sample covariance at each time step

#         Parameters
#         ----------
#         X : ndarray, shape = (n_samples * n_times, n_dimensions)
#             Data matrix.
#         y : ndarray, shape = (n_times,)
#             Indicate the temporal belonging of each sample.
#         """
        
#         X, y = check_X_y(
#             X, y, accept_sparse=False, dtype=np.float64, order="C",
#             ensure_min_features=2, estimator=self)

#         n_dimensions = X.shape[1]
#         self.classes_, n_samples = np.unique(y, return_counts=True)
#         n_times = self.classes_.size

#         # n_samples = np.array([x.shape[0] for x in X])
#         if self.assume_centered:
#             self.location_ = np.zeros((n_times, n_dimensions))
#         else:
#             self.location_ = np.array(
#                 [X[y == cl].mean(0) for cl in self.classes_])

#         precisions = self.get_observed_precision()

#         emp_pre_score = []
#         sam_pre_score = []
#         slack = []

#         for i in range(precisions.shape[0]):
#             emp_cov = empirical_covariance(X[y == i] - self.location_[i], assume_centered=True)
#             precision = precisions[i, :, :]
#             # slack.append(-self.constrained_to[i] + logl(emp_cov, precision))
#             _, log_det = np.linalg.slogdet(precision)
#             emp_pre_score.append(log_det - np.trace(emp_cov @ precision))
#             sam_pre_score.append(log_det - np.array([np.trace((X[y == i][[j], :].T @ X[y == i][[j], :]) @ precision) for j in range(int(n_samples))]))

#         return self.emp_inv_score - np.array(emp_pre_score), self.sam_inv_score - np.array(sam_pre_score), precision


#     def eval_cov_pre(self):
#         """Evaluate the log likelihood of estimated precisions compared to the inverse sample covariance at each time step

#         Parameters
#         ----------
#         X : ndarray, shape = (n_dimensions, n_dimensions, n_samples, time_steps)
#         """
        
#         precisions = self.precision_

#         fit_score = []

#         for i in range(precisions.shape[0]):
#             precision = precisions[i]
#             if self.loss == 'LL':
#                 fit_score.append(neg_logl(self.emp_cov[i], precision))
#             else:
#                 fit_score.append(dtrace(self.emp_cov[i], precision))

#         return self.emp_inv_score, self.C, fit_score, precisions 


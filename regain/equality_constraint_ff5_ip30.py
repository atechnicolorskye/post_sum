import numpy as np
import pandas as pd
from numpy.random import multivariate_normal as mvnorm
from numpy.linalg import norm

from regain.covariance import TimeGraphicalLasso
from regain.norm import l1_od_norm
from regain.validation import check_norm_prox

import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20.colors)

import time
import warnings
warnings.filterwarnings('ignore')

# Set seed
seed = 0
np.random.seed(seed)

# Import data
data = pd.read_pickle("../ff5_30_nonsmooth_standard_4000_draws.pkl")

# Restrict to 100 time points
X = data[10:110].transpose(2, 1, 0)
X_cov = np.einsum('ijkl,jmkl->imkl', np.expand_dims(X, 1), np.expand_dims(X, 0))

# Get empirical inverse
n_dimensions, _, n_samples, time_steps = X_cov.shape
emp_inv = []
for i in range(time_steps):
    emp_inv.append(np.linalg.inv(np.mean(X_cov[:, :, :, i], 2)))
emp_inv = np.array(emp_inv)


def penalty_objective(Z_0, Z_1, Z_2, psi, theta):
    """Penalty-only objective function for time-varying graphical LASSO."""
    return theta * sum(map(l1_od_norm, Z_0)) + (1 - theta) * sum(map(psi, Z_2 - Z_1))


tic = time.perf_counter()
tgl = TimeGraphicalLasso(alpha=1., beta=1., mode='admm', rho=1, tol=1e-4,
            rtol=1e-4, psi='laplacian', max_iter=3000, verbose=False, assume_centered=False, 
            return_history=False, update_rho_options=None, compute_objective=True, 
            stop_at=None, stop_when=1e-4, suppress_warn_list=False, init='empirical')
fit_score_, pre_ = tgl.fit_cov(X_cov).eval_cov_pre()   
toc = time.perf_counter()
print('Vanilla Running Time :{}'.format(toc - tic))

pre_cvx = np.load("mosek_sol_ff5_30_nonsmooth_standard_alpha_0.2_laplacian.npy")

for _weights in [['lin', 2e2], ['exp', 2e2], ['rbf', 2e2]]:
    # Set parameters
    max_iter = 10000
    loss = 'LL'
    c_level = 0.2
    theta = 0.5
    rho = 1e2
    mult = 1.1
    weights = _weights
    m = 50
    eps = 2
    psi = 'laplacian'

    from regain.covariance import TaylorTimeGraphicalLasso
    tic = time.perf_counter()    
    tgl_tp = TaylorTimeGraphicalLasso(max_iter=max_iter, loss=loss, c_level=c_level, theta=theta, rho=rho, mult=mult, weights=weights, m=20, eps=eps, psi=psi)
    emp_inv_score_tp, baseline_score_tp, fit_score_tp, pre_tp = tgl_tp.fit_cov(X_cov).eval_cov_pre() 
    toc = time.perf_counter()
    print('Full Running Time :{}'.format(toc - tic))

    from regain.covariance import TaylorEqualTimeGraphicalLasso
    tic = time.perf_counter()
    tgl_tpe = TaylorEqualTimeGraphicalLasso(max_iter=max_iter, loss=loss, c_level=c_level, theta=theta, rho=rho, mult=mult, weights=weights, m=m, eps=eps, psi=psi)
    emp_inv_score_tpe, baseline_score_tpe, fit_score_tpe, pre_tpe = tgl_tpe.fit_cov(X_cov).eval_cov_pre() 
    toc = time.perf_counter()
    print('Linear Running Time :{}'.format(toc - tic))

    from regain.covariance import GradientEqualTimeGraphicalLasso 
    tic = time.perf_counter()
    tgl_g = GradientEqualTimeGraphicalLasso(max_iter=max_iter, loss=loss, c_level=c_level, theta=theta, rho=rho, mult=mult, weights=weights, m=m, eps=eps, psi=psi)
    emp_inv_score_g, baseline_score_g, fit_score_g, pre_g = tgl_g.fit_cov(X_cov).eval_cov_pre() 
    toc = time.perf_counter()
    print('Gradient Running Time :{}'.format(toc - tic))

    psi, prox_psi, psi_node_penalty = check_norm_prox(tgl_g.psi)

    pre_tgl = {}
    fit_score_tgl_thres = {}
    for i in [1e-4]:
        pre_tgl[i] = np.array([k * (np.abs(k) >= i) for k in pre_])
        tgl_g.precision_ = pre_tgl[i]
        emp_inv_score, baseline_score, fit_score_tgl_thres[i], _ = tgl_g.eval_cov_pre() 
        print('Vanilla Objective', penalty_objective(pre_tgl[i], pre_tgl[i][:-1], pre_tgl[i][1:], psi, tgl_g.theta))

    pre = {}
    fit_score_thres = {}
    for i in [1e-4]:
        pre[i] = np.array([k * (np.abs(k) >= i) for k in pre_cvx])
        tgl_g.precision_ = pre[i]
        emp_inv_score, baseline_score, fit_score_thres[i], _ = tgl_g.eval_cov_pre() 
        print('MOSEK Objective', penalty_objective(pre[i], pre[i][:-1], pre[i][1:], psi, tgl_g.theta))

    pre_tp_thres = {}
    fit_score_tp_thres = {}
    for i in [1e-4]:
        pre_tp_thres[i] = np.array([k * (np.abs(k) >= i) for k in pre_tp])
        tgl_tp.precision_ = pre_tp_thres[i]
        emp_inv_score, baseline_score, fit_score_tp_thres[i], _ = tgl_tp.eval_cov_pre() 
        print('Full Objective', penalty_objective(pre_tp_thres[i], pre_tp_thres[i][:-1], pre_tp_thres[i][1:], psi, tgl_tp.theta))

    pre_tpe_thres = {}
    fit_score_tpe_thres = {}
    for i in [1e-4]:
        pre_tpe_thres[i] = np.array([k * (np.abs(k) >= i) for k in pre_tpe])
        tgl_tpe.precision_ = pre_tpe_thres[i]
        emp_inv_score, baseline_score, fit_score_tpe_thres[i], _ = tgl_tpe.eval_cov_pre() 
        print('Linear Objective', penalty_objective(pre_tpe_thres[i], pre_tpe_thres[i][:-1], pre_tpe_thres[i][1:], psi, tgl_tpe.theta))

    pre_g_thres = {}
    fit_score_g_thres = {}
    for i in [1e-4]:
        pre_g_thres[i] = np.array([k * (np.abs(k) >= i) for k in pre_g])
        tgl_g.precision_ = pre_g_thres[i]
        emp_inv_score, baseline_score, fit_score_g_thres[i], _ = tgl_g.eval_cov_pre() 
        print('Gradient Objective', penalty_objective(pre_g_thres[i], pre_g_thres[i][:-1], pre_g_thres[i][1:], psi, tgl_g.theta))

    print('Vanilla Max', np.max(np.array(fit_score_thres[1e-4]) - baseline_score_g))
    print('MOSEK Max', np.max(np.array(fit_score_thres[1e-4]) - baseline_score_g))
    print('Full Max', np.max(np.array(fit_score_tp_thres[1e-4]) - baseline_score_g))
    print('Linear Max', np.max(np.array(fit_score_tpe_thres[1e-4]) - baseline_score_g))
    print('Gradient Max', np.max(np.array(fit_score_g_thres[1e-4]) - baseline_score_g))

    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('white')

    ax.plot(range(X_cov.shape[-1]), baseline_score_g, color='k', label=r'Constraint')
    mean_diff = np.mean(np.array(fit_score_thres[1e-4]) - baseline_score)
    ax.plot(range(X_cov.shape[-1]), fit_score_thres[1e-4], color='r', alpha=0.5, 
            label=r'Constrained MOSEK, Thres = {}, Mean Diff = {:.3f}'.format(1e-4, mean_diff))
    mean_diff = np.mean(np.array(fit_score_tp_thres[1e-4]) - baseline_score_g)
    ax.plot(range(X_cov.shape[-1]), fit_score_tp_thres[1e-4], alpha=0.5, color='m',
            label=r'Constrained ADMM Full, Mean Diff = {:.3f}'.format(mean_diff))
    mean_diff = np.mean(np.array(fit_score_tpe_thres[1e-4]) - baseline_score_g)
    ax.plot(range(X_cov.shape[-1]), fit_score_tpe_thres[1e-4], alpha=0.5, color='b',
            label=r'Constrained ADMM Linear, Mean Diff = {:.3f}'.format(mean_diff))
    mean_diff = np.mean(np.array(fit_score_g_thres[1e-4]) - baseline_score_g)
    ax.plot(range(X_cov.shape[-1]), fit_score_g_thres[1e-4], alpha=0.5, color='g',
            label=r'Constrained ADMM Gradient, Mean Diff = {:.3f}'.format(mean_diff))

    fig.legend(fontsize=15, loc='lower right', bbox_to_anchor=(0.495, 0.08, 0.5, 0.5))
    ax.set_ylabel('Negative Log Likelihood', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Time t', fontsize=15)
    ax.set_title(r'NLL for Constraint and Constrained MOSEK/ADMM', fontsize=20)
    plt.tight_layout()
    plt.savefig('ff5_ip30_10000_' + str(weights[0]) + '_diff_like.pdf')


    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('white')

    ax.plot(range(X_cov.shape[-1]), [sum(sum(abs(emp_inv[k]) > 0)) for k in range(X_cov.shape[-1])], 
            color='k', label=r'Empirical Inverse')
    supp = [sum(sum(abs(pre[1e-4][t]) > 0)) for t in range(X_cov.shape[-1])]
    mean_supp = np.mean(supp)
    ax.plot(range(X_cov.shape[-1]), supp, color='r', alpha=0.5, 
            label=r'Constrained MOSEK, Thres = {}, Mean Supp = {}'.format(1e-4, mean_supp))
    supp = [sum(sum(abs(pre_tp_thres[1e-4][t]) > 0)) for t in range(X_cov.shape[-1])]
    ax.plot(range(X_cov.shape[-1]), supp, color='m', alpha=0.5,
            label=r'Constrained ADMM Full, Mean Supp = {:.3f}'.format(np.mean(supp)))
    supp = [sum(sum(abs(pre_tpe_thres[1e-4][t]) > 0)) for t in range(X_cov.shape[-1])]
    ax.plot(range(X_cov.shape[-1]), supp, color='b', alpha=0.5,
            label=r'Constrained ADMM Linear, Mean Supp = {:.3f}'.format(np.mean(supp)))
    supp = [sum(sum(abs(pre_g_thres[1e-4][t]) > 0)) for t in range(X_cov.shape[-1])]
    ax.plot(range(X_cov.shape[-1]), supp, color='g', alpha=0.5,
            label=r'Constrained ADMM Gradient, Mean Supp = {:.3f}'.format(np.mean(supp)))

    fig.legend(fontsize=15, loc='lower left', bbox_to_anchor=(0.05, 0.08, 0.5, 0.5))
    ax.set_ylabel('Support', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Time t', fontsize=15)
    ax.set_title(r'Support for Empirical Inverse and Constrained MOSEK/ADMM', fontsize=20)
    plt.tight_layout()
    plt.savefig('ff5_ip30_10000_' + str(weights[0]) + '_diff_support.pdf')


    fig, ax = plt.subplots(figsize=(20, 10))
    fig.patch.set_facecolor('white')

    diff = [norm(pre[1e-4][t] - pre[1e-4][t-1], 'fro') for t in range(1, X_cov.shape[-1])]
    mean_diff = np.mean(diff)
    ax.plot(range(1, X_cov.shape[-1]), diff, color='r', alpha=0.5, 
            label=r'Constrained MOSEK, Thres = {}, Mean Diff = {:.3f}'.format(1e-4, mean_diff))
    diff_grad = [norm(pre_tp_thres[1e-4][t] - pre_tp_thres[1e-4][t-1], 'fro') for t in range(1, X_cov.shape[-1])]
    ax.plot(range(1, X_cov.shape[-1]), diff_grad, color='m', alpha=0.5,
            label=r'Constraint ADMM Full, Mean Diff = {:.3f}'.format(np.mean(diff_grad)))
    diff_grad = [norm(pre_tpe_thres[1e-4][t] - pre_tpe_thres[1e-4][t-1], 'fro') for t in range(1, X_cov.shape[-1])]
    ax.plot(range(1, X_cov.shape[-1]), diff_grad, color='b', alpha=0.5,
            label=r'Constraint ADMM Linear, Mean Diff = {:.3f}'.format(np.mean(diff_grad)))
    diff_grad = [norm(pre_g_thres[1e-4][t] - pre_g_thres[1e-4][t-1], 'fro') for t in range(1, X_cov.shape[-1])]
    ax.plot(range(1, X_cov.shape[-1]), diff_grad, color='g', alpha=0.5,
            label=r'Constraint ADMM Gradient, Mean Diff = {:.3f}'.format(np.mean(diff_grad)))

    fig.legend(fontsize=15, loc='upper right', bbox_to_anchor=(0.495, 0.45, 0.5, 0.5))
    ax.set_ylabel('Difference in Frobenius Norm', fontsize=15)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel('Time t', fontsize=15)
    ax.set_title(r'Difference in Frobenius Norm for Constrained MOSEK/ADMM', fontsize=20)
    plt.tight_layout()
    plt.savefig('ff5_ip30_10000_' + str(weights[0]) + '_diff_fro.pdf')
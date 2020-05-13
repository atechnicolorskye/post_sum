library(Matrix)
library(mvtnorm)

library(reticulate)
use_python("/usr/bin/python")
# use_python("/apps/python/3.6/3.6.3/bin/python3")
np <- import("numpy")

library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

seed <- 1
set.seed(seed)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_T <- 2
n_N <- 500
n_F <- 5
n_X <- 10
n_D <- 2000
n_W <- 1000
R <- 0.5
AD <- 0.99
TD <- 15
# SS <- 0.1
discount <- 0.99

# Create data strucutures
X <- array(0, c(n_T, n_N, n_X))
B <- array(0, c(n_T, n_X, n_F))
log_F_sigma <- array(0, c(n_T, n_F))

# x hyperparameters 
# x_sigma
x_sigma <- exp(rep(rnorm(1), n_X))
# y_sigma <- matrix(rep(1. / rgamma(1, 3, 1), n_Y), c(n_Y, 1))

# f hyperparameters
# Covariance
log_F_sigma_loc <- rnorm(n_F, mean=-0, sd=0.25)
log_F_sigma_scale <- abs(rnorm(n_F, mean=0, sd=0.25))
# chol_F_sigma <- diag(abs(rnorm(n_F, mean=0, sd=0.1)))
chol_f_sigma <- matrix(rnorm(n_F * n_F, sd=0.1), c(n_F, n_F))
diag(chol_f_sigma) <- abs(diag(chol_f_sigma))

# B hyperparameters
# Want B to be 'almost sparse' where B contains mostly values close to 0.05
base_order <- 0.05
bias_order <- 0.5
p_connect <- 0.3
B_init <- matrix(0, n_X, n_F)
diag(B_init) <- 1
n_l_t <- sum(lower.tri(B_init))
B_init[lower.tri(B_init)] <- rnorm(n_l_t, 0, .05) +
  rbinom(n_l_t, 1, .3) * (2 * rbinom(n_l_t, 1, .5) - 1) * 
  # rnorm(n_l_t, bias_order, 1)
  (bias_order + abs(rnorm(n_l_t)))

# Initialise
log_F_sigma[1, ] <- log_F_sigma_loc + chol_F_sigma %*% rnorm(n_F)
B[1, , ] <- B_init
# X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
for (j in 1:n_N){
  X[1, j, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
}


# Generate data
for (i in 2:n_T){
  log_F_sigma[i, ] <- log_F_sigma_loc + log_F_sigma_scale * (log_F_sigma[i-1, ] - log_F_sigma_loc) + chol_F_sigma %*% rnorm(n_F)
  B[i, , ] <- B[i-1, , ] # + discount * base_order * lower.tri(B[i, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
  # X[i,  ] <- (B[i, , ] %*% (exp(log_F_sigma[i, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
  for (j in 1:n_N){
    X[i, j,  ] <- (B[i, , ] %*% (exp(log_F_sigma[i, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
  }
}

# Standardise X
# X_mu <- colMeans(X, dims = 1)
# X_sd <- apply(X, 2, sd)
# X_s <- t((t(X) - X_mu) / X_sd)
# X_s <- scale(X)

# matplot(X, type='l')

fit <- stan(file='infer_tv_base_mv.stan', data=list(P=n_X, F=n_F, N=n_N, TS=n_T, disc=discount, x=X), iter=n_D, warmup=n_W, 
            refresh=n_W, init_r=R, seed=seed, chains=1, control=list(adapt_delta=AD, max_treedepth=TD))

# fit <- stan(file='infer_tv.stan', data=list(P=n_X, F=n_F, TS=n_T, N=n_N, disc=discount, x=X), iter=n_D, warmup=n_W, 
# refresh=100, seed=seed, chains=1, control=list(adapt_delta=AD, max_treedepth=TD))

# summary(fit, pars=c('x_sd', 'chol_log_f_sd'))
par_sum <- summary(fit, pars=c('x_sd', 'chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'))$summary
print(par_sum)

print(x_sigma)
print(chol_F_sigma) 
print(log_F_sigma_loc) 
print(log_F_sigma_scale)

# monitor(extract(fit, pars=c('x_sd', 'chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'), 
#                 permuted = FALSE, inc_warmup = FALSE), digits_summary=5)

saveRDS(fit, "draws.rds")
# params <- extract(fit)
# print(colMeans(params$x_sd, dims=1))
# print(colMeans(params$log_f_sd_sd=1))

# Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

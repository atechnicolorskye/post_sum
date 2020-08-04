rm(list=ls())
gc()

library(posterior)
library(tidyverse)
options(tibble.print_max = Inf)

# library(reticulate)
# use_python("/usr/bin/python")
# np <- import("numpy")

library(cmdstanr)
library(rstan)
options(mc.cores = parallel::detectCores())

seed <- 1
set.seed(seed)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_T <- 300
n_F <- 3
n_X <- 10
n_D <- 3000
n_W <- 1000
R <- 0.5
AD <- 0.95
TD <- 11
# SS <- 0.01
discount <- 1

# Create data strucutures
X <- array(0, c(n_T, n_X))
B <- array(0, c(n_T, n_X, n_F))
log_F_sigma <- array(0, c(n_T, n_F))
log_X_sigma <- array(0, c(n_T, n_X))

# x hyperparameters
# x_sigma <- exp(rep(rnorm(1), n_X))
x_sigma <- abs(rep(rnorm(1, sd=0.5), n_X)) # abs(rnorm(n_X, sd=0.5)) 
# log_X_sigma_loc <- rnorm(n_X, mean=0, sd=0.25)
# log_X_sigma_scale <- abs(rnorm(n_X, mean=0, sd=0.25))
# chol_X_sigma <- matrix(0, n_X, n_X)
# diag(chol_X_sigma) <- abs(rnorm(n_X, mean=0, sd=0.1))

# f hyperparameters
log_F_sigma_loc <- rep(0, n_F) # rnorm(n_F, sd=0.3)
# log_F_sigma_scale <- rnorm(n_F, mean=0, sd=0.25)
log_F_sigma_scale <- rbeta(n_F, 2, 6)
# chol_F_sigma <- diag(abs(rnorm(n_F, mean=0, sd=0.1)))
# chol_F_sigma <- matrix(rnorm(n_F * n_F, sd=0.2), c(n_F, n_F))
# chol_F_sigma[upper.tri(chol_F_sigma)] <- 0
# diag(chol_F_sigma) <- abs(diag(chol_F_sigma))
chol_F_sigma <- matrix(rep(0, n_F * n_F), c(n_F, n_F))
diag(chol_F_sigma) <- abs(rep(rnorm(1, sd=0.1), n_F)) # abs(rnorm(n_F, sd=0.1))

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
log_F_sigma[1, ] <- log_F_sigma_loc + chol_F_sigma %*% rnorm(n_F) # log_F_sigma_loc * rnorm(n_F)
B[1, , ] <- B_init
# log_X_sigma[1, ] <- log_X_sigma_loc + chol_X_sigma %*% rnorm(n_X)
X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
# X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + exp(log_X_sigma[1, ]) * rnorm(n_X)

# Generate data
for (i in 2:n_T){
  # log_F_sigma[i, ] <- log_F_sigma_loc + (log_F_sigma_scale * log_F_sigma[i-1, ]) + chol_F_sigma %*% rnorm(n_F)
  log_F_sigma[i, ] <- log_F_sigma_loc + (log_F_sigma_scale * (log_F_sigma[i-1, ] - log_F_sigma_loc)) + chol_F_sigma %*% rnorm(n_F)
  B[i, , ] <- B[i-1, , ]
  # log_X_sigma[i, ] <- log_X_sigma_loc + (log_X_sigma_scale * log_X_sigma[i-1, ]) + chol_X_sigma %*% rnorm(n_X)
  X[i,  ] <- (B[i, , ] %*% (exp(log_F_sigma[i, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
  # X[i,  ] <- (B[i, , ] %*% (exp(log_F_sigma[i, ]) * rnorm(n_F))) + exp(log_X_sigma[i, ]) * rnorm(n_X)
}

# Standardise X
X_mu <- colMeans(X, dims=1)
X_sd <- apply(X, 2, sd)
X_s <- scale(X)

matplot(X, type='l')
matplot(log_F_sigma, type='l')
# matplot(log_X_sigma, type='l')

base <- cmdstan_model('infer_tv_base.stan', quiet=FALSE, force_recompile=TRUE)
fit_b <- base$sample(data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X),
                     num_samples=n_D,
                     num_warmup=n_W,
                     refresh=1000,
                     # init=R,
                     seed=seed,
                     adapt_delta=AD,
                     max_depth=TD,
                     # stepsize=SS,
                     num_chains=1,
                     num_cores=4)

# stanfit <- stan(file='infer_tv_base.stan',
#             data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X),
#             iter=(n_D+n_W),
#             warmup=n_W,
#             refresh=100,
#             seed=seed,
#             chains=1,
#             control=list(adapt_delta=AD, max_treedepth=TD))
# 
# stanfit_alt <- stan(file='infer_tv_base_alt.stan',
#                 data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X),
#                 iter=(n_D+n_W),
#                 warmup=n_W,
#                 refresh=100,
#                 seed=seed,
#                 chains=1,
#                 control=list(adapt_delta=AD, max_treedepth=TD))
# 
# params = rnorm(get_num_upars(stanfit))
# sum(abs(grad_log_prob(stanfit, params) - grad_log_prob(stanfit_alt, params)))

stanfit <- read_stan_csv(fit_b$output_files())
# par_sum <- summary(stanfit, pars=c('x_sd', 'chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'))$summary
par_sum <- summary(stanfit, pars=c('x_sd', 'chol_log_f_sd_sd', 'log_f_sd_scale'))$summary

# print(x_sigma)
# print(chol_F_sigma)
# print(log_F_sigma_loc)
# print(log_F_sigma_scale)

params <- extract(stanfit)
# x_sigma_hat <- colMeans(params$x_sd, dims=1)
beta_hat <- colMeans(params$beta, dims=1)
chol_F_sigma_hat <- colMeans(params$chol_log_f_sd_sd, dims=1)
# log_F_sigma_loc_hat <- colMeans(params$log_f_sd_loc, dims=1)
log_F_sigma_scale_hat <- colMeans(params$log_f_sd_scale, dims=1)
# chol_X_sigma_hat <- colMeans(params$chol_log_x_sd, dims=1)
# log_X_sigma_loc_hat <- colMeans(params$log_x_sd_loc, dims=1)
# log_X_sigma_scale_hat <- colMeans(params$log_x_sd_scale, dims=1)
# pdf("comp.pdf")
par(pty="s")
# plot(c(t(x_sigma_hat)) ~ c(t(x_sigma)), cex=2, pch=10, col="dark gray", xlab="Simulated X SD", ylab="Estimated X SD", asp=1)
# fit_B <- lm(c(t(x_sigma_hat)) ~ c(t(x_sigma))) # Fit a linear model
# abline(fit_B)
# abline(0,1, lty = 2) # 1:1 line
plot(c(t(beta_hat)) ~ c(t(B_init)), cex=2, pch=10, col="dark gray", xlab="Simulated Loadings", ylab="Estimated Loadings", asp=1)
fit_B <- lm(c(t(beta_hat)) ~ c(t(B_init))) # Fit a linear model
abline(fit_B)
abline(0,1, lty = 2) # 1:1 line
plot(c(t(chol_F_sigma_hat)) ~ c(t(chol_F_sigma)), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Cholesky", ylab="Estimated Log Factor Cholesky", asp=1)
fit_chol <- lm(c(t(chol_F_sigma_hat)) ~ c(t(chol_F_sigma))) # Fit a linear model
abline(fit_chol)
abline(0,1, lty = 2) # 1:1 line
# plot(c(log_F_sigma_loc_hat) ~ c(log_F_sigma_loc), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Location", ylab="Estimated Log Factor Location", asp=1)
# fit_loc <- lm(c(log_F_sigma_loc_hat) ~ c(log_F_sigma_loc)) # Fit a linear model
# abline(fit_loc)
# abline(0,1, lty = 2) # 1:1 line
plot(c(log_F_sigma_scale_hat) ~ c(log_F_sigma_scale), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Scale", ylab="Estimated Log Factor Scale", asp=1)
fit_scale <- lm(c(log_F_sigma_scale_hat) ~ c(log_F_sigma_scale)) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line
# plot(c(t(chol_X_sigma_hat)) ~ c(t(chol_X_sigma)), cex=2, pch=10, col="dark gray", xlab="Simulated Log X Cholesky", ylab="Estimated Log X Cholesky")
# fit_chol <- lm(c(t(chol_X_sigma_hat)) ~ c(t(chol_X_sigma))) # Fit a linear model
# abline(fit_chol)
# abline(0,1, lty = 2) # 1:1 line
# plot(c(log_X_sigma_loc_hat) ~ c(log_X_sigma_loc), cex=2, pch=10, col="dark gray", xlab="Simulated Log X Location", ylab="Estimated Log X Location")
# fit_loc <- lm(c(log_X_sigma_loc_hat) ~ c(log_X_sigma_loc)) # Fit a linear model
# abline(fit_loc)
# abline(0,1, lty = 2) # 1:1 line
# plot(c(log_X_sigma_scale_hat) ~ c(log_X_sigma_scale), cex=2, pch=10, col="dark gray", xlab="Simulated Log X Scale", ylab="Estimated Log X Scale")
# fit_scale <- lm(c(log_X_sigma_scale_hat) ~ c(log_X_sigma_scale)) # Fit a linear model
# abline(fit_scale)
# abline(0,1, lty = 2) # 1:1 line
# dev.off()

print(fit_b$draws() %>%
        subset_draws(variable = c('lp__', 'x_sd', 'chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'), regex = TRUE) %>%
        summarise_draws())


# # Save draws
# saveRDS(fit, "draws.rds")
# py_save_object(r_to_py(params), 'draws.pkl')

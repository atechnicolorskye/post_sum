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
n_T <- 100
n_F <- 3
n_X <- 10
n_D <- 1000
n_W <- 1000
R <- 2
AD <- 0.95
TD <- 11
# SS <- 0.01
discount <- 1

# Create data strucutures
X <- array(0, c(n_T, n_F))
log_F_sigma <- array(0, c(n_T, n_F))

# f hyperparameters
log_F_sigma_loc <- rnorm(n_F, sd=0.5)
log_F_sigma_scale <- abs(rnorm(n_F, mean=0, sd=0.5))
# log_F_sigma_scale <- rbeta(n_F, 2, 8)
chol_F_sigma <- matrix(rnorm(n_F * n_F, sd=0.2), c(n_F, n_F))
chol_F_sigma[upper.tri(chol_F_sigma)] <- 0
diag(chol_F_sigma) <- abs(diag(chol_F_sigma))

# Initialise
log_F_sigma[1, ] <- log_F_sigma_loc + chol_F_sigma %*% rnorm(n_F) # log_F_sigma_loc * rnorm(n_F)
# log_X_sigma[1, ] <- log_X_sigma_loc + chol_X_sigma %*% rnorm(n_X)
X[1, ] <- exp(log_F_sigma[1, ]) * rnorm(n_F)
# X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + exp(log_X_sigma[1, ]) * rnorm(n_X)

# Generate data
for (i in 2:n_T){
  # log_F_sigma[i, ] <- log_F_sigma_loc + (log_F_sigma_scale * log_F_sigma[i-1, ]) + chol_F_sigma %*% rnorm(n_F)
  log_F_sigma[i, ] <- log_F_sigma_loc + (log_F_sigma_scale * (log_F_sigma[i-1, ] - log_F_sigma_loc)) + chol_F_sigma %*% rnorm(n_F)
  X[i,  ] <- exp(log_F_sigma[i, ]) * rnorm(n_F)
}

# Standardise X
X_mu <- colMeans(X, dims=1)
X_sd <- apply(X, 2, sd)
# X_s <- scale(X)

matplot(X, type='l')
matplot(log_F_sigma, type='l')
# matplot(log_X_sigma, type='l')

base <- cmdstan_model('infer_tv_MSV.stan', quiet=FALSE, force_recompile=TRUE)
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

stanfit <- read_stan_csv(fit_b$output_files())
par_sum <- summary(stanfit, pars=c('chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'))$summary

params <- extract(stanfit)
chol_F_sigma_hat <- colMeans(params$chol_log_f_sd_sd, dims=1)
log_F_sigma_loc_hat <- colMeans(params$log_f_sd_loc, dims=1)
log_F_sigma_scale_hat <- colMeans(params$log_f_sd_scale, dims=1)
par(pty="s")
plot(c(t(chol_F_sigma_hat)) ~ c(t(chol_F_sigma)), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Cholesky", ylab="Estimated Log Factor Cholesky", asp=1)
fit_chol <- lm(c(t(chol_F_sigma_hat)) ~ c(t(chol_F_sigma))) # Fit a linear model
abline(fit_chol)
abline(0,1, lty = 2) # 1:1 line
plot(c(log_F_sigma_loc_hat) ~ c(log_F_sigma_loc), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Location", ylab="Estimated Log Factor Location", asp=1)
fit_loc <- lm(c(log_F_sigma_loc_hat) ~ c(log_F_sigma_loc)) # Fit a linear model
abline(fit_loc)
abline(0,1, lty = 2) # 1:1 line
plot(c(log_F_sigma_scale_hat) ~ c(log_F_sigma_scale), cex=2, pch=10, col="dark gray", xlab="Simulated Log Factor Scale", ylab="Estimated Log Factor Scale", asp=1)
fit_scale <- lm(c(log_F_sigma_scale_hat) ~ c(log_F_sigma_scale)) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line

print(fit_b$draws() %>%
        subset_draws(variable = c('lp__', 'x_sd', 'chol_log_f_sd_sd', 'log_f_sd_loc', 'log_f_sd_scale'), regex = TRUE) %>%
        summarise_draws())

print(log_F_sigma_scale)

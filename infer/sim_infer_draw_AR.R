library(posterior)
library(tidyverse)
options(tibble.print_max = Inf)

# library(reticulate)
# use_python("/usr/bin/python")
# np <- import("numpy")

library(cmdstanr)
library(rstan)

seed <- 1
set.seed(seed)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_T <- 100
n_F <- 5
n_X <- 10
n_D <- 1000
n_W <- 1000
R <- 1
AD <- 0.8
TD <- 10
# SS <- 0.1
discount <- 1

n_B <- sum(lower.tri(matrix(0, n_X, n_F)))

# Create data strucutures
# X <- array(0, c(n_T, n_X))
B <- array(0, c(n_T, n_B))
# B <- array(0, c(n_T, n_X, n_F))
# log_F_sigma <- array(0, c(n_T, n_F))

# # x hyperparameters 
# # x_sigma
# x_sigma <- exp(rep(rnorm(1), n_X))

# # f hyperparameters
# # Covariance
# log_F_sigma_loc <- rnorm(n_F, mean=0, sd=0.25)
# log_F_sigma_scale <- abs(rnorm(n_F, mean=0, sd=0.25))
# # chol_F_sigma <- diag(abs(rnorm(n_F, mean=0, sd=0.1)))
# chol_F_sigma <- matrix(rnorm(n_F * n_F, sd=0.1), c(n_F, n_F))
# chol_F_sigma[upper.tri(chol_F_sigma)] <- 0
# diag(chol_F_sigma) <- abs(diag(chol_F_sigma))

# B hyperparameters
# Want B to be 'almost sparse' where B contains mostly values close to 0.05
base_order <- 0.1
bias_order <- 0.5
p_connect <- 0.3
# B_loc <- matrix(0, n_X, n_F)
# diag(B_loc) <- 1
# n_l_t <- sum(lower.tri(B_loc))
# B_loc[lower.tri(B_loc)] <- rnorm(n_l_t, 0, .05) +
#                           rbinom(n_l_t, 1, .3) * (2 * rbinom(n_l_t, 1, .5) - 1) * 
#                           rnorm(n_l_t, bias_order, 1)
#                           (bias_order + abs(rnorm(n_l_t)))
B_loc <- rnorm(n_B, 0, .05) +
        rbinom(n_B, 1, .3) * (2 * rbinom(n_B, 1, .5) - 1) *
        rnorm(n_B, bias_order, 1)
B_scale <- abs(rnorm(n_B, mean=0, sd=0.25))
# B_sigma <- abs(rnorm(n_B, mean=0, sd=base_order))

# Initialise
# log_F_sigma[1, ] <- log_F_sigma_loc + chol_F_sigma %*% rnorm(n_F)
B[1, ] <- B_loc * rnorm(n_B)
# B[1, , ] <- B_loc + base_order * lower.tri(B[i, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
# X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
# X[1, ] <- (B[1, , ] %*% (exp(log_F_sigma[1, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)

# Generate data
for (i in 2:n_T){
  # log_F_sigma[i, ] <- log_F_sigma_loc + (log_F_sigma_scale * log_F_sigma[i-1, ]) + chol_F_sigma %*% rnorm(n_F)
  B[i, ] <- B_loc +  B_scale * (B[i-1, ] - B_loc) + base_order * rnorm(n_B) + 0.2 * rnorm(n_B)
  # B[i, , ] <- B_loc + (B_scale * B[i-1, , ]) + discount * base_order * lower.tri(B[i, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
  # X[i,  ] <- (B[i, , ] %*% (exp(log_F_sigma[i, ]) * rnorm(n_F))) + x_sigma * rnorm(n_X)
}

matplot(B, type='l')
# matplot(matrix(B, c(n_T, n_X * n_F)), type='l')

model <- cmdstan_model('infer_tv_AR.stan', quiet=FALSE, force_recompile=TRUE)
fit <- model$sample(data=list(P=n_B, F=n_F, TS=n_T, disc=discount, x=B), 
                    num_samples=n_D, 
                    num_warmup=n_W, 
                    refresh=100, 
                    init=R, 
                    seed=seed, 
                    adapt_delta=AD, 
                    max_depth=TD, 
                    # stepsize=SS,
                    num_chains=1,
                    num_cores=4)

stanfit <- read_stan_csv(fit$output_files())

# print(x_sigma)
# print(chol_F_sigma) 
# print(log_F_sigma_loc) 
# print(log_F_sigma_scale)

params <- extract(stanfit)
beta_init_hat <- colMeans(params$beta_init, dims=1)
beta_loc_hat <- colMeans(params$beta_loc, dims=1)
beta_scale_hat <- colMeans(params$beta_scale, dims=1)
# pdf("comp.pdf")
plot(c(beta_init_hat) ~ c(B[1, ]), cex=2, pch=10, col="dark gray", xlab="Simulated Beta Init", ylab="Estimated Beta Init")
fit_loc <- lm(c(beta_init_hat) ~ c(B[1, ])) # Fit a linear model
abline(fit_loc)
abline(0,1, lty = 2) # 1:1 line
plot(c(beta_loc_hat) ~ c(B_loc), cex=2, pch=10, col="dark gray", xlab="Simulated Beta Loc", ylab="Estimated Beta Loc")
fit_loc <- lm(c(beta_loc_hat) ~ c(B_loc)) # Fit a linear model
abline(fit_loc)
abline(0,1, lty = 2) # 1:1 line
plot(c(beta_scale_hat) ~ c(B_scale), cex=2, pch=10, col="dark gray", xlab="Simulated Beta Scale", ylab="Estimated Beta Scale")
fit_scale <- lm(c(beta_scale_hat) ~ c(B_scale)) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line
# dev.off()

print(fit$draws() %>%
        subset_draws(variable = c('lp__', 'x_sd', 'beta_sd', 'beta_scale'), regex = TRUE) %>%
        summarise_draws())
# saveRDS(fit, "draws.rds")
# 
# # Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

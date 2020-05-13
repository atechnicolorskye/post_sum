library(posterior)
library(tidyverse)

# library(reticulate)
# use_python("/usr/bin/python")
# np <- import("numpy")

library(cmdstanr)
library(rstan)
# options(mc.cores = parallel::detectCores())

seed <- 1
set.seed(seed)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_T <- 100
n_F <- 2
n_X <- 5
n_D <- 1000
n_W <- 1000
R <- 0.5
AD <- 0.99
TD <- 10
# SS <- 0.1
discount <- 1

# Create data strucutures
X <- array(0, c(n_T, n_X))
B <- array(0, c(n_T, n_X, n_F))

# x hyperparameters 
x_sigma <- abs(rep(rnorm(1, sd=0.25), n_X))
# x_sigma <- abs(rnorm(n_X, sd=0.25))
# x_sigma <- rep(1, n_X)


# f hyperparameters
# F_sigma <- abs(rnorm(n_F, sd=0.5))

# B hyperparameters
# Want B to be 'almost sparse' where B contains mostly values close to 0.05
base_order <- 0.1
bias_order <- 0.5
p_connect <- 0.3
B_loc <- matrix(0, n_X, n_F)
diag(B_loc) <- 1
n_l_t <- sum(lower.tri(B_loc))
B_loc[lower.tri(B_loc)] <- rnorm(n_l_t, 0, .05) +
                           rbinom(n_l_t, 1, .3) * (2 * rbinom(n_l_t, 1, .5) - 1) * 
                           rnorm(n_l_t, bias_order, 1)
                           # (bias_order + abs(rnorm(n_l_t)))
B_scale <- matrix(0, n_X, n_F)
B_scale[lower.tri(B_scale)] <- abs(rnorm(n_l_t, mean=0, sd=0.5)) # rbeta(n_l_t, 2, 2) # rep(0.5, n_l_t)

# Initialise
B[1, , ] <- B_loc + base_order * lower.tri(B[1, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
# X[1, ] <- (B[1, , ] %*% (F_sigma * rnorm(n_F))) + x_sigma * rnorm(n_X)
X[1, ] <- (B[1, , ] %*% rnorm(n_F)) + x_sigma * rnorm(n_X)

# Generate data
for (i in 2:n_T){
  B[i, , ] <- B_loc + (B_scale * (B[i-1, , ] - B_loc)) + discount * base_order * lower.tri(B[i, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
  X[i,  ] <- (B[i, , ] %*% rnorm(n_F)) + x_sigma * rnorm(n_X)
}

# # Standardise X
# X_mu <- colMeans(X, dims = 1)
# X_sd <- apply(X, 2, sd)
# X_s <- scale(X)

matplot(X, type='l')
matplot(matrix(B, c(n_T, n_X * n_F)), type='l')

model <- cmdstan_model('infer_tv.stan', quiet=FALSE, force_recompile=TRUE)
fit <- model$sample(data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X), 
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
# beta_lower_sd_hat <- colMeans(params$raw_beta_lower_sd, dims=1)
beta_lower_loc_hat <- colMeans(params$beta_lower_loc, dims=1)
beta_lower_scale_hat <- colMeans(params$beta_lower_scale, dims=1)
# F_sigma_hat <- colMeans(params$f_sd, dims=1)
# pdf("comp.pdf")
plot(c(beta_lower_loc_hat) ~ c(B_loc[lower.tri(B_loc)]), cex=2, pch=10, col="dark gray", xlab="Simulated Beta Loc", ylab="Estimated Beta Loc")
fit_scale <- lm(c(beta_lower_loc_hat) ~ c(B_loc[lower.tri(B_loc)])) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line
plot(c(beta_lower_scale_hat) ~ c(B_scale[lower.tri(B_scale)]), cex=2, pch=10, col="dark gray", xlab="Simulated Beta Scale", ylab="Estimated Beta Scale")
fit_scale <- lm(c(beta_lower_scale_hat) ~ c(B_scale[lower.tri(B_scale)])) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line
# plot(c(F_sigma_hat) ~ c(F_sigma), cex=2, pch=10, col="dark gray", xlab="Simulated F SD", ylab="Estimated F_SD")
# fit_scale <- lm(c(F_sigma_hat) ~ c(F_sigma)) # Fit a linear model
# abline(fit_scale)
# abline(0,1, lty = 2) # 1:1 line
# dev.off()

print(fit$draws() %>%
        subset_draws(variable = c('lp__', 'x_sd', 'raw_beta_lower_sd', 'beta_lower_scale'), regex = TRUE) %>%
        summarise_draws())
# saveRDS(fit, "draws.rds")
# 
# # Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

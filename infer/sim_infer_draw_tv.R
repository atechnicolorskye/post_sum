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
n_D <- 100
n_W <- 100
n_C <- 4
R <- 0.5
AD <- 0.999
TD <- 15
# SS <- 0.1
discount <- 1

# Create data structures
X <- array(0, c(n_T, n_X))
B <- array(0, c(n_T, n_X, n_F))

# x hyperparameters 
# x_sigma <- abs(rep(rnorm(1, sd=0.25), n_X))
x_sigma <- abs(rnorm(n_X, sd=0.25))
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
B_loc[lower.tri(B_loc)] <- rnorm(n_l_t, 0, .025) +
                           rbinom(n_l_t, 1, .3) * (2 * rbinom(n_l_t, 1, .5) - 1) * 
                           rnorm(n_l_t, bias_order, 0.5)
                           # (bias_order + abs(rnorm(n_l_t)))
B_scale <- matrix(0, n_X, n_F)
B_scale[lower.tri(B_scale)] <- runif(n_l_t, 0, 1) # abs(rnorm(n_l_t, mean=0.5, sd=0.5)) # rbeta(n_l_t, 2, 2) # rep(0.5, n_l_t)
# B_sigma <- matrix(0, n_X, n_F)
# B_sigma[lower.tri(B_sigma)] <- abs(rnorm(n_l_t, mean=0, sd=0.2)) # rbeta(n_l_t, 2, 2) # rep(0.5, n_l_t)
# B_sigma[lower.tri(B_sigma)] <- rep(rnorm(1, mean=0, sd=0.2), n_l_t) # rbeta(n_l_t, 2, 2) # rep(0.5, n_l_t)
B_sigma <- 0.2

# Initialise
# B[1, , ] <- B_loc + base_order * lower.tri(B[1, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
# B_loc <- matrix(0, n_X, n_F)
# diag(B_loc) <- 1
B[1, , ] <- B_loc + B_sigma * lower.tri(B[1, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
X[1, ] <- (B[1, , ] %*% rnorm(n_F)) + x_sigma * rnorm(n_X)

# Generate data
for (i in 2:n_T){
  B[i, , ] <- B_loc + (B_scale * (B[i-1, , ] - B_loc)) + discount * B_sigma * lower.tri(B[i, , ]) * matrix(rnorm(n_X * n_F), c(n_X, n_F))
  # B[i, , ] <- B_loc + (B_scale * (B[i-1, , ] - B_loc)) + discount * B_sigma * matrix(rnorm(n_X * n_F), c(n_X, n_F))
  X[i,  ] <- (B[i, , ] %*% rnorm(n_F)) + x_sigma * rnorm(n_X)
}

# # Standardise X
# X_mu <- colMeans(X, dims = 1)
# X_sd <- apply(X, 2, sd)
# X_s <- scale(X)

# matplot(X, type='l')
# matplot(matrix(B, c(n_T, n_X * n_F)), type='l')

model <- cmdstan_model('infer_tv.stan', quiet=FALSE, force_recompile=TRUE)
start <- proc.time()
fit <- model$sample(data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X), 
                    num_samples=n_D, 
                    num_warmup=n_W, 
                    refresh=100, 
                    init=R, 
                    seed=seed, 
                    adapt_delta=AD, 
                    max_depth=TD, 
                    # stepsize=SS,
                    num_chains=n_C,
                    num_cores=getOption("mc.cores"))

total <- proc.time() - start
sprintf("Time taken: %f s", total[3])

stanfit <- read_stan_csv(fit$output_files())
varstring <- paste(n_X, n_F, n_C, sep = "_")
saveRDS(stanfit, paste0('integrated_draws_', varstring, '.rds'))

# print(x_sigma)
# print(chol_F_sigma) 
# print(log_F_sigma_loc) 
# print(log_F_sigma_scale)

params <- extract(stanfit)
x_sd_hat <- colMeans(params$x_sd, dims=1)
# beta_lower_sd_hat <- colMeans(params$beta_lower_sd, dims=1)
beta_lower_sd_hat <- mean(params$beta_lower_sd)
beta_lower_loc_hat <- colMeans(params$beta_lower_loc, dims=1)
beta_lower_scale_hat <- colMeans(params$beta_lower_scale, dims=1)
# F_sigma_hat <- colMeans(params$f_sd, dims=1)
pdf(paste0('integrated_comp_', varstring, '.pdf'))
par(pty="s")
plot(c(x_sd_hat) ~ c(x_sigma), cex=2, pch=10, col="dark gray", xlab="Simulated X SD", ylab="Estimated X SD")
fit_scale <- lm(c(x_sd_hat) ~ c(x_sigma)) # Fit a linear model
abline(fit_scale)
abline(0,1, lty = 2) # 1:1 line
# plot(c(beta_lower_sd_hat) ~ c(B_sigma[lower.tri(B_sigma)]), cex=2, pch=10, col="dark gray", xlab="Simulated Beta SD", ylab="Estimated Beta SD")
# fit_scale <- lm(c(beta_lower_sd_hat) ~ c(B_sigma[lower.tri(B_sigma)])) # Fit a linear model
# abline(fit_scale)
# abline(0,1, lty = 2) # 1:1 line
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
dev.off()

# print(fit$draws() %>%
#         subset_draws(variable = c('lp__', 'x_sd', 'beta_lower_scale'), regex = TRUE) %>%
#         summarise_draws())

# # Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

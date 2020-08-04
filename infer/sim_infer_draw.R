library(Matrix)
library(mvtnorm)

library(reticulate)
# use_python("/usr/bin/python")
# use_python("/apps/python/3.6/3.6.3/bin/python3")
np <- import("numpy")

library(rstan)
options(mc.cores = parallel::detectCores())
# rstan_options(auto_write = TRUE)

seed <- 1
set.seed(seed)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_N <- 500
n_F <- 5
n_X <- 10
n_D <- 2000
n_W <- 1000
AD <- 0.99
TD <- 15

# Create data strucutures
X <- matrix(0, n_N, n_X)
B <- matrix(0, n_X, n_F)
log_F_sigma <- array(0, c(n_F))

# x hyperparameters 
# x_sigma
log_x_sigma <- rep(rnorm(1), n_X)

# f hyperparameters
# Covariance
log_f_sigma <- rnorm(n_F, mean=-1, sd=0.25)
# chol_f_sigma <- diag(abs(rnorm(n_F, mean=0, sd=0.1)))
chol_f_sigma <- matrix(rnorm(n_F * n_F, sd=0.1), c(n_F, n_F))
# diag(chol_f_sigma) <- abs(diag(chol_f_sigma))

# B hyperparameters
# Want B to be 'almost sparse' where B contains mostly values close to 0.05
base_order <- 0.05
bias_order <- 0.5
p_connect <- 0.3
B_ <- matrix(0, n_X, n_F)
diag(B_) <- 1
n_l_t <- sum(lower.tri(B_))
B_[lower.tri(B_)] <- rnorm(n_l_t, 0, .05) +
                      rbinom(n_l_t, 1, .3) * (2 * rbinom(n_l_t, 1, .5) - 1) * 
                      # rnorm(n_l_t, bias_order, 1)
                      (bias_order + abs(rnorm(n_l_t, 0, 1)))

# n_connect <- p_connect * n_X * n_F
# add <- (2 * rbinom(n_connect, 1, 0.5) - 1) * (bias_order + abs(rnorm(n_connect)))
# B_ <- base_order * rnorm(n_X * n_F)
# B_[1:n_connect] <- B_[1:n_connect] + add
# B_ <- matrix(sample(B_), c(n_X, n_F))
# # To ensure identifiability, set upper triangle to be 0 and diagonal to be 1
# B_ [upper.tri(B_)] <- 0
# diag(B_) <- 1

# Initialise
log_F_sigma <- as.vector(log_f_sigma + chol_f_sigma %*% rnorm(n_F))
B <- B_ + base_order * lower.tri(matrix(rnorm(n_X * n_F), c(n_X, n_F)))
cov_X <- B %*% diag(exp(2 * log_F_sigma)) %*% t(B) + diag(exp(2 * log_x_sigma))

# Generate data
for (i in 1:n_N){
  X[i, ] <- rmvnorm(1, sigma=cov_X)
}

# Standardise X
# X_mu <- colMeans(X, dims = 1)
# X_sd <- apply(X, 2, sd)
# X_s <- t((t(X) - X_mu) / X_sd)

fit <- stan(file='infer.stan', data=list(P=n_X, F=n_F, N=n_N, x=X), iter=n_D, warmup=n_W,
            refresh=n_D, seed=seed, control=list(adapt_delta=AD, max_treedepth=TD))

params <- extract(fit)
beta_hat <- c(t(colMeans(params$beta, dims=1)))
pdf("beta_comp.pdf")
plot(c(t(B_)) ~ c(t(B)), cex=2, pch=10, col="dark gray", xlab="True Loadings", ylab="Noisy Loadings")
plot(beta_hat ~ c(t(B)), cex=2, pch=10, col="dark gray", xlab="Noisy Simulated Loadings", ylab="Estimated Loadings")
fit_B <- lm(beta_hat ~ c(t(B))) # Fit a linear model
abline(fit_B)
abline(0,1, lty = 2) # 1:1 line
dev.off()

# Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

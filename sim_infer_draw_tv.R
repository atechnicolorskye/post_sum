library(Matrix)
library(mvtnorm)
library(rstan)
# options(mc.cores = parallel::detectCores())

set.seed(1)

# Simulate data according to Lopes and Carvalho (2007) without switching
n_T <- 100
n_F <- 5
n_Y <- 10
n_D <- 1000

# Create data strucutures
Y <- array(0, c(n_T, n_Y))
B <- array(0, c(n_T, n_Y, n_F))
log_F_sigma <- array(0, c(n_T, n_F))

# y hyperparameters 
# y_sigma
y_sigma <- matrix(rep(1. / rgamma(1, 3, 1), n_Y), c(n_Y, 1))

# x hyperparameters
# Covariance
log_f_sigma <- rnorm(n_F, mean=-0.75, sd=0.5)
chol_f_sigma <- matrix(rnorm(n_F * n_F, sd=0.25), c(n_F, n_F))
diag(chol_f_sigma) <- abs(diag(chol_f_sigma))
chol_f_sigma[upper.tri(chol_f_sigma)] <- 0
chol_f_sigma[lower.tri(chol_f_sigma)] <- 0

# B hyperparameters
# Want B to be 'almost sparse' where B contains mostly values close to 0.05
discount <- 0.95
base <- 0.05
p_connect <- 0.3
n_connect <- p_connect * n_Y * n_F
add <- (2 * rbinom(n_connect, 1, 0.5) - 1) * (0.5 + abs(rnorm(n_connect)))
B_ <- base * rnorm(n_Y * n_F)
B_[1:n_connect] <- B_[1:n_connect] + add
B_ <- matrix(sample(B_), c(n_Y, n_F))
# To ensure identifiability, set upper triangle to be 0 and diagonal to be 1
B_ [upper.tri(B_)] <- 0
diag(B_) <- 1

# Initialise
log_F_sigma[1, ] <- log_f_sigma + chol_f_sigma %*% rnorm(n_F)
B[1, , ] <- B_ + discount * base * lower.tri(matrix(rnorm(n_Y * n_F), c(n_Y, n_F)))
Y[1,  ] <- B[1, , ] %*% t(rmvnorm(1, sigma=diag(exp(log_F_sigma[1, ])))) + y_sigma * rnorm(n_Y)

# Generate data
for (i in 2:n_T){
  log_F_sigma[i, ] <- log_F_sigma[i-1, ] + chol_f_sigma %*% rnorm(n_F)
  B[i, , ] <- B[i-1, , ] + discount * base * lower.tri(B[i, , ]) * matrix(rnorm(n_Y * n_F), c(n_Y, n_F))
  Y[i, ] <- B[i, , ] %*% t(rmvnorm(1, sigma=diag(exp(log_F_sigma[i, ])))) + y_sigma * rnorm(n_Y)
}

fit <- stan(file='infer.stan', data=list(P=n_Y, D=n_F, TS=n_T, y=Y), iter=5000, warmup=1000, refresh=2000, seed=1, control=list(adapt_delta=0.95))
# fit <- stan(file='infer.stan', data=list(P=n_Y, D=n_F, TS=n_T, y=Y), seed=1, control=list(adapt_delta=0.8))

# # David's code
# MultiGeneralDLM = function(Y, X, m0, C0, n0, S0, deltaB, deltaE) 
# {
#   if (dim(Y)[1] == dim(X)[1]) {
#     numT = dim(Y)[1]
#   } else {
#     stop("check X and Y dimensions")
#   }
#   numY <- dim(Y)[2]
#   numX <- dim(X)[2]
#   if (dim(C0)[1] != numX | dim(C0)[2] != numX) {
#     stop("Check X,C0 dimension compatibility")
#   }
#   if (dim(as.matrix(S0))[1] != numY | dim(S0)[2] != numY) {
#     stop("Check Y,S0 dimension compatibility")
#   }
#   # Data Structures
#   S <- array(0, c(numT, numY, numY))
#   m <- array(0, c(numT, numX, numY))
#   R <- array(0, c(numT, numX, numX))
#   W <- array(0, c(numT, numX, numX))
#   C <- array(0, c(numT, numX, numX))
#   f <- array(0, c(numT, numY))
#   e <- array(0, c(numT, numY))
#   Q <- numeric(numT)
#   n <- numeric(numT)
#   V <- numeric(numT) + 1  # Vt 
#   At <- numeric(numX)
#   
#   # Priors at time 0 (t=1)
#   m[1,,] <- m0
#   C[1,,] <- C0
#   S[1,,] <- S0
#   n[1] <- n0
#   
#   # Recursion
#   for (t in 2:(numT)) {
#     # Prior at t W[t,,] = C[t-1,,]*(delta^-1 - 1) #Need we keep track of W? R[t,,] = C[t-1,,] + W[t,,] note a[t,,] = G%*%m[t-1,,], but G is identity matrix
#     R[t,,] = C[t - 1,,]/deltaB
#     W[t,,] = C[t-1,,]*(deltaB^-1 - 1)
#     # Forecast for t using t-1 data
#     f[t, ] = t(X[t - 1,]) %*% m[t - 1,,]
#     Q[t] = V[t] + t(X[t - 1,]) %*% R[t,,] %*% X[t - 1,]
#     # Posterior at t
#     At = R[t,,] %*% X[t - 1,]/Q[t]
#     e[t,] = as.numeric(Y[t - 1,] - f[t,])
#     m[t,,] = m[t - 1,,] + At %*% t(e[t,])
#     C[t,,] = R[t,,] - At %*% t(At) * Q[t]
#     n[t] = deltaE * n[t - 1] + 1
#     S[t,,] = n[t]^-1 * (deltaE * n[t - 1] * S[t - 1,,] + e[t, ] %*% t(e[t,])/Q[t])
#     
#   }
#   return(list(m = m, C = C, n=n, S = S, R = R, W=W, muHat = f, Q = Q, error = e))
# }

# # Set hyperparameters for Normal-Inverse Wishart
# m0 <- 0
# c0 <- diag(numX)
# S0 <- diag(numY)
# n0 <- 1
# deltaB <- 0.99
# deltaE <- 0.99
# 
# res <- MultiGeneralDLM(y, X, m0, c0, n0, S0, deltaB, deltaE)
# 
# # Create data strucutures for posterior draws 
# D <- array(0, c(numT, numD, numX))
# 
# # Draw from posterior
# for (i in 1:numT){
#   D[i, , ] <- mvrnorm(numD, res$m[i, , ], res$C[i, , ])
# }
# 
# filename <- paste('sim_x_', numX, '_y_', numY,  '_T_', numT, '_D_', numD, '.RData', sep="")
# to_save <- list(y=y, X=X, b=b, pos_m=res$m, pos_S=res$C, pos_draws=D)
# save(to_save, file=filename, compress='xz')
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

data <- read_csv('data/17_Industry_Portfolios.CSV', skip = 11, n_max=1100)
data <- data[, -1]

n_T <- 100
n_F <- 5
n_X <- dim(data)[2]
n_D <- 1000
n_W <- 1000
R <- 0.5
AD <- 0.999
TD <- 15
# SS <- 0.1
discount <- 1

X <- data[1001:1100, ]

# matplot(X, type='l')

model <- cmdstan_model('infer_tv.stan', quiet=FALSE, force_recompile=TRUE)
start <- proc.time()
fit <- model$sample(data=list(P=n_X, F=n_F, TS=n_T, disc=discount, x=X), 
                    num_samples=n_D, 
                    num_warmup=n_W, 
                    refresh=500, 
                    init=R, 
                    seed=seed, 
                    adapt_delta=AD, 
                    max_depth=TD, 
                    # stepsize=SS,
                    num_chains=4,
                    num_cores=getOption("mc.cores"))

total <- proc.time() - start
sprintf("Time taken: %f s", total[3])

stanfit <- read_stan_csv(fit$output_files())
saveRDS(stanfit, "industry_portfolio_17_5_draws.rds")

# print(fit$draws() %>%
#         subset_draws(variable = c('lp__', 'x_sd', 'beta_lower_scale'), regex = TRUE) %>%
#         summarise_draws())

# # Save draws as Python pickle
# py_save_object(r_to_py(params), 'draws.pkl')

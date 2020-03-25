library(factorstochvol)
library(reticulate)
np <- import("numpy", convert = TRUE)

set.seed(1)
                                                                                                                                                                                        
# simulate data
numT <- 100
p <- 10
numF <- 3

# use provided simulator
sim = fsvsim(n=numT, series=p, factors=numF, facload = "dense")

burnin <- c(10000)
draws <- c(10000)

fit = fsvsample(sim$y, factors=numF, burnin=burnin, draws=draws, keeptime='all')
covar <- covmat(fit)

fit_name <- paste('sim_p_', p, '_f_', numF,  '_T_', numT, '_b_', burnin, '_d_', draws, '.rData', sep="")
np_name <- paste('draws_sim_p_', p, '_f_', numF,  '_T_', numT, '_b_', burnin, '_d_', draws, sep="")
save(fit, file=fit_name)
np$savez_compressed(np_name, r_to_py(covar))

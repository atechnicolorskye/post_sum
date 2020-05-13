library(Matrix)

MultiGeneralDLM = function(Y, X, m0, C0, n0, S0, deltaB, deltaE) 
{
  if (dim(Y)[1] == dim(X)[1]) {
    numT = dim(Y)[1]
  } else {
    stop("check X and Y dimensions")
  }
  numY <- dim(Y)[2]
  numX <- dim(X)[2]
  if (dim(C0)[1] != numX | dim(C0)[2] != numX) {
    stop("Check X,C0 dimension compatibility")
  }
  if (dim(as.matrix(S0))[1] != numY | dim(S0)[2] != numY) {
    stop("Check Y,S0 dimension compatibility")
  }
  # Data Structures
  S <- array(0, c(numT, numY, numY))
  m <- array(0, c(numT, numX, numY))
  R <- array(0, c(numT, numX, numX))
  W <- array(0, c(numT, numX, numX))
  C <- array(0, c(numT, numX, numX))
  f <- array(0, c(numT, numY))
  e <- array(0, c(numT, numY))
  Q <- numeric(numT)
  n <- numeric(numT)
  V <- numeric(numT) + 1  # Vt 
  At <- numeric(numX)
  
  # Priors at time 0 (t=1)
  m[1,,] <- m0
  C[1,,] <- C0
  S[1,,] <- S0
  n[1] <- n0
  
  # Recursion
  for (t in 2:(numT)) {
    # Prior at t W[t,,] = C[t-1,,]*(delta^-1 - 1) #Need we keep track of W? R[t,,] = C[t-1,,] + W[t,,] note a[t,,] = G%*%m[t-1,,], but G is identity matrix
    R[t,,] = C[t - 1,,]/deltaB
    W[t,,] = C[t-1,,]*(deltaB^-1 - 1)
    # Forecast for t using t-1 data
    f[t, ] = t(X[t - 1,]) %*% m[t - 1,,]
    Q[t] = V[t] + t(X[t - 1,]) %*% R[t,,] %*% X[t - 1,]
    # Posterior at t
    At = R[t,,] %*% X[t - 1,]/Q[t]
    e[t,] = as.numeric(Y[t - 1,] - f[t,])
    m[t,,] = m[t - 1,,] + At %*% t(e[t,])
    C[t,,] = R[t,,] - At %*% t(At) * Q[t]
    n[t] = deltaE * n[t - 1] + 1
    S[t,,] = n[t]^-1 * (deltaE * n[t - 1] * S[t - 1,,] + e[t, ] %*% t(e[t,])/Q[t])
    
  }
  return(list(m = m, C = C, n=n, S = S, R = R, W=W, muHat = f, Q = Q, error = e))
}



ConstructMUtSIGt = function(m,C,n,d,mFF,CFF,nFF,SFF,numsim)
{
  library(mvtnorm)
  numF = dim(m)[1]
  p = dim(m)[2]
  # finally, the mean and variance of X
  MU = array(0,c(p,numsim))
  SIG = array(0,c(p,p,numsim))
  BETA = array(0,c(p,numF,numsim))
  SIGMA = array(0,c(p,numsim))
  
  # drawing factor moments
  MUFF = array(0,c(numF,numsim))
  SIGFF = array(0,c(numF,numF,numsim))
  for(nn in 1:numsim)
  {
    thelist = rmniw(mFF,as.matrix(CFF),SFF,nFF)
    MUFF[,nn] = thelist$mu
    SIGFF[,,nn] = thelist$SIG
  }
  for(pp in 1:p)
  {
    BETA[pp,,] = t(rmvt(numsim,df=n[pp],sigma=C[,,pp],type='shifted',delta=m[,pp]))
    SIGMA[pp,] = 1/rgamma(numsim,shape=n[pp]/2,rate=d[pp]/2)
    MU[pp,] = matprod(t(BETA[pp,,]),t(MUFF))
  }
  for(nn in 1:numsim)
  {
    SIG[,,nn] = BETA[,,nn] %*% SIGFF[,,nn] %*% t(BETA[,,nn]) + diag(SIGMA[,nn]) 
  }
  MUmean = apply(MU,c(1),mean)
  SIGmean = apply(SIG,c(1,2),mean)
  return(list(mu=MUmean,SIG=SIGmean,mudraw=t(MU),SIGdraw=SIG))
}


# library(factorstochvol)
# library(reticulate)
# np <- import("numpy", convert = TRUE)
# 
# set.seed(1)
#                                                                                                                                                                                         
# # simulate data
# numT <- 100
# p <- 10
# numF <- 3
# 
# # use provided simulator
# sim = fsvsim(n=numT, series=p, factors=numF, facload = "dense")
# 
# burnin <- c(10000)
# draws <- c(10000)
# 
# fit = fsvsample(sim$y, factors=numF, burnin=burnin, draws=draws, keeptime='all')
# covar <- covmat(fit)
# 
# fit_name <- paste('sim_p_', p, '_f_', numF,  '_T_', numT, '_b_', burnin, '_d_', draws, '.rData', sep="")
# np_name <- paste('draws_sim_p_', p, '_f_', numF,  '_T_', numT, '_b_', burnin, '_d_', draws, sep="")
# save(fit, file=fit_name)
# np$savez_compressed(np_name, r_to_py(covar))

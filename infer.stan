data {
  int <lower=1> P; // number of dimensions 
  int <lower=1> N; // number of    
  int <lower=1> D; // number of latent factors
  vector[P] y[N];
}
transformed data {
  // vector[D] fac_mu = rep_vector(0.0, D);
  // vector[D] log_f_sd = rep_vector(0.0, D);
  vector[D] L_diag = rep_vector(1.0, D);
  int<lower=1> beta_l; // number of lower-triangular, non-zero loadings
  beta_l = P * (P - D) + D * (D - 1) / 2;
}
parameters {
  // Parameters
  vector[D] z_log_f_sd; // reparameterised log latent factor standard deviations
  vector[D] z_fac[N]; // reparameterised latent factors
  vector[beta_l] z_log_L_lower_sd; // reparameterised lower diagional loading standard deviation
  // vector[D] log_f_sd; // log latent factor standard deviations
  vector[D] log_f_sd_raw; // raw latent factor standard deviation transform
  vector[D] fac_raw;
  // vector[beta_l] L_lower; // lower diagonal loadings
  vector[beta_l] L_lower_raw; // raw lower diagonal loadings
  // vector<lower=0>[D] L_diag; // positive diagonal loadings
  real log_L_lower_sd; // lower diagonal standard deviation
  real log_y_sd; // y standard deviation
  // Hyperparameters
  vector<lower=0>[D] tau; // scale
  cholesky_factor_corr[D] corr_log_f_sd; //prior correlation
}
transformed parameters {
  matrix[P, D] beta;
  vector[D] log_f_sd;
  vector[beta_l] L_lower;
  vector[D] fac[N]; // latent factors
  
  log_f_sd = log_f_sd_raw + diag_pre_multiply(tau, corr_log_f_sd) * z_log_f_sd;
  L_lower = L_lower_raw + exp(log_L_lower_sd) * z_log_L_lower_sd;

  // beta[T] and Q[T]
  {
  int idx;
  idx = 1;
  for (j in 1:D) {
    beta[j, j] = L_diag[j]; // set positive diagonal loadings
    for (k in (j+1):D){
      beta[j, k] = 0; // set upper triangle values to 0
    }
    for (i in (j+1):P){
      beta[i, j] = L_lower[idx]; // set lower diagonal loadings
      idx = idx + 1;
      }
    }
  }
  for (i in 1:N){
    fac[i] = fac_raw + exp(log_f_sd) .* z_fac[i];
  }
}
model {
  // priors
  log_f_sd_raw ~ normal(0, 1);
  tau ~ normal(0, 1);
  corr_log_f_sd ~ lkj_corr_cholesky(1);
  fac_raw ~ normal(0, 1);
  L_lower_raw ~ normal(0, 1);
  // L_lower_sd ~ gamma(2, 0.1);
  log_L_lower_sd ~ normal(0, 1);
  // L_diag ~ gamma(2, 0.5);
  log_y_sd ~ normal(0, 1);
  
  z_log_f_sd ~ std_normal();
  z_log_L_lower_sd ~ std_normal();
  
  for (i in 1:N){
    z_fac[i] ~ std_normal();
    y[i] ~ normal(beta * fac[i], exp(log_y_sd));
  }
}

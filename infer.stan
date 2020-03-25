data {
  int <lower=1> P; // number of dimensions 
  int <lower=1> TS; // number of time steps   
  int <lower=1> D; // number of latent factors
  vector[P] y[TS];
}
transformed data {
  int<lower=1> beta_l; // number of lower-triangular, non-zero loadings
  beta_l = P * (P - D) + D * (D - 1) / 2;
}
parameters {
  // Parameters
  vector[D] log_f_sd[TS]; // log latent factor standard deviations
  vector[D] log_f_sd_beta; // log latent factor standard deviation transform
  vector[D] log_f_sd_bias; // log latent factor standard deviation bias
  vector[beta_l] L_lower[TS]; // lower diagonal loadings
  vector[beta_l] L_lower_beta; // lower diagonal transform
  vector[beta_l] L_lower_bias; // lower diagonal bias
  vector<lower=0>[beta_l] L_lower_sd; // lower diagonal standard deviations
  vector<lower=0>[D] L_diag[TS]; // positive diagonal loadings
  vector[P] y_mu; // y means
  vector[P] log_y_sd; // y standard deviations
  // Hyperparameters
  vector<lower=0>[D] tau; // scale
  cholesky_factor_corr[D] sigma_log_f_sd; //prior correlation
}
transformed parameters {
  matrix[P, D] beta [TS];
  cov_matrix[P] Q[TS];
  vector[D] t_log_f_sd[TS];
  vector[beta_l] t_L_lower[TS];
  // beta[T] and Q[T]
  for (t in 1:TS){ 
    int idx;
    idx = 1;
    for (j in 1:D) {
      beta[t][j, j] = L_diag[t][j]; // set positive diagonal loadings
      for (k in (j+1):D){
        beta[t][j, k] = 0; // set upper triangle values to 0
      }
      for (i in (j+1):P){
        beta[t][i, j] = L_lower[t][idx]; // set lower diagonal loadings
        idx = idx + 1;
      }
    }
  Q[t] = tcrossprod(diag_post_multiply(beta[t], exp(log_f_sd[t]))) + diag_matrix(exp(2 * log_y_sd)); // specify covariance of y
  if (t==1) {
    t_log_f_sd[t] = log_f_sd[t];
    t_L_lower[t] = L_lower[t];
    }
  else {
    t_log_f_sd[t] = log_f_sd_bias + (log_f_sd_beta .* log_f_sd[t-1]);
    t_L_lower[t] = L_lower_bias + (L_lower_beta .* L_lower[t-1]);
    }
  }
}
model {
  // priors
  log_f_sd[1] ~ normal(0, 1);  
  L_lower[1] ~ normal(0, 1);
  L_diag[1] ~ gamma(2, 0.5);
  tau ~ normal(0, 1);
  sigma_log_f_sd ~ lkj_corr_cholesky(1);
  L_lower_sd ~ gamma(2, 0.2);
  log_y_sd ~ normal(0, 1);
  
  log_f_sd ~ multi_normal_cholesky(t_log_f_sd, diag_pre_multiply(tau, sigma_log_f_sd));
  L_lower ~ multi_normal_cholesky(t_L_lower, diag_matrix(L_lower_sd));
  
  for (t in 1:TS){
    // log_f_sd[t] ~ multi_normal_cholesky(log_f_sd[t-1], diag_pre_multiply(tau, sigma_f_sd));
    // L_lower[t] ~ normal(L_lower[t-1], L_lower_sd);
    y[t] ~ multi_normal(y_mu, Q[t]);
  }
}
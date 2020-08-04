data {
  int<lower=1> P; // # dimensions 
  int<lower=1> F; // # factors
  int<lower=1> N; // # datapoints, # posterior draws
  vector[P] x[N];
}
transformed data {
  // vector[F] fac_mu = rep_vector(0.0, F);
  // vector[F] log_f_sd = rep_vector(0.0, F);
  vector[F] L_diag = rep_vector(1.0, F);
  // vector[P] x_mu = rep_vector(0.0, P);
  int<lower=1> beta_l = F * (P - F) + F * (F - 1) / 2; // number of lower-triangular, non-zero loadings
}
parameters {
  // Parameters
  // vector[F] z_log_f_sd; // reparameterised log latent factor standard deviations
  vector[F] z_fac[N]; // reparameterised latent factors
  // vector[beta_l] z_log_L_lower_sd; // reparameterised lower diagional loading standard deviation
  
  vector[F] log_f_sd; // log latent factor standard deviations
  vector[beta_l] L_lower; // lower diagonal loadings
  // vector<lower=0>[F] L_diag; // positive diagonal loadings
  
  // vector[F] log_f_sd_raw; // raw latent factor standard deviation transform
  // vector[F] fac_raw;
  // vector[beta_l] L_lower_raw; // raw lower diagonal loadings
  // vector[P] mu_raw;
  
  // vector<lower=0>[F] tau; // scale
  // cholesky_factor_corr[F] corr_log_f_sd; //prior correlation
  // real log_L_lower_sd; // lower diagonal standard deviation
  real log_x_sd; // x standard deviation
}
transformed parameters {
  matrix[P, F] beta;
  
  vector[P] mu[N];
  
  vector[F] f_sd;
  f_sd = exp(log_f_sd);
  
  // vector[F] log_f_sd;
  // vector[beta_l] L_lower;
  
  // log_f_sd = log_f_sd_raw + diag_pre_multiply(tau, corr_log_f_sd) * z_log_f_sd;
  // L_lower = L_lower_raw + exp(log_L_lower_sd) * z_log_L_lower_sd;

  // cov_matrix[P] cov_X;
  {
  int idx;
  idx = 1;
  for (j in 1:F) {
    beta[j, j] = L_diag[j]; // set positive diagonal loadings
    for (k in (j+1):F){
      beta[j, k] = 0; // set upper triangle values to 0
    }
    for (i in (j+1):P){
      beta[i, j] = L_lower[idx]; // set lower diagonal loadings
      idx = idx + 1;
      }
    }
  }
  for (i in 1:N){
    mu[i] = beta * (f_sd .* z_fac[i]);
  }
  // cov_X = tcrossprod(diag_post_multiply(beta, exp(2 * log_f_sd))) + diag_matrix(exp(2 * rep_vector(log_x_sd, P)));
}
model {
  real x_sd;
  x_sd = exp(log_x_sd);
  // priors
  // log_f_sd_raw ~ normal(0, 1);
  // fac_raw ~ std_normal();
  // L_lower_raw ~ normal(0, 1);
  L_lower ~ std_normal();
  // mu_raw ~ std_normal();
  
  // tau ~ normal(0, 1);
  // corr_log_f_sd ~ lkj_corr_cholesky(1);
  log_f_sd ~ std_normal();
  // L_lower_sd ~ gamma(2, 0.1);
  // log_L_lower_sd ~ normal(0, 1);
  log_x_sd ~ std_normal();
  
  // L_diag ~ gamma(2, 0.3);
  
  // z_log_f_sd ~ std_normal();
  // z_log_L_lower_sd ~ std_normal();
  
  for (i in 1:N){
    z_fac[i] ~ std_normal();
    x[i] ~ normal(mu[i], x_sd);
    // x[i] ~ multi_normal(x_mu, cov_X);
  }
}
// generated quantities{
//   vector[F] zeros_F = rep_vector(0.0, F);
//   vector[P] zeros_P = rep_vector(0.0, P);
//   cholesky_factor_cov[F] f_sd = diag_matrix(exp(log_f_sd));
//   cholesky_factor_cov[P] x_sd = diag_matrix(rep_vector(exp(log_x_sd), P));
//   vector[P] gen_y[N];
//   // matrix[P, P] gen_cov[N];
//   for (i in 1:N){
//     gen_y[i] = beta * multi_normal_cholesky_rng(zeros_F, f_sd) + multi_normal_cholesky_rng(zeros_P, x_sd);
//     // gen_cov[i] = gen_y[i] * gen_y[i]';
//   }
// }

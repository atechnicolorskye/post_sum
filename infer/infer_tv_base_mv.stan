data {
  int<lower=1> P; // number of dimensions 
  int<lower=1> F; // number of latent factors
  int<lower=1> N; // number of draws per time step
  int<lower=1> TS; // number of time steps
  real disc; // discount
  matrix[N, P] x[TS];
}
transformed data {
  vector[F] L_diag = rep_vector(1.0, F);
  int<lower=1> beta_l = F * (P - F) + F * (F - 1) / 2; // number of lower-triangular, non-zero loadings
  vector[N * P] vec_x[TS];
  for (t in 1:TS){
    vec_x[t] = to_vector(x[t]);
  }
}
parameters {
  // reparameterisation parameters
  matrix[N, F] z_log_f_sd[TS]; // reparameterised log latent factor standard deviations
  matrix[N, F] z_fac[TS]; // reparameterised latent factors
  
  // parameters
  vector[beta_l] L_lower_init; // initial lower diagonal loadings
  
  vector[F] log_f_sd_loc; // latent factor standard deviation transform location
  vector<lower=0, upper=1>[F] log_f_sd_scale; // latent factor standard deviation transform scale
  // vector[P] x_loc; // observation location
  // vector[P] x_scale; //observation scale

  // priors
  cholesky_factor_corr[F] L_Omega;
  vector<lower=0>[F] tau;
  // vector<lower=0,upper=pi()/2>[F] tau_unif;
  // vector<lower=0>[F] chol_log_f_sd_sd; // diagonal standard deviation
  real<lower=0> x_sd; // x standard deviation
  // vector[F] log_f_sd_sd; // diagonal standard deviation
  // real log_x_sd; // x standard deviation
}
transformed parameters {
  matrix[N, F] log_f_sd[TS];
  matrix[N, F] rep_log_f_sd_loc;
  matrix[N, F] rep_log_f_sd_scale;
  matrix[F, P] beta_T;
  matrix[N, P] rep_x_loc;
  matrix[N, P] rep_x_scale;
  // matrix[N, P] mu[TS];
  // vector[F] tau = tan(tau_unif);
  cholesky_factor_cov[F] chol_log_f_sd_sd = diag_pre_multiply(tau, L_Omega);
  for (i in 1:N){
    rep_log_f_sd_loc[i] = log_f_sd_loc';
    rep_log_f_sd_scale[i] = log_f_sd_scale';
  }
  // for (i in 1:N){
  //   rep_x_loc[i] = x_loc';
  //   rep_x_scale[i] = x_scale';
  // }
  // vector[F] chol_log_f_sd_sd = exp(log_f_sd_sd);
  // real x_sd = exp(log_x_sd);
  
  {
    int idx;
    idx = 1;
    for (i in 1:F) {
      beta_T[i, i] = L_diag[i]; // set positive diagonal loadings
      for (k in (i+1):F){
        beta_T[k, i] = 0; // set upper triangle values to 0
      }
      for (j in (i+1):P){
        beta_T[i, j] = L_lower_init[idx]; // set lower diagonal loadings
        idx = idx + 1;
      }
    }
  }
  
  for (t in 1:TS){
    log_f_sd[t] = rep_log_f_sd_loc + z_log_f_sd[t] * chol_log_f_sd_sd;
    // log_f_sd[t] = log_f_sd_loc + chol_log_f_sd_sd * z_log_f_sd[t];
    if (t > 1){
      log_f_sd[t] += rep_log_f_sd_scale .* (log_f_sd[t-1] - rep_log_f_sd_loc);
    }
    // mu[t] =  rep_x_loc + rep_x_scale .* ((exp(log_f_sd[t]) .* z_fac[t]) * beta_T);
    // mu[t] = (exp(log_f_sd[t]) .* z_fac[t]) * beta_T;
  }
}
model {
  // priors
  L_lower_init ~ std_normal();
  
  log_f_sd_loc ~ std_normal();
  log_f_sd_scale ~ std_normal();
  // x_loc ~ std_normal();
  // x_scale ~ std_normal();
  
  L_Omega ~ lkj_corr_cholesky(1.);
  // chol_log_f_sd_sd ~ gamma(2, 1./10.);
  x_sd ~ gamma (2, 1./10.);
  // log_f_sd_sd ~ normal(-2, 0.1);
  // log_x_sd ~ normal(-2, 0.1);
  
  // likelihood 
  for (t in 1:TS){
    to_vector(z_log_f_sd[t]) ~ std_normal();
    to_vector(z_fac[t]) ~ std_normal();
    // vec_x[t] ~ normal(to_vector(rep_x_loc + rep_x_scale .* ((exp(log_f_sd[t]) .* z_fac[t]) * beta_T)), x_sd);
    vec_x[t] ~ normal(to_vector((exp(log_f_sd[t]) .* z_fac[t]) * beta_T), x_sd);
  }
}

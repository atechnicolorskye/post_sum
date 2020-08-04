data {
  int <lower=1> P; // number of dimensions 
  int <lower=1> F; // number of latent factors
  int <lower=1> TS; // number of time steps
  real disc; // discount
  vector[F] x[TS];
}
transformed data {
  // vector[F] beta_diag = rep_vector(1.0, F);
  // int beta_l = F * (P - F) + F * (F - 1) / 2; // number of lower-triangular, non-zero loadings
  matrix[F, F] ones;
  for (i in 1:F){
    for (j in 1:F){
      ones[i][j] = 1;
    }
  }
}  
parameters {
  // nuisance parameters
  vector[F] z_log_f_sd[TS]; // reparameterised log latent factor standard deviations
  // vector[F] z_fac[TS]; // reparameterised latent factors
  
  // parameters
  vector[F] log_f_sd_loc; // latent factor standard deviation location
  vector<lower=0.0, upper=1.0>[F] log_f_sd_scale; // latent factor standard deviation scale

  cholesky_factor_corr[F] log_f_sd_Omega;
  vector<lower=0.0>[F] log_f_sd_tau;
  
  // priors
  vector<lower=0.0>[F] m_log_f_sd_scale;
  vector<lower=0.0>[F] sd_log_f_sd_scale;
}
transformed parameters { 
  vector[F] log_f_sd[TS];
  matrix[F, F] chol_log_f_sd_sd = diag_pre_multiply(log_f_sd_tau, log_f_sd_Omega);
  // matrix[F, F] init_chol = cholesky_decompose(chol_log_f_sd_sd * chol_log_f_sd_sd' ./ (ones - log_f_sd_scale * log_f_sd_scale'));

  log_f_sd[1] = log_f_sd_loc + chol_log_f_sd_sd * z_log_f_sd[1];
  // log_f_sd[1] = log_f_sd_loc + init_chol * z_log_f_sd[1];
  for (t in 2:TS){
    log_f_sd[t] = log_f_sd_loc + (log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc)) + (chol_log_f_sd_sd * z_log_f_sd[t]);
  }
}
model {
  vector[F] sigma;
  // hyperpriors
  m_log_f_sd_scale ~ normal(0, 5);
  sd_log_f_sd_scale ~ lognormal(0, 1);
  
  // priors
  log_f_sd_loc ~ std_normal();
  // log_f_sd_scale ~ normal(0, p_log_f_sd_scale);
  for (f in 1:F){
    log_f_sd_scale[f] ~ normal(m_log_f_sd_scale[f], sd_log_f_sd_scale[f]) T[0, ];
  }
  // log_f_sd_scale ~ beta(2, 1./20.);
  // log_f_sd_scale ~ gamma(2, 1./8.);
  // log_f_sd_scale ~ lognormal(-2, 0.5);
  
  log_f_sd_Omega ~ lkj_corr_cholesky(1);
  log_f_sd_tau ~ cauchy(0, 0.5);
  
  // likelihood 
  for (t in 1:TS){
    z_log_f_sd[t] ~ std_normal();
    x[t] ~ normal(0, exp(log_f_sd[t]));
  }
}

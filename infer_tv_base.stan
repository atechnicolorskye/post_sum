data {
  int <lower=1> P; // number of dimensions 
  int <lower=1> F; // number of latent factors
  int <lower=1> TS; // number of time steps
  real disc; // discount
  vector[P] x[TS];
}
transformed data {
  vector[F] beta_diag = rep_vector(1.0, F);
  int beta_l = F * (P - F) + F * (F - 1) / 2; // number of lower-triangular, non-zero loadings
}
parameters {
  // nuisance parameters
  vector[F] z_log_f_sd[TS]; // reparameterised log latent factor standard deviations
  vector[F] z_fac[TS]; // reparameterised latent factors
   
  // parameters
  // vector[F] log_f_sd_loc; // latent factor standard deviation location
  vector<lower=0.0, upper=1.0>[F] log_f_sd_scale; // latent factor standard deviation scale
  
  // cholesky_factor_corr[F] log_f_sd_Omega;
  // vector<lower=0.0>[F] log_f_sd_tau;
  real<lower=0.0> log_f_sd_sd;

  vector[beta_l] beta_lower_init; // initial lower diagonal loadings
  // vector<lower=0.0>[F] beta_diag; // positive diagonal loadings
  
  // vector<lower=0.0>[P] x_sd; // x standard deviation
  real<lower=0.0> x_sd; // x standard deviation
  
}
transformed parameters { 
  vector[F] log_f_sd[TS];
  matrix[P, F] beta;
  
  // vector[F] abs_log_f_sd_scale;
  // vector[F] log_f_sd_scale = rep_vector(r_log_f_sd_scale, F);
  // matrix[F, F] chol_log_f_sd_sd = diag_pre_multiply(log_f_sd_tau, log_f_sd_Omega);
  matrix[F, F] chol_log_f_sd_sd = diag_matrix(rep_vector(log_f_sd_sd, F));
  
  // for (f in 1:F){
  //   abs_log_f_sd_scale[f] = abs(log_f_sd_scale[f]);
  // }
  
  {
    int idx;
    idx = 1;
    for (j in 1:F) {
      beta[j, j] = beta_diag[j]; // set positive diagonal loadings
      for (k in (j+1):F){
        beta[j, k] = 0; // set upper triangle values to 0
      }
      for (i in (j+1):P){
        beta[i, j] = beta_lower_init[idx]; // set lower diagonal loadings
        idx = idx + 1;
      }
    }
  }
  
  // log_f_sd[1] = log_f_sd_loc + chol_log_f_sd_sd * z_log_f_sd[1];
  // log_f_sd[1] = log_f_sd_init;
  // for (t in 2:TS){
  //   // log_f_sd[t] = log_f_sd_loc + (log_f_sd_scale .* log_f_sd[t-1]) + (chol_log_f_sd_sd * z_log_f_sd[t]);
  //   log_f_sd[t] = log_f_sd_loc + (log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc)) + (chol_log_f_sd_sd * z_log_f_sd[t]);
  //   // log_f_sd[t] = log_f_sd_loc + (abs_log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc)) + (chol_log_f_sd_sd * z_log_f_sd[t]);
  // }
  
  // log_f_sd[1] = log_f_sd_loc + chol_log_f_sd_sd * z_log_f_sd[1];
  log_f_sd[1] = chol_log_f_sd_sd * z_log_f_sd[1];
  for (t in 2:TS){
    // log_f_sd[t] = log_f_sd_loc + (log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc)) + (chol_log_f_sd_sd * z_log_f_sd[t]);
    log_f_sd[t] = (log_f_sd_scale .* log_f_sd[t-1]) + (chol_log_f_sd_sd * z_log_f_sd[t]);
  }
}
model {
  vector[P] x_mu;
  // matrix[F, F] cov_log_f_sd_init = chol_log_f_sd_sd * chol_log_f_sd_sd' ./ (ones - log_f_sd_scale * log_f_sd_scale');
  // priors
  beta_lower_init ~ std_normal();
  // beta_diag ~ normal(0, 5);
  
  // log_f_sd_loc ~ std_normal();
  // log_f_sd_scale ~ cauchy(0, 0.5);
  log_f_sd_scale ~ beta(2, 2);
  // for (f in 1:F){
  //   log_f_sd_scale[f] ~ normal(0, 0.5) T[0, ];
  // }
  // log_f_sd_scale ~ gamma(2, 1./20.);
  // log_f_sd_scale ~ lognormal(-1, 0.5);
  // log_f_sd_scale ~ gamma(2, 1./8.);
  log_f_sd_sd ~ gamma(2, 1./10.);
  
  x_sd ~ gamma(2, 1./10);
  
  // log_f_sd_Omega ~ lkj_corr_cholesky(1);
  // log_f_sd_tau ~ normal(0, 0.1);

  // likelihood 
  for (t in 1:TS){
    z_log_f_sd[t] ~ std_normal();
    z_fac[t] ~ std_normal();
    x[t] ~ normal(beta * (exp(log_f_sd[t]) .* z_fac[t]), x_sd);
    // x_mu = beta * (exp(log_f_sd[t]) .* z_fac[t]);
    // for (p in 1:P){
    //   x[t][p] ~ normal(x_mu[p], x_sd[p]);
    // }
  }
}

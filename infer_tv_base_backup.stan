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
  // vector[P] z_log_x_sd[TS]; // reparameterised nosie
  
  // parameters
  vector[F] log_f_sd_loc; // latent factor standard deviation location
  // vector<lower=0.0, upper=1.0>[F] log_f_sd_scale; // latent factor standard deviation scale
  vector<lower=-1.0, upper=1.0>[F] log_f_sd_scale; // latent factor standard deviation scale
  // real<lower=0.0, upper=1.0> r_log_f_sd_scale; // latent factor standard deviation scale
  
  vector[beta_l] beta_lower_init; // initial lower diagonal loadings
  // vector<lower=0.0>[F] beta_diag; // positive diagonal loadings
  
  // vector[P] log_x_sd_loc; 
  // vector<lower=0.0, upper=1.0>[P] log_x_sd_scale; 
  real<lower=0.0> x_sd; // x standard deviation
  
  // priors
  cholesky_factor_corr[F] log_f_sd_Omega;
  vector<lower=0.0>[F] log_f_sd_tau;
  // cholesky_factor_corr[P] log_x_sd_Omega;
  // vector<lower=0.0>[P] log_x_sd_tau;
  // vector<lower=0.0>[P] diag_log_x_sd;
}
transformed parameters { 
  vector[F] log_f_sd[TS];
  matrix[P, F] beta;
  
  // vector[P] log_x_sd[TS];
  
  // vector[F] log_f_sd_t;
  vector<lower=0.0, upper=1.0>[F] abs_log_f_sd_scale;
  // vector[F] log_f_sd_scale = rep_vector(r_log_f_sd_scale, F);
  matrix[F, F] chol_log_f_sd_sd = diag_pre_multiply(log_f_sd_tau, log_f_sd_Omega);
  // matrix[P, P] chol_log_x_sd = diag_pre_multiply(log_x_sd_tau, log_x_sd_Omega);
  // matrix[P, P] chol_log_x_sd = diag_matrix(diag_log_x_sd);
  
  for (f in 1:F){
    abs_log_f_sd_scale[f] = fabs(log_f_sd_scale[f]);
  }
  
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
  
  // print("loc:", log_f_sd_loc);
  // print("scale:", log_f_sd_scale);
  // print("chol:", chol_log_f_sd_sd)
  // for (i in 1:F){
    //   if (is_nan(log_f_sd_loc[i])){
      //     print("log_f_sd is nan");
      //     print("loc:", log_f_sd_loc);
      //     print("chol:", chol_log_f_sd_sd);
      //   }
    //   if (is_nan(z_log_f_sd[1][i])){
      //     print("z_log_f_sd[1] is nan");
      //   }
    //   for (j in 1:F){
      //     if (is_nan(chol_log_f_sd_sd[i][j])){
        //       print("chol_log_f_sd is nan");
        //       print("log:", log_f_sd_loc);
        //       print("chol:", chol_log_f_sd_sd);
        //     }
      //   }
    // }
  log_f_sd[1] = log_f_sd_loc + chol_log_f_sd_sd * z_log_f_sd[1];
  // log_x_sd[1] = log_x_sd_loc + chol_log_x_sd * z_log_x_sd[1];
  for (t in 2:TS){
    // log_f_sd[t] = log_f_sd_loc + (log_f_sd_scale .* log_f_sd[t-1]) + (chol_log_f_sd_sd * z_log_f_sd[t]);
    log_f_sd[t] = log_f_sd_loc + (abs_log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc)) + (chol_log_f_sd_sd * z_log_f_sd[t]);
    // log_x_sd[t] = log_x_sd_loc + (log_x_sd_scale .* log_x_sd[t-1]) + (chol_log_x_sd * z_log_x_sd[t]);
  }
}
model {
  vector[P] mu;
  // priors
  beta_lower_init ~ std_normal();
  // beta_diag ~ normal(0.5, 1);
  
  log_f_sd_loc ~ normal(0, 0.5);
  // r_log_f_sd_scale ~ gamma(2, 1./10.);
  // r_log_f_sd_scale ~ normal(0, 0.5);
  log_f_sd_scale ~ normal(0, 0.5);
  // log_f_sd_scale ~ lognormal(0, 0.5);
  
  // log_x_sd_loc ~ normal(0, 0.5);
  // log_x_sd_scale ~ normal(0, 0.5);
  x_sd ~ gamma(2, 1./10.);
  
  log_f_sd_Omega ~ lkj_corr_cholesky(1);
  log_f_sd_tau ~ normal(0, 0.2);
  // log_x_sd_Omega ~ lkj_corr_cholesky(10);
  // log_x_sd_tau ~ normal(0, 0.2);
  // diag_log_x_sd ~ gamma(2, 1./10.);
  
  // likelihood 
  for (t in 1:TS){
    z_log_f_sd[t] ~ std_normal();
    z_fac[t] ~ std_normal();
    // z_log_x_sd[t] ~ std_normal();
    // mu = beta * (exp(log_f_sd[t]) .* z_fac[t]);
    // if (is_nan(mu[1]) || is_nan(mu[2]) || is_nan(mu[3]) || is_nan(mu[4]) || is_nan(mu[5]) ||
           //     is_nan(mu[6]) || is_nan(mu[7]) || is_nan(mu[8]) || is_nan(mu[9]) || is_nan(mu[10])){
      //   print("chol_log_f_sd_sd", chol_log_f_sd_sd);
      //   print("log_f_sd_loc", log_f_sd_loc);
      //   print("log_f_sd_scale", log_f_sd_scale);
      //   print("z_log_f_sd", z_log_f_sd[t]);
      //   print("log_f_sd", log_f_sd[t]);
      //   print("f_sd", exp(log_f_sd[t]));
      // }
    x[t] ~ normal(beta * (exp(log_f_sd[t]) .* z_fac[t]), x_sd);
    // x[t] ~ multi_normal_cholesky(beta * (exp(log_f_sd[t]) .* z_fac[t]), diag_matrix(exp(log_x_sd[t])));
  }
}

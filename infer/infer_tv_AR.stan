data {
  int <lower=1> P; // number of dimensions 
  int <lower=1> F; // number of latent factors
  int <lower=1> TS; // number of time steps
  real disc; // discount
  vector[P] x[TS];
}
transformed data {
  // vector[F] beta_diag = rep_vector(1.0, F);
  // int beta_l = F * (P - F) + F * (F - 1) / 2; // number of lower-triangular, non-zero loadings
  int beta_l = P; // number of lower-triangular, non-zero loadings
}
parameters {
  // nuisance parameters
  // vector[F] z_log_f_sd[TS]; // reparameterised log latent factor standard deviations
  // vector[F] z_fac[TS]; // reparameterised latent factors
  vector[beta_l] z_beta[TS];
  
  // parameters
  // vector[F] log_f_sd_loc; // latent factor standard deviation location
  // vector[F] log_f_sd_scale; // latent factor standard deviation scale
  // vector<lower=0.0, upper=1.0>[F] log_f_sd_scale; // latent factor standard deviation scale
  // real<lower=0.0, upper=1.0> r_log_f_sd_scale; // latent factor standard deviation scale
  // vector[F] log_f_sd_init; // initial log_f_sd
  
  vector[beta_l] beta_init; // initial lower diagonal loadings
  vector[beta_l] beta_loc; // initial lower diagonal loadings
  vector<lower=0.0, upper=1.0>[beta_l] beta_scale; // positive diagonal loadings
  real<lower=0.0> beta_sd; 
  
  // vector[P] x_loc;
  real<lower=0.0> x_sd; // x standard deviation
  
  // priors
  // cholesky_factor_corr[F] log_f_sd_Omega;
  // vector<lower=0.0>[F] log_f_sd_tau;
}
transformed parameters { 
  // vector[F] log_f_sd[TS];
  // matrix[P, F] beta;
  vector[beta_l] beta[TS];
  
  // 
  // // vector[F] abs_log_f_sd_scale;
  // // vector[F] log_f_sd_scale = rep_vector(r_log_f_sd_scale, F);
  // matrix[F, F] chol_log_f_sd_sd = diag_pre_multiply(log_f_sd_tau, log_f_sd_Omega);
  // 
  // // for (f in 1:F){
  //   //   abs_log_f_sd_scale[f] = abs(log_f_sd_scale[f]);
  //   // }
  // 
  // {
  //   int idx;
  //   idx = 1;
  //   for (j in 1:F) {
  //     beta[j, j] = beta_diag[j]; // set positive diagonal loadings
  //     for (k in (j+1):F){
  //       beta[j, k] = 0; // set upper triangle values to 0
  //     }
  //     for (i in (j+1):P){
  //       beta[i, j] = beta_lower_init[idx]; // set lower diagonal loadings
  //       idx = idx + 1;
  //     }
  //   }
  // }
  // 
  // for (t in 1:TS){
  //   log_f_sd[t] = log_f_sd_loc + (chol_log_f_sd_sd * z_log_f_sd[t]);
  //   if (t > 1){
  //     log_f_sd[t] += log_f_sd_scale .* (log_f_sd[t-1] - log_f_sd_loc);
  //   }
  // }
  beta[1] = beta_loc + (beta_sd * z_beta[1]);
  for (t in 2:TS){
    beta[t] = beta_loc + (beta_scale .* (beta[t-1] - beta_loc)) + (beta_sd * z_beta[t]);
    }
}
model {
  // priors
  // beta_lower_init ~ std_normal();
  // beta_diag ~ normal(0.5, 1);
  
  // log_f_sd_loc ~ normal(0, 0.5);
  // log_f_sd_scale ~ normal(0, 0.5);
  // for (f in 1:F){
    //   log_f_sd_scale[f] ~ normal(0, 0.3) T[0, ];
    // }
  // log_f_sd_scale ~ gamma(2, 1./20.);
  // log_f_sd_scale ~ lognormal(-2, 0.5);
  // r_log_f_sd_scale ~ gamma(2, 1./10.);
  // beta_init ~ std_normal();
  beta_loc ~ std_normal();
  // beta_scale ~ uniform(0, 1);
  beta_scale ~ cauchy(0, 0.5); //increasing bias
  // beta_scale ~ gamma(2, 1./10.); // increasing bias
  // beta_scale ~ lognormal(-2, 0.5); // constant bias
  beta_sd ~ gamma(2, 1./10.);
  
  x_sd ~ gamma(2, 1./10.);
  
  // log_f_sd_Omega ~ lkj_corr_cholesky(1);
  // log_f_sd_tau ~ normal(0, 0.2);

  // likelihood 
  for (t in 1:TS){
    z_beta[t] ~ std_normal();
    x[t] ~ normal(beta[t], x_sd);
    // z_log_f_sd[t] ~ std_normal();
    // z_fac[t] ~ std_normal();
    // x[t] ~ normal(beta * (exp(log_f_sd[t]) .* z_fac[t]), x_sd);
    // x[t] ~ normal(x_loc + beta * (exp(log_f_sd[t]) .* z_fac[t]), x_sd);
  }
}

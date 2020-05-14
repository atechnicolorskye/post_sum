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
  vector[F] z_fac_sd[TS]; // reparameterised latent factors
  vector[beta_l] z_beta_lower_sd[TS]; // reparameterised lower diagonal loading standard deviation
  
  // parameters
  vector[beta_l] beta_lower_loc; // lower diagonal loadings location
  vector<lower=0.0, upper=1.0>[beta_l] beta_lower_scale; // lower diagonal loadings scale
  // real<lower=0.0, upper=1.0> raw_beta_lower_scale; // lower diagonal loadings scale
  // vector<lower=0>[F] beta_diag; // positive diagonal loadings
  
  // priors
  // real<lower=0> raw_beta_lower_sd;
  vector<lower=0>[beta_l] beta_lower_sd;
  // real<lower=0> x_sd; // x standard deviation
  vector<lower=0>[P] x_sd;
}
transformed parameters {
  vector[beta_l] beta_lower[TS];
  matrix[P, F] beta [TS];
  
  // vector[beta_l] beta_lower_scale = rep_vector(raw_beta_lower_scale, beta_l);
  // vector[beta_l] beta_lower_sd = rep_vector(raw_beta_lower_sd, beta_l);
  
  for (t in 1:TS){
    beta_lower[t] = beta_lower_loc + beta_lower_sd .* z_beta_lower_sd[t];
    // beta_lower[t] = beta_lower_loc + beta_lower_sd * z_beta_lower_sd[t];
    if (t > 1){
      beta_lower[t] += beta_lower_scale .* (beta_lower[t-1] - beta_lower_loc);
      }
      
    {
    int idx;
    idx = 1;
    for (j in 1:F) {
      beta[t][j, j] = beta_diag[j]; // set positive diagonal loadings
      for (k in (j+1):F){
        beta[t][j, k] = 0; // set upper triangle values to 0
        }
      for (i in (j+1):P){
        beta[t][i, j] = beta_lower[t][idx]; // set lower diagonal loadings
        idx = idx + 1;
        }
      }
    }
  }
}
model {
  vector[P] mu;
  // priors
  beta_lower_loc ~ std_normal();
  // beta_lower_scale ~ std_normal();
  beta_lower_scale ~ cauchy(0, 0.2);
  // beta_lower_scale ~ lognormal(-2, 1);
  // beta_lower_scale ~ beta(2, 5);
  // beta_lower_scale ~ gamma(2, 1./10.);
  // beta_lower_scale ~ beta(2., 2.);
  
  // beta_lower_sd ~ std_normal();
  beta_lower_sd ~ normal(0, 0.1);
  
  // x_sd ~ gamma (2, 1./10.);
  // x_sd ~ std_normal();
  
  x_sd ~ normal(0, 0.1);
  
  for (t in 1:TS){
    z_fac_sd[t] ~ std_normal();
    z_beta_lower_sd[t] ~ std_normal();
    // x[t] ~ normal(beta[t] * (f_sd .* z_fac_sd[t]), x_sd);
    // x[t] ~ normal(beta[t] * z_fac_sd[t], x_sd);
    mu = beta[t] * z_fac_sd[t];
    for (p in 1:P){
      x[t][p] ~ normal(mu[p], x_sd[p]);
    }
  }
}

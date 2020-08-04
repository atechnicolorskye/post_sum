import numpy as np

import pdb

class GenerateSyntheticGraphs(object):
    def __init__(self, random_seed):
        self.random_seed = random_seed

    def rand_walk(self, dim, n_steps, b_co=0.1, x_co=1, stand=False):
        '''
        Generate for n_steps, a vanilla random walk with independent noise.
        Sample beta and x from unit Gaussain.
        b_co and x_co are the multipliers for the covariance matrices
        for beta and x respectively.
        stand standardises the output.
        '''
        # Define  mean and covariance
        self.mean = np.zeros(dim)
        self.z_cov = np.diag(np.ones(dim))

        # Generate x_0, beta_0, beta_noise, x_noise
        self.x_0, self.beta_0 = np.random.multivariate_normal(self.mean, self.z_cov, 2)
        self.beta_noise = np.random.multivariate_normal(self.mean, b_co * self.z_cov, (n_steps+1))
        self.x_noise = np.random.multivariate_normal(self.mean, x_co * self.z_cov, (n_steps+1))

        # pdb.set_trace()

        self.beta, self.data = [], []
        beta_ = self.beta_0 + self.beta_noise[0]
        self.beta.append(beta_)
        self.x = np.inner(beta_, self.x_0) + self.x_noise[0]
        self.data.append(self.x)
        for i in range(1, n_steps+1):
            beta_ = beta_ + self.beta_noise[i]
            self.beta.append(beta_)
            self.x = np.inner(beta_, self.x_0) + self.x_noise[i]
            self.data.append(self.x)

        self.beta = np.array(self.beta)
        self.data = np.array(self.data)

        if stand:
            mean = np.mean(self.data, 0)
            std = np.std(self.data, 0)
            return self.beta, (self.data - mean) / std
        else:
            return self.beta, self.data

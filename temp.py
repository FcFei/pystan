# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import matplotlib.pyplot as plt

import pystan 
import numpy as np
#import matplotlib
schools_code = """
data {
    int<lower=0> N; // number of schools
    vector[N] x;
    vector[N] y;
}
parameters {
    real alpha;
    real beta;
    real sigma;
}
model {
    y ~ normal(alpha+ beta*x,sigma);
}
"""

N = 1000
simulate_x = np.random.randn(N)
simulate_y = simulate_x + 0.05*np.random.randn(N)
schools_dat = {'N': N,
               'x': simulate_x,
               'y': simulate_y,}

fit = pystan.stan(model_code=schools_code, data=schools_dat,
                  iter=2000, chains=4)
all_data = fit.extract()
        
          
iter_step = np.arange(0,4000)
alpha = all_data['alpha']
abc = plt.plot(iter_step,alpha)
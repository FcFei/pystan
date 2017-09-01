# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import matplotlib.pyplot as plt
import time

import pystan
import pickle
from hashlib import md5
import numpy as np
#import matplotlib
def main():
    schools_code = """
    data {
        int<lower=0> N; // number of schools
        vector[N] y;        
    }
    parameters {
        real alpha;
        real beta;
        real sigma;    
    }
    model {     
        y[2:N] ~ normal(alpha + beta * y[1:(N-1)],sigma);
    }
    """
#y ~ log((x1 + exp(beta1 * log(x1)) + (x3 + exp(beta1 * log(x4)))*beta2) .* exp(x3*beta3) + x4*beta4) + normal(0,sigma);


    N_iter = 2000
    N = 1000
    simulate_x = np.random.randn(N)
    simulate_y = np.ones(N)    
    
    alpha = 2
    beta = 0.5
    sigma = 0.5  
    for ii in range(N-1):
        simulate_y[ii+1] = alpha + beta * simulate_y[ii] + sigma * simulate_x[ii]        
    
    schools_dat = {'N': N,
                   'y' : simulate_y}

    fit = stan_cache(model_code=schools_code, data=schools_dat,\
        iter = N_iter,chains = 4)
#    all_data = fit.extract()
    
    return fit



def stan_cache(model_code, model_name=None, **kwargs):
    """Use just as you would `stan`"""
    """We can specify the model name if we want more models to be stored"""
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    if model_name is None:
        cache_fn = 'cached-model-{}.pkl'.format(code_hash)
    else:
        cache_fn = 'cached-{}-{}.pkl'.format(model_name, code_hash)
    try:
        sm = pickle.load(open(cache_fn, 'rb'))
    except:
        sm = pystan.StanModel(model_code=model_code)
        with open(cache_fn, 'wb') as f:
            pickle.dump(sm, f)
    else:
        print("Using cached StanModel")
    return sm.sampling(**kwargs)


if __name__ == '__main__':
    start_time = time.time()
    abc = main()    
    print("--- %s seconds ---" % (time.time() - start_time))    
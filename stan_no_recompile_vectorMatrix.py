# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import matplotlib.pyplot as plt

import pystan
import pickle
from hashlib import md5
import numpy as np
#import matplotlib
def main():
    schools_code = """
    data {
        int<lower=0> N; // 
        int<lower=0> M; // number of parameters
        matrix[N,M] x;
        vector[N] y;
    }
    parameters {
        real alpha;
        vector[M] beta;
        real sigma;    
    }
    model {
        y ~ normal(alpha + x * beta,sigma);
    }
    """
    N_iter = 2000
    N = 1000
    M = 3
    alpha = 2
    beta = np.array([[2],[2],[2]])
    sigma = 0.5
    
    simulate_x = np.random.rand(N,3) + 1 
    simulate_y = alpha + simulate_x.dot(beta) + sigma*np.random.randn(N,1)
    simulate_y = simulate_y.reshape(N)  # important; translate (N,1) to (N)
    schools_dat = {'N': N,
                   'M': M,
                   'x': simulate_x,
                   'y': simulate_y,}
    
    fit = stan_cache(model_code=schools_code, data=schools_dat,\
        iter = N_iter,chains = 4)
#    all_data = fit.extract()
    fit.plot('alpha')
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
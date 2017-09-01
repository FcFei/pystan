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
    N_iter = 2000
    N = 1000
    alpha = 2
    beta = 3
    sigma = 0.5
    
    simulate_x = np.random.rand(N) + 1 
    simulate_y = alpha + beta * simulate_x + sigma*np.random.randn(N)
    schools_dat = {'N': N,
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
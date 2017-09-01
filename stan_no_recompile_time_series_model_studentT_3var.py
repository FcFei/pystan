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
    transformed data{
        matrix[N-3,3] X;
        vector[N-3] Y;
        X[:,1] = y[1:(N-3)];
        X[:,2] = y[2:(N-2)];
        X[:,3] = y[3:(N-1)];
        Y = y[4:N];
    }
    parameters {
        real alpha;
        vector[3] beta;
        real sigma; 
        real nu;
        real unknown;
    }
    model {
       Y ~ student_t(nu, alpha + X * beta ,sigma);

    }
    """
#        
    N_iter = 2000
    N = 1000
    nu = 1.5
    alpha = 0.5
    beta = np.array([[0.3],[0.4],[0.2]])
    sigma = 0.5      
    simulate_x  = np.random.standard_t(nu,[N,1])
    simulate_y = np.ones(N)    
    

    for ii in range(N-3):
        simulate_y[ii+3] = alpha + simulate_y[ii:(ii+3)].dot(beta) + sigma * simulate_x[ii]        
        
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
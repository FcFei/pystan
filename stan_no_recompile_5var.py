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
        vector[N] x1;
        vector[N] x2;
        vector[N] x3;
        vector[N] x4;
        vector[N] y;        
    }
    parameters {
        real<lower=1> beta1;
        real<upper=5> beta2;
        real<lower=0,upper=3> beta3;
        real<lower=5,upper=20> beta4;
        real sigma;    
    }
    model {     
        y ~ normal(log((x1 + exp(beta1 * log(x2)) + (x3 + exp(beta1 * log(x4)))*beta2) .* exp(x3*beta3) + x4*beta4),sigma);
    }
    """
#y ~ log((x1 + exp(beta1 * log(x1)) + (x3 + exp(beta1 * log(x4)))*beta2) .* exp(x3*beta3) + x4*beta4) + normal(0,sigma);


    N_iter = 2000
    N = 1000
    simulate_x1 = np.random.rand(N) + 1
    simulate_x2 = np.random.rand(N) + 1 
    simulate_x3 = np.random.rand(N) + 1 
    simulate_x4 = np.random.rand(N) + 1
#    simulate_x1 = np.random.rand(N) * 5 
#    simulate_x2 = np.random.rand(N) * 5 
#    simulate_x3 = np.random.rand(N) * 5 
#    simulate_x4 = np.random.rand(N) * 5 
    beta1 = 2
    beta2 = 2.5
    beta3 = 0.5
    beta4 = 10
    sigma = 0.5    
    
    
    
    simulate_y = np.log((simulate_x1 + simulate_x2**beta1 + \
    (simulate_x3 + simulate_x4**beta1)*beta2) * np.exp(simulate_x3*beta3) + simulate_x4*beta4) + \
                    sigma*np.random.randn(N)
    
    schools_dat = {'N': N,
                   'x1': simulate_x1,
                   'x2': simulate_x2,
                   'x3': simulate_x3,
                   'x4': simulate_x4,
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
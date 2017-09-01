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
        vector[N] y1;
        vector[N] z;
    }
    parameters {
        real alpha;
        real beta;
        real<lower=1,upper=8> gamma;
        real sigma;    
    }
    model {
        vector[N] inter_mid;
        for (i in 1:N)
        inter_mid[i] = exp(log(x1[i]) * gamma);        
        y1 ~ normal(alpha+ beta * inter_mid .* z,sigma);
    }
    """
#        for (i in 0:N-1){
#        inter_mid[i] <- x[i];        
#        }    
    N_iter = 2000
    N = 1000
    simulate_x = np.random.rand(N)+3
    simulate_z = np.random.randn(N)
    simulate_y = simulate_x ** 2 * simulate_z + 0.05*np.random.randn(N)
    
    schools_dat = {'N': N,
                   'x1': simulate_x,
                   'y1': simulate_y,
                   'z': simulate_z}
    
    fit = stan_cache(model_code=schools_code, data=schools_dat,\
        iter = N_iter,chains = 8)
#    all_data = fit.extract()
    fit.plot('gamma')
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
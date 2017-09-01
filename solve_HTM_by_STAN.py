# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 16:58:27 2017

@author: feic

"""

import read_data_from_excel
import do_Kalman_filter


from matplotlib.ticker import LinearLocator, FormatStrFormatter 
import matplotlib.pyplot as plt
import time

import pystan
import pickle
from hashlib import md5
import numpy as np
import pandas as pd
import statsmodels.formula.api as API


def main():
#==============================================================================
########## Use OLS to calcualte the time_inv part in a toy model
#    trend_data, time_inv_data, org_data = read_data_from_excel.create_toy_data()
#    my_model_data = define_time_inv_model_data(time_inv_data)
#     
#    fit, reg_std = generate_coef_by_LS(time_inv_data)
#    
#    my_model_code2 = define_model_time_varying_trend_component()
#    my_model_data2 = define_data_time_varying_trend_component_reg(fit,reg_std,org_data)
#    fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
#         iter = 2000, chains = 4)
#    return fit, fit2
#==============================================================================    


    
#==============================================================================
########## Use OLS to calcualte the time_inv part
#    trend_data, time_inv_data, org_data = read_data_from_excel.read_data_remove_trend()
######    my_model_data = define_time_inv_model_data(time_inv_data)
#     
#    fit, reg_std = generate_coef_by_LS(time_inv_data)
#    
#    my_model_code2 = define_model_time_varying_trend_component()
#    my_model_data2 = define_data_time_varying_trend_component_reg(fit,reg_std,org_data)
#    fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
#         iter = 2000,chains = 4)
#    return fit, fit2
#==============================================================================


#==============================================================================
########## Use OLS to calcualte the time_inv part and treat it as initial guess, 
########## in STAN, we estimate all the parameters using Month Mean data
#    trend_data, time_inv_data, org_data = read_data_from_excel.create_toy_data()
#    trend_data, time_inv_data, org_data = read_data_from_excel.read_data_remove_trend()
#    fit, reg_std = do_Kalman_filter.generate_coef_by_LS(time_inv_data)
#    
#    my_model_code2 = do_Kalman_filter.define_model_time_varying_trend_component_include_beta()
#    my_model_data2 = do_Kalman_filter.define_data_time_varying_trend_component_reg_include_beta(fit,reg_std,org_data)
#    fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
#         iter = 2000,chains = 4)
###### Plot the trend component filtered by the data    
#    trend_month = fit2.summary()['summary'][0:93,0]
#    data_with_date = trend_data.resample('M').iloc[:,0].copy()
#    data_with_date.iloc[:] = trend_month.copy()
#    read_data_from_excel.plot_against_date(data_with_date,org_data)
#    return fit, fit2
#==============================================================================

  
  
  
#==============================================================================
########## Use OLS to calcualte the time_inv part and treat it as initial guess, 
########## in STAN, we estimate all the parameters using all the data
##    trend_data, time_inv_data, org_data = read_data_from_excel.create_toy_data()
#    trend_data, time_inv_data, org_data = read_data_from_excel.read_data_remove_trend()
#    fit, reg_std = do_Kalman_filter.generate_coef_by_LS(time_inv_data)
#    
##    my_model_code2 = do_Kalman_filter.define_model_time_varying_trend_component_include_beta_all_data()
#    my_model_code2 = do_Kalman_filter.define_model_time_varying_trend_component_include_beta_all_data_reshape()
#    print "finish model"    
#    my_model_data2 = do_Kalman_filter.define_data_time_varying_trend_component_reg_include_beta_all_data(fit,reg_std,org_data)
#    fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
#         iter = 1000,chains = 1)
#         
###### Plot the trend component filtered by the data    
#    trend_month = fit2.summary()['summary'][0:93,0]
#    data_with_date = trend_data.resample('M').iloc[:,0].copy()
#    data_with_date.iloc[:] = trend_month.copy()
#    read_data_from_excel.plot_against_date(data_with_date,org_data)
#    return fit, fit2
#============================================================================== 
  
#==============================================================================
########## Use OLS to calcualte the time_inv part and treat it as initial guess, 
########## in STAN, we estimate all the parameters using all the data
#    trend_data, time_inv_data, org_data = read_data_from_excel.create_toy_data()
    trend_data, time_inv_data, org_data = read_data_from_excel.read_data_remove_trend()
    fit, reg_std = do_Kalman_filter.generate_coef_by_LS(time_inv_data)
    
#    my_model_code2 = do_Kalman_filter.define_model_time_varying_trend_component_include_beta_all_data()
#    my_model_code2 = do_Kalman_filter.define_model_DHRD_all_data_reshape()
#    my_model_code2 = do_Kalman_filter.define_model_DHRD_all_data_reshape_transPara()
    my_model_code2 = do_Kalman_filter.define_model_DHRD_all_data_reshape_transPara_studentt()
    print "finish model"    
    my_model_data2 = do_Kalman_filter.define_data_DHRD(fit,reg_std,org_data)
    fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
         iter = 1000,chains = 1)
         
##### Plot the trend component filtered by the data    
    trend_month = fit2.summary()['summary'][0:93,0]
    data_with_date = trend_data.resample('M').iloc[:,0].copy()
    data_with_date.iloc[:] = trend_month.copy()
    read_data_from_excel.plot_against_date(data_with_date,org_data)
    return fit, fit2
#==============================================================================  

 
  
  
#==============================================================================
########## Use STAN to calculate the time_inv part  
#     trend_data, time_inv_data, org_data = read_data_from_excel.read_data_remove_trend()
#     my_model_code = define_time_inv_model_code()
#     my_model_data = define_time_inv_model_data(time_inv_data)
# #    
#     fit1 = stan_cache(model_code=my_model_code, data=my_model_data,\
#         iter = 1000,chains = 4)      
     
#     my_model_code2 = define_model_time_varying_trend_component()
#     my_model_data2 = define_data_time_varying_trend_component(fit1,org_data)
#     fit2 = stan_cache(model_code=my_model_code2, data=my_model_data2,\
#         iter = 1000,chains = 4)
#     return fit1,fit2
#==============================================================================




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
   fit_time_inv, fit_time_trend = main()
   print("--- %s seconds ---" % (time.time() - start_time))    
   
   f = open('DHRD_outcome_transpara_studentt','w')
   print >> f, (time.time() - start_time), fit_time_trend, fit_time_inv.summary()
   f.close()
   
   print fit_time_trend
   print fit_time_inv.summary()

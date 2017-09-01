# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:37:19 2017

@author: feic
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as API








def generate_coef_by_LS(time_inv_data):
    fit = API.ols(formula="time_inv_price ~ time_inv_space + time_inv_area +  \
    time_inv_age + time_inv_dumgar + time_inv_dumber -1",data=time_inv_data).fit()
    para = fit.params    
    reg_std = np.std(time_inv_data['time_inv_price'] - \
                     para[0] * time_inv_data['time_inv_space'] - \
                     para[1] * time_inv_data['time_inv_area'] - \
                     para[2] * time_inv_data['time_inv_age'] - \
                     para[3] * time_inv_data['time_inv_dumgar'] - \
                     para[4] * time_inv_data['time_inv_dumber'])
#    print fit.summary(),reg_std
    return fit, reg_std


def define_data_time_varying_trend_component_reg_include_beta_all_data(fit,reg_std,org_data):    
# This function uses the input from OLS to record the parameter betas.    
    beta1 = fit.params['time_inv_space']
    beta2 = fit.params['time_inv_area']
    beta3 = fit.params['time_inv_age']
    beta4 = fit.params['time_inv_dumgar']
    beta5 = fit.params['time_inv_dumber']
    sigma = reg_std
    print "time_inv error is", sigma, "\n"
    price = org_data.iloc[:,0]   ####### Here using the trend_data should give same result
    space = org_data.iloc[:,7]
    occupy_area = org_data.iloc[:,6]
    age = org_data.iloc[:,1]
    with_garage = org_data.iloc[:,4]
    with_storage = org_data.iloc[:,5]        
    
#    fitted_price = beta1 * space + beta2 * occupy_area + beta3 * age + \
#                    beta4 * with_garage + beta5 * with_storage

    MON_price = price.resample('M')    ####### Here using the trend_data should give same result
#    MON_fitted_timeinv_price = fitted_price.resample('M') # this should be the same as timing beta with trend component
    
   
    
#    obs_noise = 0.2 * sigma  # the factor number should be same as in "read_data_from_excel" L35         
    obs_noise =  np.std(MON_price)  # the factor number should be same as in "read_data_from_excel" L35       
    print "initial obs_noise is", obs_noise, "\n"    
    
    ABC = price.groupby([price.index.year,price.index.month]).agg('count')    
#    print ABC
    M = len(MON_price.index) 
    model_data = {'M': M,
                  'N': len(price.index),
                  'number_of_data_per_month': ABC,   
                  'MON_price': price,
                  'MON_space': space,
                  'MON_area': occupy_area,
                  'MON_age': age,
                  'MON_with_garage': with_garage,
                  'MON_with_storage': with_storage,
                  'obs_noise_init': obs_noise,
                  'sigma_epsilon_init': sigma,
                  'beta1_init': beta1,
                  'beta2_init': beta2,
                  'beta3_init': beta3,
                  'beta4_init': beta4,
                  'beta5_init': beta5}
    return model_data 


def define_data_DHRD(fit,reg_std,org_data):    
# This function uses the input from OLS to record the parameter betas.    
    beta1 = fit.params['time_inv_space']
    beta2 = fit.params['time_inv_area']
    beta3 = fit.params['time_inv_age']
    beta4 = fit.params['time_inv_dumgar']
    beta5 = fit.params['time_inv_dumber']
    sigma_epsilon_init = reg_std
#    print "time_inv error is", sigma, "\n"
    price = org_data.iloc[:,0]   ####### Here using the trend_data should give same result
    space = org_data.iloc[:,7]
    occupy_area = org_data.iloc[:,6]
    age = org_data.iloc[:,1]
    with_garage = org_data.iloc[:,4]
    with_storage = org_data.iloc[:,5]        
        
    if_DH = org_data.iloc[:,8]
    if_RD = org_data.iloc[:,9]
    if_house = org_data.iloc[:,10]
    if_apartment = org_data.iloc[:,11]

    number_of_data_per_month = price.groupby([price.index.year,price.index.month]).agg('count')    
    
    model_data = {'M': len(price.resample('M').index) ,
                  'N': len(price.index),
                  'number_of_data_per_month': number_of_data_per_month,   
                  'MON_price': price,
                  'MON_space': space,
                  'MON_area': occupy_area,
                  'MON_age': age,
                  'MON_with_garage': with_garage,
                  'MON_with_storage': with_storage,
                  'if_DH': if_DH,
                  'if_house': if_house,                   
                  'sigma_epsilon_init': sigma_epsilon_init,
                  'beta1_init': beta1,
                  'beta2_init': beta2,
                  'beta3_init': beta3,
                  'beta4_init': beta4,
                  'beta5_init': beta5}
    return model_data 


def define_data_time_varying_trend_component_reg_include_beta(fit,reg_std,org_data):    
# This function uses the input from OLS to record the parameter betas.    
    beta1 = fit.params['time_inv_space']
    beta2 = fit.params['time_inv_area']
    beta3 = fit.params['time_inv_age']
    beta4 = fit.params['time_inv_dumgar']
    beta5 = fit.params['time_inv_dumber']
    sigma = reg_std
    print "time_inv error is", sigma, "\n"
    price = org_data.iloc[:,0]   ####### Here using the trend_data should give same result
    space = org_data.iloc[:,7]
    occupy_area = org_data.iloc[:,6]
    age = org_data.iloc[:,1]
    with_garage = org_data.iloc[:,4]
    with_storage = org_data.iloc[:,5]        
    
#    fitted_price = beta1 * space + beta2 * occupy_area + beta3 * age + \
#                    beta4 * with_garage + beta5 * with_storage

    MON_price = price.resample('M')    ####### Here using the trend_data should give same result
#    MON_fitted_timeinv_price = fitted_price.resample('M') # this should be the same as timing beta with trend component
    
   
    
#    obs_noise = 0.2 * sigma  # the factor number should be same as in "read_data_from_excel" L35         
    obs_noise =  np.std(MON_price)  # the factor number should be same as in "read_data_from_excel" L35       
    print "initial obs_noise is", obs_noise, "\n"    
    
    M = len(MON_price.index) 
    model_data = {'M': M,
                  'MON_price': MON_price,
                  'MON_space': space.resample('M'),
                  'MON_area': occupy_area.resample('M'),
                  'MON_age': age.resample('M'),
                  'MON_with_garage': with_garage.resample('M'),
                  'MON_with_storage': with_storage.resample('M'),
                  'obs_noise_init': obs_noise,
                  'beta1_init': beta1,
                  'beta2_init': beta2,
                  'beta3_init': beta3,
                  'beta4_init': beta4,
                  'beta5_init': beta5}
    return model_data 





def define_data_time_varying_trend_component_reg(fit,reg_std,org_data):    
# This function uses the input from OLS to record the parameter betas.    
    beta1 = fit.params['time_inv_space']
    beta2 = fit.params['time_inv_area']
    beta3 = fit.params['time_inv_age']
    beta4 = fit.params['time_inv_dumgar']
    beta5 = fit.params['time_inv_dumber']
    sigma = reg_std
    print "time_inv error is", sigma, "\n"
    price = org_data.iloc[:,0]   ####### Here using the trend_data should give same result
    space = org_data.iloc[:,7]
    occupy_area = org_data.iloc[:,6]
    age = org_data.iloc[:,1]
    with_garage = org_data.iloc[:,4]
    with_storage = org_data.iloc[:,5]        
    
    fitted_price = beta1 * space + beta2 * occupy_area + beta3 * age + \
                    beta4 * with_garage + beta5 * with_storage
#    common_trend_component = price - fitted_price 
#    
#    MON_common_trend_component = common_trend_component.resample('M')
    MON_price = price.resample('M')    ####### Here using the trend_data should give same result
    MON_fitted_timeinv_price = fitted_price.resample('M') # this should be the same as timing beta with trend component
    
    print "Std of trend_component monethly mean is", np.std(MON_price),"\n"    
    
#    obs_noise = 0.2 * sigma  # the factor number should be same as in "read_data_from_excel" L35         
    obs_noise = np.std(MON_price)  # the factor number should be same as in "read_data_from_excel" L35       
    
    M = len(MON_price.index) 
    model_data = {'M': M,
                  'MON_price': MON_price,
                  'MON_fitted_timeinv_price': MON_fitted_timeinv_price,
                  'obs_noise': obs_noise}
    return model_data  

def define_data_time_varying_trend_component(fit,org_data):  
# This function uses the input from STAN to record the parameter betas.        
    beta1 = fit.summary()['summary'][0,0]
    beta2 = fit.summary()['summary'][1,0]
    beta3 = fit.summary()['summary'][2,0]
    beta4 = fit.summary()['summary'][3,0]
    beta5 = fit.summary()['summary'][4,0]
    sigma = fit.summary()['summary'][5,0]
    
    price = org_data.iloc[:,0]
    space = org_data.iloc[:,7]
    occupy_area = org_data.iloc[:,6]
    age = org_data.iloc[:,1]
    with_garage = org_data.iloc[:,4]
    with_storage = org_data.iloc[:,5]        
    
    fitted_price = beta1 * space + beta2 * occupy_area + beta3 * age + \
                    beta4 * with_garage + beta5 * with_storage
#    common_trend_component = price - fitted_price 
#    
#    MON_common_trend_component = common_trend_component.resample('M')
    MON_price = price.resample('M')
    MON_fitted_timeinv_price = fitted_price.resample('M') 
    print "Std of trend_component monethly mean is", np.std(MON_price),"\n"
    obs_noise = 0.1 * sigma     
#    M = len(common_trend_component.resample('M').index)  
    
    M = len(MON_price.index) 
    model_data = {'M': M,
                  'MON_price': MON_price,
                  'MON_fitted_timeinv_price': MON_fitted_timeinv_price,
                  'obs_noise': obs_noise}
    return model_data                  

#    model_data = {'M': M,
#                  'MON_price': MON_price,
#                  'MON_fitted_timeinv_price': MON_fitted_timeinv_price,
#                  'obs_noise': obs_noise,
#                  'beta1_init': beta1,
#                  'beta2_init': beta2,
#                  'beta3_init': beta3,
#                  'beta4_init': beta4,
#                  'beta5_init': beta5}

def define_model_time_varying_trend_component_include_beta_all_data_reshape():
    model_code = """
    data {
        int<lower=0> M; // number of months
        int N;  // number of total samples
        int number_of_data_per_month[M];
        vector[N] MON_price;           
        vector[N] MON_space;
        vector[N] MON_area;
        vector[N] MON_age;
        vector[N] MON_with_garage;
        vector[N] MON_with_storage;
        real obs_noise_init;
        real sigma_epsilon_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector<lower = 0>[M] MON_common_trend;
        real alpha0;
        real alpha1;
        real<lower = 0> sigma2;
        real<lower = 0> obs_noise;
        real<lower = 0> sigma_epsilon;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
    }

    model {   
        int sel_left;
        int sel_right;    
        vector[N] TREND;
        alpha0 ~ normal(0,3);
        alpha1 ~ normal(1,3);
        obs_noise ~ normal(obs_noise_init,abs(0.1*obs_noise_init)); // The uncertainty cannot be too large (0.3)
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        sigma_epsilon ~ normal(sigma_epsilon_init,abs(0.1*sigma_epsilon_init));
        
        MON_common_trend[2:M] ~ normal(alpha0 + alpha1 * MON_common_trend[1:M-1],sigma2);   
        
        TREND[1:number_of_data_per_month[1]] = rep_vector(MON_common_trend[1],number_of_data_per_month[1]);
        for (jj in 2:M) {
            TREND[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_common_trend[jj],number_of_data_per_month[jj]);
        }          
        
        MON_price ~ normal(TREND + beta1 * MON_space + beta2 * MON_area + beta3 * MON_age + beta4 * MON_with_garage + beta5 * MON_with_storage,sigma_epsilon);       
    }
    """
    return model_code

def define_model_DHRD_all_data_reshape():
    model_code = """
    data {
        int<lower=0> M; // number of months
        int N;  // number of total samples
        int number_of_data_per_month[M];
        vector[N] MON_price;           
        vector[N] MON_space;
        vector[N] MON_area;
        vector[N] MON_age;
        vector[N] MON_with_garage;
        vector[N] MON_with_storage;
        
        vector[N] if_house;       
        vector[N] if_apartment;
        vector[N] if_DH;
        vector[N] if_RD;
        
        real sigma_epsilon_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector<lower = 0>[M] MON_common_trend;
        vector<lower = 0>[M] MON_house_trend;
        vector<lower = 0>[M] MON_apartment_trend;
        vector<lower = 0>[M] MON_RD_trend;
        vector<lower = 0>[M] MON_DH_trend;
        
        real<lower = 0> sigma_common_trend;
        real<lower = 0> sigma_house_trend; 
        real<lower = 0> sigma_apartment_trend;
        real<lower = 0> sigma_RD_trend;
        real<lower = 0> sigma_DH_trend;   
        
        real<lower = 0> sigma_epsilon;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
    }

    model {     
        vector[N] TREND_common;
        vector[N] TREND_house;
        vector[N] TREND_apartment;
        vector[N] TREND_DH;
        vector[N] TREND_RD;
        
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        sigma_epsilon ~ normal(sigma_epsilon_init,abs(0.1*sigma_epsilon_init));
        
        MON_common_trend[2:M] ~ normal(MON_common_trend[1:M-1],sigma_common_trend);   
        MON_house_trend[2:M] ~ normal(MON_house_trend[1:M-1],sigma_house_trend);
        MON_apartment_trend[2:M] ~ normal(MON_apartment_trend[1:M-1],sigma_apartment_trend);
        MON_RD_trend[2:M] ~ normal(MON_RD_trend[1:M-1],sigma_RD_trend);
        MON_DH_trend[2:M] ~ normal(MON_DH_trend[1:M-1],sigma_DH_trend);
        
        TREND_common[1:number_of_data_per_month[1]] = rep_vector(MON_common_trend[1],number_of_data_per_month[1]);
        TREND_house[1:number_of_data_per_month[1]] = rep_vector(MON_house_trend[1],number_of_data_per_month[1]);
        TREND_apartment[1:number_of_data_per_month[1]] = rep_vector(MON_apartment_trend[1],number_of_data_per_month[1]);
        TREND_RD[1:number_of_data_per_month[1]] = rep_vector(MON_RD_trend[1],number_of_data_per_month[1]);
        TREND_DH[1:number_of_data_per_month[1]] = rep_vector(MON_DH_trend[1],number_of_data_per_month[1]);
        
        
        for (jj in 2:M) {
            TREND_common[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_common_trend[jj],number_of_data_per_month[jj]);
            TREND_house[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_house_trend[jj],number_of_data_per_month[jj]);
            TREND_apartment[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_apartment_trend[jj],number_of_data_per_month[jj]);
            TREND_RD[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_RD_trend[jj],number_of_data_per_month[jj]);
            TREND_DH[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_DH_trend[jj],number_of_data_per_month[jj]);
        }          
        
        MON_price ~ normal(TREND_common + TREND_house .* if_house + TREND_apartment .* if_apartment + TREND_RD .* if_RD + TREND_DH .* if_DH + beta1 * MON_space + beta2 * MON_area + beta3 * MON_age + beta4 * MON_with_garage + beta5 * MON_with_storage,sigma_epsilon);       
    }
    """
    return model_code



def define_model_DHRD_all_data_reshape_transPara():
    model_code = """
    data {
        int<lower=0> M; // number of months
        int N;  // number of total samples
        int number_of_data_per_month[M];
        vector[N] MON_price;           
        vector[N] MON_space;
        vector[N] MON_area;
        vector[N] MON_age;
        vector[N] MON_with_garage;
        vector[N] MON_with_storage;
        
        vector[N] if_house;       
        vector[N] if_DH;
        
        real sigma_epsilon_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector<lower = 0>[M] MON_common_trend;
        vector[M] MON_house_trend;
        vector[M] MON_DH_trend;
        
        real<lower = 0> sigma_common_trend;
        real<lower = 0> sigma_house_trend; 
        real<lower = 0> sigma_DH_trend;   
        
        real<lower = 0> sigma_epsilon;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
    }
            
    model {
        vector[N] TREND_common;
        vector[N] TREND_house;
        vector[N] TREND_DH;  
    
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        sigma_epsilon ~ normal(sigma_epsilon_init,abs(0.3*sigma_epsilon_init));
        
        MON_common_trend[2:M] ~ normal(MON_common_trend[1:M-1],sigma_common_trend);   
        MON_house_trend[2:M] ~ normal(MON_house_trend[1:M-1],sigma_house_trend);
        MON_DH_trend[2:M] ~ normal(MON_DH_trend[1:M-1],sigma_DH_trend);        

        TREND_common[1:number_of_data_per_month[1]] = rep_vector(MON_common_trend[1],number_of_data_per_month[1]);
        TREND_house[1:number_of_data_per_month[1]] = rep_vector(MON_house_trend[1],number_of_data_per_month[1]);
        TREND_DH[1:number_of_data_per_month[1]] = rep_vector(MON_DH_trend[1],number_of_data_per_month[1]);
                
        for (jj in 2:M) {
            TREND_common[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_common_trend[jj],number_of_data_per_month[jj]);
            TREND_house[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_house_trend[jj],number_of_data_per_month[jj]);            
            TREND_DH[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_DH_trend[jj],number_of_data_per_month[jj]);
         }
         
        MON_price ~ normal(TREND_common + TREND_house .* if_house + TREND_DH .* if_DH + beta1 * MON_space + beta2 * MON_area + beta3 * MON_age + beta4 * MON_with_garage + beta5 * MON_with_storage,sigma_epsilon);       
    }
    """
    return model_code

def define_model_DHRD_all_data_reshape_transPara_studentt():
    model_code = """
    data {
        int<lower=0> M; // number of months
        int N;  // number of total samples
        int number_of_data_per_month[M];
        vector[N] MON_price;           
        vector[N] MON_space;
        vector[N] MON_area;
        vector[N] MON_age;
        vector[N] MON_with_garage;
        vector[N] MON_with_storage;
        
        vector[N] if_house;       
        vector[N] if_DH;
        
        real sigma_epsilon_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector<lower = 0>[M] MON_common_trend;
        vector[M] MON_house_trend;
        vector[M] MON_DH_trend;
        
        real<lower = 0> sigma_common_trend;
        real<lower = 0> sigma_house_trend; 
        real<lower = 0> sigma_DH_trend;   
        
        real<lower = 0> sigma_epsilon;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
                
        real<lower = 0> nu_price;
        real<lower = 0> nu_common_trend;
    } 
    
    model {    
        vector[N] TREND_common;
        vector[N] TREND_house;
        vector[N] TREND_DH;          
        
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        sigma_epsilon ~ normal(sigma_epsilon_init,abs(0.3*sigma_epsilon_init));
        
        MON_common_trend[2:M] ~ student_t(nu_common_trend, MON_common_trend[1:M-1],sigma_common_trend);   
        MON_house_trend[2:M] ~ normal(MON_house_trend[1:M-1],sigma_house_trend);
        MON_DH_trend[2:M] ~ normal(MON_DH_trend[1:M-1],sigma_DH_trend);        

        TREND_common[1:number_of_data_per_month[1]] = rep_vector(MON_common_trend[1],number_of_data_per_month[1]);
        TREND_house[1:number_of_data_per_month[1]] = rep_vector(MON_house_trend[1],number_of_data_per_month[1]);
        TREND_DH[1:number_of_data_per_month[1]] = rep_vector(MON_DH_trend[1],number_of_data_per_month[1]);
                
        for (jj in 2:M) {
            TREND_common[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_common_trend[jj],number_of_data_per_month[jj]);
            TREND_house[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_house_trend[jj],number_of_data_per_month[jj]);            
            TREND_DH[(sum(number_of_data_per_month[1:(jj-1)])+1):sum(number_of_data_per_month[1:jj])] = rep_vector(MON_DH_trend[jj],number_of_data_per_month[jj]);
         }
        
        MON_price ~ student_t(nu_price, TREND_common + TREND_house .* if_house + TREND_DH .* if_DH + beta1 * MON_space + beta2 * MON_area + beta3 * MON_age + beta4 * MON_with_garage + beta5 * MON_with_storage,sigma_epsilon);       
    }
    """
    return model_code






def define_model_time_varying_trend_component_include_beta_all_data():
    model_code = """
    data {
        int<lower=0> M; // number of months
        int N;  // number of total samples
        int number_of_data_per_month[M];
        vector[N] MON_price;           
        vector[N] MON_space;
        vector[N] MON_area;
        vector[N] MON_age;
        vector[N] MON_with_garage;
        vector[N] MON_with_storage;
        real obs_noise_init;
        real sigma_epsilon_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector<lower = 0>[M] MON_common_trend;
        real alpha0;
        real alpha1;
        real<lower = 0> sigma2;
        real<lower = 0> obs_noise;
        real<lower = 0> sigma_epsilon;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
    }
    model {   
        int sel_left;
        int sel_right;        
        alpha0 ~ normal(0,3);
        alpha1 ~ normal(1,3);      
        obs_noise ~ normal(obs_noise_init,abs(0.1*obs_noise_init)); // The uncertainty cannot be too large (0.3)
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        sigma_epsilon ~ normal(sigma_epsilon_init,abs(0.1*sigma_epsilon_init));
        MON_common_trend[2:M] ~ normal(alpha0 + alpha1 * MON_common_trend[1:M-1],sigma2);   
        MON_price[1:number_of_data_per_month[1]] ~ normal(MON_common_trend[1] + beta1 * MON_space[1:number_of_data_per_month[1]]  + beta2 * MON_area[1:number_of_data_per_month[1]]  + beta3 * MON_age[1:number_of_data_per_month[1]]  + beta4 * MON_with_garage[1:number_of_data_per_month[1]]  + beta5 * MON_with_storage[1:number_of_data_per_month[1]],sigma_epsilon);

        for (jjj in 2:M) {
            sel_left = sum(number_of_data_per_month[1:(jjj-1)])+1;
            sel_right = sum(number_of_data_per_month[1:jjj]);
            MON_price[sel_left : sel_right] ~ normal(MON_common_trend[jjj] + beta1 * MON_space[sel_left : sel_right]  + beta2 * MON_area[sel_left : sel_right]  + beta3 * MON_age[sel_left : sel_right]  + beta4 * MON_with_garage[sel_left : sel_right]  + beta5 * MON_with_storage[sel_left : sel_right],sigma_epsilon);
            } 
    }
    """
    return model_code 





def define_model_time_varying_trend_component_include_beta():
    model_code = """
    data {
        int<lower=0> M; // number of samples
        vector[M] MON_price;           
        vector[M] MON_space;
        vector[M] MON_area;
        vector[M] MON_age;
        vector[M] MON_with_garage;
        vector[M] MON_with_storage;
        real obs_noise_init;
        real beta1_init;
        real beta2_init;
        real beta3_init;
        real beta4_init;
        real beta5_init;
    }
    parameters {
        vector[M] MON_common_trend;
        real alpha0;
        real alpha1;
        real sigma2;
        real obs_noise;
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;
    }
    model {   
        alpha0 ~ normal(0,3);
        alpha1 ~ normal(1,3);
        obs_noise ~ normal(obs_noise_init,abs(0.1*obs_noise_init)); // The uncertainty cannot be too large (0.3)
        beta1 ~ normal(beta1_init,abs(0.1*beta1_init));
        beta2 ~ normal(beta2_init,abs(0.1*beta2_init));
        beta3 ~ normal(beta3_init,abs(0.1*beta3_init));
        beta4 ~ normal(beta4_init,abs(0.1*beta4_init));
        beta5 ~ normal(beta5_init,abs(0.1*beta5_init));
        MON_common_trend[2:M] ~ normal(alpha0 + alpha1 * MON_common_trend[1:M-1],sigma2);        
        MON_price ~ normal(MON_common_trend + beta1 * MON_space + beta2 * MON_area + beta3 * MON_age + beta4 * MON_with_garage + beta5 * MON_with_storage,obs_noise);
    }
    """
    return model_code 




    
    
def define_model_time_varying_trend_component():
    model_code = """
    data {
        int<lower=0> M; // number of samples
        vector[M] MON_price;           
        vector[M] MON_fitted_timeinv_price;
        real obs_noise;
    }
    parameters {
        vector[M] MON_common_trend;
        real alpha0;
        real alpha1;
        real sigma2;   
    }
    model {   
        alpha0 ~ normal(0,3);
        alpha1 ~ normal(1,3);
        MON_common_trend[2:M] ~ normal(alpha0 + alpha1 * MON_common_trend[1:M-1],sigma2);        
        MON_price ~ normal(MON_common_trend + MON_fitted_timeinv_price,obs_noise);
    }
    """
    return model_code 


def define_time_inv_model_data(time_inv_data):
    N = len(time_inv_data.index)
    price = time_inv_data.iloc[:,0]
    space = time_inv_data.iloc[:,7]
    occupy_area = time_inv_data.iloc[:,6]
    age = time_inv_data.iloc[:,1]
    with_garage = time_inv_data.iloc[:,4]
    with_storage = time_inv_data.iloc[:,5]
    
    model_data = {'N': N,
                  'price': price,
                   'space': space,
                   'occupy_area': occupy_area,
                   'age': age,
                   'with_garage': with_garage,
                   'with_storage' : with_storage}
    return model_data


def define_time_inv_model_code():
    model_code = """
    data {
        int<lower=0> N; // number of samples
        vector[N] price;           
        vector[N] space;        
        vector[N] occupy_area;
        vector[N] age;
        vector[N] with_garage;
        vector[N] with_storage;   
    }
    parameters {
        real beta1;
        real beta2;
        real beta3;
        real beta4;
        real beta5;     
        real sigma;    
    }
    model {   
        vector[N] number_related;
        vector[N] yesno_related;
        number_related = space * beta1 + occupy_area * beta2 + age * beta3;
        yesno_related = with_garage * beta4 + with_storage * beta5;
        price ~ normal(number_related + yesno_related,sigma);
    }
    """
    return model_code


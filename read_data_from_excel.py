# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:37:19 2017

@author: feic
"""

import numpy as np
import pandas as pd
import xlwt

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as API

def create_toy_data():
    DATA_input = pd.read_excel("Test_KF_in_STAN.xlsx",sheetname="appartment")
    DATA_sorted = DATA_input.sort_index()
    
    DATA_selected = DATA_sorted
    
    
    beta1 = 0.005
    beta2 = 0.003
    beta3 = -0.005
    beta4 = 0.22
    beta5 = 0.12
    sigma = 0.2
    
    mu_0 = 5
    alpha0 = 0
    alpha1 = 1
    sigma2 = 0.01   # This error refers to the process noise in Wiki.
    obs_error = 0.2 * sigma    # This error refers to the observation/measurement noise in Wiki. 
    print "obs_error used in creating data is", obs_error, "\n"
    
#    true_price = DATA_selected.iloc[:,0].copy()    
    true_data = DATA_selected.copy()
    
#    print DATA_selected.iloc[:,0]
    trend_data = pd.DataFrame(get_trend_all(true_data.copy()))
    trend_data = trend_data.rename(columns = {"price":"Trend_price","age":"Trend_age",\
    "fw_type":"Trend_fw_type","soc2_123":"Trend_soc2_123", "dumgar":"Trend_dumgar","dumber":"Trend_dumber",\
    "occupy_area":"Trend_area", "space":"Trend_space"})
    
#    true_price = pd.DataFrame(true_price)
    true_data = pd.DataFrame(true_data)
    true_data = true_data.rename(columns = {"price":"Trend_price","age":"Trend_age",\
    "fw_type":"Trend_fw_type","soc2_123":"Trend_soc2_123", "dumgar":"Trend_dumgar","dumber":"Trend_dumber",\
    "occupy_area":"Trend_area", "space":"Trend_space"})
    

    time_inv_data = pd.DataFrame(true_data - trend_data)

    time_inv_data['Trend_price'] = beta1 * time_inv_data['Trend_space'] + \
                                      beta2 * time_inv_data['Trend_area'] + \
                                      beta3 * time_inv_data['Trend_age'] + \
                                      beta4 * time_inv_data['Trend_dumgar'] + \
                                      beta5 * time_inv_data['Trend_dumber'] + \
                                      np.random.normal(0,sigma,len(time_inv_data.index))
                                      
#============================================================================== Incorrect AR(1) of price index
 #    trend_data_price = trend_data.iloc[:,0]
 #    trend_price_month_mean = trend_data_price.resample('M')
 #    for jj in range(1,len(trend_price_month_mean.index)): 
 #        trend_price_month_mean[jj] = alpha0 + alpha1 * trend_price_month_mean[jj-1] + np.random.normal(0,sigma2,1)
#==============================================================================
                                      
#==============================================================================
    trend_data_price = trend_data.iloc[:,0]
    trend_AR_component = trend_data_price.resample('M')  ## just for initiation to get the same size and index
    trend_price_month_mean = trend_data_price.resample('M')  ## just for initiation to get the same size and index                            

    space_month_mean = trend_data['Trend_space'].resample('M')  
    area_month_mean = trend_data['Trend_area'].resample('M')
    age_month_mean = trend_data['Trend_age'].resample('M')   
    dumgar_month_mean = trend_data['Trend_dumgar'].resample('M') 
    dumber_month_mean = trend_data['Trend_dumber'].resample('M') 
    
    trend_AR_component.iloc[0] = mu_0
    trend_price_month_mean.iloc[0] = trend_AR_component.iloc[0] \
                                    + beta1 * space_month_mean.iloc[0] \
                                    + beta2 * area_month_mean.iloc[0] \
                                    + beta3 * age_month_mean.iloc[0] \
                                    + beta4 * dumgar_month_mean.iloc[0] \
                                    + beta5 * dumber_month_mean.iloc[0] \
                                    + np.random.normal(0,obs_error,1)
    for jj in range(1,len(trend_AR_component.index)):
        trend_AR_component.iloc[jj] = alpha0 + alpha1 * trend_AR_component[jj-1] + np.random.normal(0,sigma2,1)
        trend_price_month_mean.iloc[jj] = trend_AR_component.iloc[jj]  \
                                    + beta1 * space_month_mean.iloc[jj] \
                                    + beta2 * area_month_mean.iloc[jj] \
                                    + beta3 * age_month_mean.iloc[jj] \
                                    + beta4 * dumgar_month_mean.iloc[jj] \
                                    + beta5 * dumber_month_mean.iloc[jj] \
                                    + np.random.normal(0,obs_error,1)
                                          
#==============================================================================
    
    
    for i in range(len(trend_price_month_mean.index)):        
        trend_data_price.loc[(trend_data_price.index.month == trend_price_month_mean.index[i].month) & \
        (trend_data_price.index.year == trend_price_month_mean.index[i].year)] = trend_price_month_mean.iloc[i]
    
    true_data = pd.DataFrame(trend_data + time_inv_data)
    Time_inv_data = time_inv_data.rename(columns = {"Trend_price":"time_inv_price","Trend_age":"time_inv_age",\
    "Trend_fw_type":"time_inv_fw_type","Trend_soc2_123":"time_inv_soc2_123", "Trend_dumgar":"time_inv_dumgar","Trend_dumber":"time_inv_dumber",\
    "Trend_area":"time_inv_area", "Trend_space":"time_inv_space"})
    
    Org_data = true_data.copy()
    return trend_data, Time_inv_data, Org_data




    

def read_data_remove_trend():
    DATA_input = pd.read_excel("preprocessed_DH_RD_data_v3_short.xlsx",sheetname="DHRD")
#    DATA_input = pd.read_excel("Cleaned_preprocess_data_short.xlsx",sheetname="appartment")
#    DATA_input = pd.read_excel("local_hpi.xlsx",sheetname="data")
    
    DATA_sorted = DATA_input.sort_index()
    
    #==============================================================================
    # all properties: 2,9,10,11,15,16,17,18,19,20,21,22,23,24
    # in Cleaned_preprocess_data: 8-16
    #==============================================================================
    DATA_selected = DATA_sorted
    
    
#    true_price = DATA_selected.iloc[:,0].copy()    
    true_data = DATA_selected.copy()
    
#    print DATA_selected.iloc[:,0]
    trend_data = pd.DataFrame(get_trend_all(true_data.copy()))
    trend_data = trend_data.rename(columns = {"price":"Trend_price","age":"Trend_age",\
    "fw_type":"Trend_fw_type","soc2_123":"Trend_soc2_123", "dumgar":"Trend_dumgar","dumber":"Trend_dumber",\
    "occupy_area":"Trend_area", "space":"Trend_space"})
    
#    true_price = pd.DataFrame(true_price)
    true_data = pd.DataFrame(true_data)
    true_data = true_data.rename(columns = {"price":"Trend_price","age":"Trend_age",\
    "fw_type":"Trend_fw_type","soc2_123":"Trend_soc2_123", "dumgar":"Trend_dumgar","dumber":"Trend_dumber",\
    "occupy_area":"Trend_area", "space":"Trend_space"})
    
    
    time_inv_data = pd.DataFrame(true_data - trend_data)
    
    Time_inv_data = time_inv_data.rename(columns = {"Trend_price":"time_inv_price","Trend_age":"time_inv_age",\
    "Trend_fw_type":"time_inv_fw_type","Trend_soc2_123":"time_inv_soc2_123", "Trend_dumgar":"time_inv_dumgar","Trend_dumber":"time_inv_dumber",\
    "Trend_area":"time_inv_area", "Trend_space":"time_inv_space"})
    
#    Time_inv_data = pd.concat([ABC,DATA_sorted],axis=1)
    Org_data = DATA_sorted.copy()
    return trend_data, Time_inv_data, Org_data



def get_trend(DATA):
    DATA_month_mean = DATA.resample('M')
#    print range(len(DATA_month_mean.index))    
    for i in range(len(DATA_month_mean.index)):
#        print DATA_month_mean.index[i]
        DATA.loc[(DATA.index.month==DATA_month_mean.index[i].month) & \
        (DATA.index.year==DATA_month_mean.index[i].year)] = DATA_month_mean.iloc[i]
     
    return DATA


def get_trend_all(DATA):
    for j in range(len(DATA.columns)):    
        DATA_each_column = DATA.iloc[:,j]
        DATA_month_mean = DATA_each_column.resample('M')
    #    print range(len(DATA_month_mean.index))    
        for i in range(len(DATA_month_mean.index)):
    #        print DATA_month_mean.index[i]
            DATA_each_column.loc[(DATA_each_column.index.month==DATA_month_mean.index[i].month) & \
            (DATA_each_column.index.year==DATA_month_mean.index[i].year)] = DATA_month_mean.iloc[i]
         
    return DATA



#==============================================================================
####### plot the monthly mean of sold house prices
def plot_against_date(DATA,org_data):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    
    plt.plot(DATA.resample('M'))
    
    plt.plot(org_data.iloc[:,0].resample('M'))    
    
    months = mdates.MonthLocator(bymonthday=1, interval=12)
    monthsFmt = mdates.DateFormatter("%b '%y")
    
    ax.xaxis.set_major_locator(months)
    ax.xaxis.set_major_formatter(monthsFmt)
    ax.autoscale_view()
    ax.grid(True)
    
    fig.autofmt_xdate()
#==============================================================================

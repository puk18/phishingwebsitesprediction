#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 13:21:28 2019

@author: pulkitwadhwa
"""
import pandas as pd
import numpy as np
from pprint import pprint
import math
#Import the dataset and define the feature as well as the target datasets / columns#


def training(filename):
    
   # filename = '../dataset/csv_result-Training Dataset.csv'
    cols = pd.read_csv(filename).columns
    #rows = pd.read_csv('/Users/pulkitwadhwa/Desktop/Workbook1.csv').rows
    
    dataset = pd.read_csv(filename,skiprows=1 ,
                          names=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol',
                                 'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
                                 'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
                                 'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic',
                                 'Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',])
    
    
    trainigSet=dataset.head(8843)
    count_row = dataset.shape[0]  # gives number of row count
    print("dataset count = "+ str(count_row))
    #calculating priors
    
    
    n_Result1 = trainigSet['Result'][trainigSet['Result'] == 1].count()
    n_Result0 = trainigSet['Result'][trainigSet['Result'] == -1].count()
    total_ppl = trainigSet['Result'].count()
    Prior_Result1 = n_Result1/total_ppl
    Prior_Result0 = n_Result0/total_ppl
    
    print("P(YES) = " + str(Prior_Result1))
    print("P(NO) = " + str(Prior_Result0))
    #calculating likelihood
    predictorPrior=dict()
    likelihood=dict()
    countdic=dict()
    c=0
    for i in cols[:-1]:
        uniqueList=dataset[i].unique()
        
        for j in uniqueList:
            
          
    #     print(i+':'+str(dataset[i][dataset[i] == j].count())+"for "+str(j))
            countdic[str(i)+'='+str(j)]=trainigSet[i][trainigSet[i] == j].count()
            countdic[str(i)+'='+str(j)+'/(result=1)']=trainigSet['Result'][trainigSet[i] == j][trainigSet['Result'] == 1].count()
            countdic[str(i)+'='+str(j)+'/(result=-1)']=trainigSet['Result'][trainigSet[i] == j][trainigSet['Result'] == -1].count()
            predictorPrior[str(i)+'='+str(j)]=trainigSet[i][trainigSet[i] == j].count()/count_row
    #        print(dataset['Result'][dataset[i] == j][dataset['Result'] == 1].count())
    #        print(i+'='+str(j)+'/(result=1)='+str(dataset['Result'][dataset[i] == j][dataset['Result'] == 1].count()/n_Result1))
            likelihood[i+'='+str(j)+'/(result=1)']=(trainigSet['Result'][trainigSet[i] == j][trainigSet['Result'] == 1].count()+1)/n_Result1
    #        print(dataset['Result'][dataset[i] == j][dataset['Result'] == -1].count())
    #        print(i+'='+str(j)+'/(result=-1)='+str(dataset['Result'][dataset[i] == j][dataset['Result'] == -1].count()/n_Result0))
            likelihood[i+'='+str(j)+'/(result=-1)']=(trainigSet['Result'][trainigSet[i] == j][trainigSet['Result'] == -1].count()+1)/n_Result0
    print(countdic)
    print(likelihood)
    
    
def testing():    
    for l in range(8844,11055):
    #for k in range(count_row):
        columns = list(dataset) 
        likelihood0=0
        likelihood1=0
        
            # printing the third element of the column 
        for key in likelihood.keys():
#            print("processing....")
            for i in columns[:-1]:   
        #        print(key)
        #        print(str(dataset[i][1]))
        #        print(i+'='+str(dataset[i][21])+'/(result=-1)')
                if key == i+'='+str(dataset[i][l])+'/(result=-1)':
                    likelihood0 += math.log(likelihood[key])
                elif key == i+'='+str(dataset[i][l])+'/(result=1)':
                    likelihood1 += math.log(likelihood[key])
            
                   
        print(likelihood0)
        print(likelihood1)    
        posterior1=likelihood1*Prior_Result1
        posterior0=likelihood0*Prior_Result0
        print(posterior1)
        print(posterior0)
        
        
        predicted=0
        if posterior1>posterior0:
            predicted= 1
            print('result:1')
        else:
            predicted=-1
            print('result:-1')    
        if(dataset['Result'][l]==predicted):
            c+=1
        print('Actual:'+str(dataset['Result'][l]))
    
    
    print('accuracy = ' + str((c * 100)/(11055-8844+1)))
    
    #for key in countdic.keys():   
    #    print(key+':'+str(countdic[key]))
    #print(countdic) 
    #
    #print(predictorPrior)   
    #print(likelihood)
   
filename = '../dataset/csv_result-Training Dataset.csv'
training(filename)






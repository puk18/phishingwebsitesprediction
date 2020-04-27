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
cols = pd.read_csv('dataset/csv_result-Training Dataset.csv').columns

dataset = pd.read_csv('dataset/csv_result-Training Dataset.csv',skiprows=1 ,
                      names=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol',
                             'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
                             'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
                             'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic',
                             'Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',])


dataset.head()
count_row = dataset.shape[0]  # gives number of row count
print(count_row)
##############calculating priors#########################
trainingset=dataset.head(8843)
count_row1 = trainingset.shape[0]  # gives number of row count
print(count_row1)

n_Result1 = trainingset['Result'][trainingset['Result'] == 1].count()
n_Result0 = trainingset['Result'][trainingset['Result'] == -1].count()
total_ppl = trainingset['Result'].count()
Prior_Result1 = n_Result1/total_ppl
Prior_Result0 = n_Result0/total_ppl

print(Prior_Result1)
print(Prior_Result0)
###############calculating likelihood#####################
predictorPrior=dict()
likelihood=dict()
countdic=dict()

for i in cols[:-1]:
    uniqueList=trainingset[i].unique()
    
    for j in uniqueList:
        
      
        countdic[str(i)+'='+str(j)]=trainingset[i][trainingset[i] == j].count()
        countdic[str(i)+'='+str(j)+'/(result=1)']=trainingset['Result'][trainingset[i] == j][trainingset['Result'] == 1].count()
        countdic[str(i)+'='+str(j)+'/(result=-1)']=trainingset['Result'][trainingset[i] == j][trainingset['Result'] == -1].count()
        predictorPrior[str(i)+'='+str(j)]=trainingset[i][trainingset[i] == j].count()/count_row
        likelihood[i+'='+str(j)+'/(result=1)']=(trainingset['Result'][trainingset[i] == j][trainingset['Result'] == 1].count()+1)/n_Result1
        likelihood[i+'='+str(j)+'/(result=-1)']=(trainingset['Result'][trainingset[i] == j][trainingset['Result'] == -1].count()+1)/n_Result0


#for k in range(count_row
print(countdic)        
counta=0
countw=0
for row in range(8844,11055):
    columns = list(dataset) 
    likelihood0=0
    likelihood1=0
    predicted=0

    for key in likelihood.keys():
    
        for i in columns[:-1]:   

            if key==i+'='+str(dataset[i][row])+'/(result=-1)':
                
                val=likelihood[key]
                if(val!=0):  
#                    
                    likelihood0+=math.log(likelihood[key])
#                
            elif key==i+'='+str(dataset[i][row])+'/(result=1)':
                likelihood1+=math.log(likelihood[key])
    
               
#    print(likelihood0)
#    print(likelihood1)    
    posterior1=likelihood1*Prior_Result1
    posterior0=likelihood0*Prior_Result0
#    print(posterior1)
#    print(posterior0)
    if posterior1>posterior0:
        predicted=1
    else:
        predicted=-1   
    Actual=dataset['Result'][row]
    print("----")
    print('Actual:'+str(Actual))     
    print('Predicted:'+ str(predicted))
    print("----")

    if Actual==predicted:
        counta+=1
    else:
        countw+=1
print(counta)
print(countw)
print(counta/(counta+countw))

#for key in countdic.keys():   
#    print(key+':'+str(countdic[key]))
#print(countdic) 
#
#print(predictorPrior)   
#print(likelihood)








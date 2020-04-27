#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 18:05:01 2019

@author: munishsehdev
"""
import pandas as pd
import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.01, num_iter=100000, training_Method_intercept=False, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.training_Method_intercept = training_Method_intercept
        self.verbose = verbose
        self.theta = np.zeros(X.shape[1])
    

    def intercept_Method(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def sigmoid_Method(self, z):
        return 1 / (1 + np.exp(-z))
    def loss_Method(self, h, y):
        return  (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    #(-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    #-np.average(y*np.log(h) + (1-y)*np.log(1-h))
    
    def training_Method(self, X, y):
        print("Processing.....")

        if self.training_Method_intercept:
            X = self.intercept_Method(X)
        

        # weights initialization
        self.theta = np.zeros(X.shape[1])
        
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.sigmoid_Method(z)
            gradient = np.dot(X.T, (h - y)) / y.size

            self.theta -= self.lr * gradient

            z = np.dot(X, self.theta)
            h = self.sigmoid_Method(z)
            loss = self.loss_Method(h, y)
                
            if(self.verbose ==True and i % 10000 == 0):
                print(f'loss: {loss} \t')
    
    def predict_prob(self, X):
        if self.training_Method_intercept:
            X = self.intercept_Method(X)
        
        return self.sigmoid_Method(np.dot(X, self.theta))
    
    def predict_Method(self, X):
        return self.predict_prob(X).round()
    
    

filename = 'updatedDataSet.csv'



emailData = pd.read_csv(filename,skiprows=1 ,
                      names=['having_IP_Address','URL_Length','Shortining_Service','having_At_Symbol',
                             'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State','Domain_registeration_length',
                             'Favicon','port','HTTPS_token','Request_URL','URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
                             'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe','age_of_domain','DNSRecord','web_traffic',
                             'Page_Rank','Google_Index','Links_pointing_to_page','Statistical_report','Result',])
X = emailData.drop('Result', axis=1)
y = emailData['Result']

model = LogisticRegression(lr=0.1, num_iter=2000)  #300000

X_train = X[:7735]
y_train = y[:7735]
model.training_Method(X_train, y_train)
           

preds = model.predict_Method(X)
(preds == y).mean()
model.theta


counta = 0
countw = 0


for x in range(7735,11055):
    if preds[x] == y[x]:
        counta+=1
    else:
        countw+=1

print('accuracy = '+str((counta/(counta+countw))*100))
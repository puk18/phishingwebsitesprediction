#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 18:07:06 2019

@author: munishsehdev
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'updatedDataSet.csv'

emailData = pd.read_csv(filename)
X = emailData.drop('Result', axis=1)
y = emailData['Result']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30,shuffle=False)

from sklearn.svm import SVC
svclassifier = SVC(kernel='poly', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

from sklearn.model_selection import cross_val_score
clf = SVC(kernel='linear', C=1)
scores = cross_val_score(clf, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.naive_bayes import BernoulliNB
clf1 = BernoulliNB()
clf1.fit(X_train, y_train)
y_prednb = svclassifier.predict(X_test)
#from sklearn.metrics import classification_report, confusion_matrix
print("--Naive Bayes--")
print(confusion_matrix(y_test,y_prednb))
print(classification_report(y_test,y_prednb))
scores1 = cross_val_score(clf1, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))


from sklearn import tree
clf2 = tree.DecisionTreeClassifier()
clf2.fit(X_train, y_train)

y_preddt = clf2.predict(X_test)
#from sklearn.metrics import classification_report, confusion_matrix
print("--DecisionTreeClassifier--")
print(confusion_matrix(y_test,y_preddt))
print(classification_report(y_test,y_preddt))
scores2 = cross_val_score(clf2, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(max_depth=11, random_state=0)
clf3.fit(X_train, y_train)

y_preddtrdt = clf3.predict(X_test)
#from sklearn.metrics import classification_report, confusion_matrix
print("Random forest Classifier")
print(confusion_matrix(y_test,y_preddtrdt))
print(classification_report(y_test,y_preddtrdt))
scores3 = cross_val_score(clf3, X, y, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores3.mean(), scores3.std() * 2))


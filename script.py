#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 15:25:26 2019

@author: munishsehdev
"""


def dt():
    print("Decision Tree")
def nb():
    file = 'Naive-Bayes/Naive.py'
    exec(file)
    print("Naïve Bayes")
def svm():
    print("Support Vector Machine")
def default():
    print("Incorrect Selection")

switcher = {
    1: dt,
    2: nb,
    3: svm,
    }

def switch(selectProject):
    return switcher.get(selectProject, default)()



def main():
    print("Welcome to Data Mining Algorithms Module")
    print("1. Decision Tree")
    print("2. Naïve Bayes")
    print("3. Support Vector Machine")
    
    selectedAlgo = int(input("Choose Alogrithm : "))
    switch(selectedAlgo)
    
main()
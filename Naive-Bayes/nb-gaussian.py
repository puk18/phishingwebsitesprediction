#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import math
import random


# In[15]:


def loadCsv(filename):
    print(filename)
    lines = csv.reader(open(filename))
    dataset = list(lines)
    for i in range(len(dataset)):
        dataset[i] = [float(x) for x in dataset[i]]
    return dataset


# In[3]:


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet,copy]


# In[4]:


def seprateByClass(dataset):
    seprated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if(vector[-1] not in seprated):
            seprated[vector[-1]] = []
        seprated[vector[-1]].appemd(vector)
    return seprated


# In[5]:


def mean(numbers):
    return sum(numbers)/float(len(numbers))


# In[6]:


def stdev(nums):
    avg = mean(nums)
    variance = sum([pow(x-avg,2) for x in nums])/float(len(nums) - 1)
    return math.sqrt(variance)


# In[7]:


def summarize(dataset):
    summaries = [(mean(attribute), stdec(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


# In[8]:


def summarizeByClass(dataset):
    seprated = separateByClass(dataset)
    summaries = {}
    for classValue, instansecs in seprated.items():
        summaries[classValue] = summarize(instances)
    return summaries
    


# In[9]:


def calcultaeProbabilty(x, mean, stdev):
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return(1/(math.sqrt(2*math.pi)*stdev))*exponent


# In[10]:


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
        return(correct/float(len(testSet))) * 100


# In[11]:


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


# In[12]:


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probabilty in probabilities.items():
        if bestLabel is None or probabilty > bestProb:
            bestProb = probabilty
            bestLabel = classValue
    return bestLabel


# In[13]:


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculate(x, mean, stdev)
        return probabilities


# In[17]:


def main():
    filename = 'dataset/csv_result-Training Dataset.csv'
    splitRatio = 0.67
    dataSet = loadCsv(filename)
    trainingSet, testSet = splitDataset(dataset, splitRatio)
    print('split {0} rows into train = {1} and test = {2} rows'.format(len(dataset),len(trainingSet),len(testSet)))
    
    #prepare model
    summaries = summarizeByClass(trainingSet)
    
    #test model
    predictions = getPredictions(summaries, testSet)
    accuracy = getAccuracy(testSet, predictions)
    print('Accuracy : {0}%' .format(accuracy))
    
main()


# In[ ]:





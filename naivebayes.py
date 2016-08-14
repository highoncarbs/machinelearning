#!/usr/bin/python
# -*- coding: utf-8 -*-
#handling the data
import csv
import os,sys

def loadCsv(filename):
	lines = csv.reader(open(filename,"rb"))
	dataset = list(lines)
	for i in range(len(dataset)):
		dataset[i] = [float(x) for x in dataset[i]]
	return dataset



#splitting data fro training and data analysis

import random
def splitDataset(dataset , splitRatio):
	trainsize = int(len(dataset)*splitRatio)
	trainset = []
	copy = list(dataset)
	while len(trainset)<trainsize:
		index = random.randrange(len(copy))
		trainset.append(copy.pop(index))
	return [trainset,copy]
# Separate the data set by class :: which is y = 0 / 1 

def separateByClass(dataset):
	separated = {}
	for i in range(len(dataset)):
		vector = dataset[i]
		if(vector[-1] not in separated):
			separated[vector[-1]] = []
		separated[vector[-1]].append(vector)
	return separated

#calculate mean for each of the attribute values of given data

import math

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([pow(x-avg , 2) for x in numbers])/float(len(numbers))
	return math.sqrt(variance)

#Summarizing the data

def summarize(dataset):
	summaries = [(mean(attribute) , stdev(attribute))for attribute in zip(*dataset)]
	del summaries[-1]
	return summaries

#summarize data by class

def summarizeByClass(dataset):
	separate  = separateByClass(dataset)
	summaries = {}
	for classValue , instances in separate.iteritems():
		summaries[classValue] = summarize(instances)
	return summaries

#So our fuctions are pretty much ready , now building the Prediction functions --->

def calculateProbability(x,mean,stdev):
	exponent = math.exp(-(math.pow(x-mean,2) / 2*math.pow(stdev,2)))
	return (1/(math.sqrt(2*math.pi)*stdev))*exponent

# Now calculating the probabilty og an attribute in Class suing the above function 

def calculateClassProbabilities(summaries , inputVector):
	probabilities = {}
	for classValue , classSummaries in summaries.iteritems():
		probabilities[classValue] = 1
		for i in range(len(classSummaries)):
			mean , stdev = classSummaries[i]
			x = inputVector[i]
			probabilities[classValue] *= calculateProbability(x,mean,stdev)
	return probabilities

#now we need to predict 

def predict(summaries, inputVector):
	probabilities = calculateClassProbabilities(summaries, inputVector)
	bestLabel, bestProb = None, -1
	for classValue, probability in probabilities.iteritems():
		if bestLabel is None or probability > bestProb:
			bestProb = probability
			bestLabel = classValue
	return bestLabel

#Return Pridictions for a dataset

def getPredictions(summaries, testSet):
	predictions = []
	for i in range(len(testSet)):
		result = predict(summaries, testSet[i])
		predictions.append(result)
	return predictions

#getting the Accuracy between 0 to 100%

def getAccuracy(testSet , predictions):
	correct = 0
	for x in range(len(testSet)):
		if(testSet[x][-1] == predictions[x]):
			correct += 1
		
	return (correct/float(len(testSet)))*100


def main():
	filename = "X:/macl/pima-indians-diabetes.csv"	
	splitRatio = 0.67
	dataset = loadCsv(filename)
	trainningSet , testSet = splitDataset(dataset , splitRatio)
	print('Split {0} row into Train = {1} and test = {2} rows').format(len(dataset) , len(trainningSet) , len(testSet))

	#preparing the model

	summaries = summarizeByClass(trainningSet)

	#test Model , babay . Fianlly ! 

	predictions = getPredictions(summaries , testSet)
	accuracy = getAccuracy(testSet ,predictions)
	print('Accuracy is  : {0} ').format(accuracy)
main()
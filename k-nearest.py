'''
Python Implementation of K Nearest Algorithm
from scratch
'''
#!/usr/bin/python
# -*- coding: utf-8 -*-

import math
import operator
import random

class knn:
	def euclideanDistance(one , two , length):
		distance = 0
		for x in range(length):
			distance += pow((one[x] - two[x]) , 2)
		return math.sqrt(distance)

	def getNeighbors(trainingSet , testInstance ,k):
		distances = []
		length = len(testInstance)-1
		for x in range(len(trainingSet)):
			dist = euclideanDistance(testInstance , trainingSet[x],length)
			distances.append((trainingSet[x] , dist))
		distances.sort(key=operator.itemgetter(1))
		neighbors = []
		for x in range(k):
			neighbors.append(distances[x][0])
		return neighbors

knn = knn()
trainSet = [[2, 2, 2, 'a'], [4, 4, 4, 'b']]
testInstance = [5, 5, 5]
k = 1

neighbors = knn.getNeighbors(trainSet , testInstance , 1)
print(neighbors)

# Backpropagation , wiht SGD

import random
import numpy as np 

def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_der(x):
	return sigmoid(x)*(1- sigmoid(x))

class network(object):
	def __init__(self,n):

		# n is a list , !st index  = input neurons , 2nd index = hidden layer neurons , 3rd index = output neurons
		self.num_layers = len(n) 
		self.n = n 
		self.bias = [np.random.randn(y,1) for y in n[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(n[:-1] , n[1:])]

 	def forward(self , a):
 		for b ,w in zip(self.bias , self.weights):
 			a = sigmoid(np.dot(w,a) + b)
 		return a

 #Stochastic Gradient Descent implementation
 
 	def sgd(self , train_data , num , minbatchsize , alpha ):
 		num_train_data = len(train_data)
 		#num is iterations for gradient descent
 		for i in xrange(num):
 			random.shuffle(train_data)
 			mini_batch = [train_data[k:k+minbatchsize] for k in xrange(0 , num_train_data , minbatchsize)]

 			for xbatch in mini_batch:
 				self.update_mini_batch(xbatch)


 	def update_mini_batch(self,xbatch , alpha):
 		mini_bias = [np.zeros(b.shape) for b in self.bias]
 		mini_weight = [np.zeros(w.shape) for w in self.weights]
 		for x,y in xbatch:
 			d_mini_bias , d_mini_weight = self.backprop(x,y)
 			mini_bias = [ mb+dmb for mb, dmb in zip(mini_bias, d_mini_bias)]
 			mini_weight = [ mw+dmw for mw , dmw in zip(mini_weight , d_mini_weight)]
 		self.weights = [ w-(alpha/len(xbatch))*mw for w, mw in zip(self.weights , mini_weight)
 		self.bias = [ b-(alpha/len(xbatch))*mb for b, mb in zip(self.bias , mini_bias)

# Backprop Implemetation :
"""
	1. input x
	2. obtain z := aw +b
	3. calculate the error : say delta = (del)COST_FN.sigma_derv(z)
		also  (del)COST_FN = (a:in  - delta:out)
	4. backpropagate the error by : for each layer
		del(l) = (w(l+1))'*delta(l+1) . sigma_derv(z(l))
	5. gradient : del C / del w = a(l-1)*delta(l) , del C / del b = delta(l)

""" 			
	def backprop(self,x,y):
		act = x
		actlist = [x]
		zlist = [] #z vectors 
		# feeds the actual data
		for b ,w in zip(self,bias , self.weights):
			z = np.dot(w,act)+b
			zlist.append(z)
			act = sigmoid(z)
			actlist.append(act)

		# error checking

		delta = self.cost_derv(actlist[-1] , y)
		mini_bias = delta 
		mini_weight = np.dot(delta,actlist[-2].transpose())

		for l in xrange(2, self.num_layers):
			z = zlist[-1]
			sd = sigmoid_der(z)
			delta = np.dot(self.weights[-l+1].transpose(),delta) * sp
			mini_bias[-1] = delta
			mini_weight = np.dot(delta , actlist[-1].transpose())
		return (mini_bias , mini_weight)

	def cost_derv(self , output_act , y):
		return (output_act - y)

	def evaluate(self , test_data):
		"""
		implement feed forward network ** !!
		
		"""
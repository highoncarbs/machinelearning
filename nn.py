#!usr/bin/python2.7

#generating moon dataset 
import numpy as np 
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt 

np.random.seed(0)
X, y = make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

layers_dim = [2,3,2]
mlnn =Model(layers_dim)
mlnn.train(X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=True)

# Plot the decision boundary
plot_decision_boundary(lambda x: model.predict(x), X, y)
plt.title("Decision Boundary for hidden layer size 3")
plt.show()

class MulGate:
	def forward(self,W,X):
		return np.dot(X,W)

	def backward(self,W,X,dZ):
		dW = np.dot(np.transpose(X) , dZ)
		dX = np.dot(dZ , np.transpose(W))
		return dW , dX

class AddGate:
	def forward(self , X ,b):
		return X+b 
	def backward(self, X ,b,dZ):
		dX = dZ*np.ones_like(X)
		db = np.dot(np.ones((1,dZ.shape[0]) , dtype = np.float64) , dZ)
		return db , dX

#layer tanH and sigmoid function

class Sigmoid:
	def forward(self , X ):
		return 1.0/(1.0 + np.exp(-X))

	def backward(self , X, err_diff):
		output = self.forward(X)
		return (1.0-output)*output*err_diff

class tanh:
	def forward(self ,X):
		return np.tanh(X)
	def backward(self , X ,err_diff):
		output = self.forward(X)
		return (1.0 - np.square(output))*err_diff

#Network output

class Softmax:
	def predict(self,X):
		exp_scores = np.exp(X)		
		return exp_scores/np.sum(exp_scores , axis=1 , keepdims=True)

	def loss(self , X , y):
		num_examples = X.shape[0]
		probs = self.predict(X)

		correct_logprobs = -np.log(probs[range(num_examples)] , y)
		data_loss = np.sum(correct_logprobs)
		return 1./num_examples*data_loss

	def diff(self,X,y):
		num_examples = X.shape[0]
		probs = self.predict(X)
		probs[range(num_examples) , y] -= 1
		return probs

class Model:
	def __init__(self , layers_dim):
		self.b = []
		self.W = []
		for i in range(len(layers_dim)-1):
			self.W.append(np.random.randn(layers_dim[i] , layers_dim[i+1]) / np.sqrt(layers_dim[i]))
			self.b.append(np.random.randn(layers_dim[i+1]).reshape(1,layers_dim[i+1]))

	def cal_loss(self , X ,y):
		mul = MulGate()
		add = AddGate()
		layer = tanh()
		softOut = Softmax()		

		input = X
		for i in range(len(self.W)):
			mul = MulGate.forward(self.W[i] , input)
			add = AddGate.forward(mul , self.b[i])
			input = layer.forward(add)

		return softOut.loss(input , y)

	def predict(self , X):
		mul = MulGate()
		add = AddGate()
		layer = tanh()
		softOut = Softmax()		

		input = X
		for i in range(len(self.W)):
			mul = MulGate.forward(self.W[i] , input)
			add = AddGate.forward(mul , self.b[i])
			input = layer.forward(add)

		probs = softOut.predict(input)
		return np.argmax(probs , axis = 1)		

	def train(self, X, y, num_passes=20000, epsilon=0.01, reg_lambda=0.01, print_loss=False):
	    mulGate = MulGate()
	    addGate = AddGate()
	    layer = tanh()
	    softmaxOutput = Softmax()

	    for epoch in range(num_passes):
	        # Forward propagation
	        input = X
	        forward = [(None, None, input)]
	        for i in range(len(self.W)):
	            mul = mulGate.forward(self.W[i], input)
	            add = addGate.forward(mul, self.b[i])
	            input = layer.forward(add)
	            forward.append((mul, add, input))

	        # Back propagation
	    dtanh = softmaxOutput.diff(forward[len(forward)-1][2], y)
        for i in range(len(forward)-1, 0, -1):
            dadd = layer.backward(forward[i][1], dtanh)
            db, dmul = addGate.backward(forward[i][0], self.b[i-1], dadd)
            dW, dtanh = mulGate.backward(self.W[i-1], forward[i-1][2], dmul)
            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW += reg_lambda * self.W[i-1]
            # Gradient descent parameter update
            self.b[i-1] += -epsilon * db
            self.W[i-1] += -epsilon * dW

        if print_loss and epoch % 1000 == 0:
            print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))


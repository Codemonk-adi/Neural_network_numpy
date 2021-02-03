import idx2numpy as idx
import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sp

np.random.seed(0)
img = r"C:\Users\Aaditya\OneDrive\Documents\ML\train-image"
lbl = r'C:\Users\Aaditya\OneDrive\Documents\ML\train-labels-idx1-ubyte'
t_lbl = r'C:\Users\Aaditya\OneDrive\Documents\ML\t10k-labels.idx1-ubyte'
t_img = r'C:\Users\Aaditya\OneDrive\Documents\ML\t10k-images.idx3-ubyte'
image = idx.convert_from_file(img)
iput = np.reshape(image, (60000,784))/255
otput = np.eye(10)[idx.convert_from_file(lbl)]
test_image = idx.convert_from_file(t_img)
test_input = np.reshape(test_image, (10000,784))/255
test_output = idx.convert_from_file(t_lbl)
# input = np.array([[0,1,1],[1,0,1],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0],[0,1,1],[1,0,1],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0]])
# output = np.array([[0,1],[0,0],[0,0],[0,0],[0,0],[1,1],[0,0],[0,1],[0,0],[0,0],[0,0],[0,0],[1,1],[0,0]])

def sigmoid(x):
    # return 2/(1+np.exp(-2*x))-1
    sigmoid = sp.expit(x)
    return sigmoid
    # return np.where(x>0,x,0)
    # return x
def tanh(x):
    return np.tanh(x)
def relu(x):
    x[x<0] = 0
    return x

def reluprime(x):
    return (x>0).astype("float")

def sigmoid_prime(x):
    # return (1-sigmoid(x)**2)
    return sigmoid(x)*(1-sigmoid(x))
    # return (x>0).astype(x.dtype)    
    # return 1
def tanh_prime(x):
    return 1 - tanh(x)**2
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons,activation="sigmoid",keep_prob=1):
        self.n_neurons=n_neurons
        if activation == "sigmoid":
            self.activation = sigmoid
            self.a_prime = sigmoid_prime
        elif activation == "tanh":
            self.activation = tanh
            self.a_prime = tanh_prime
        else :
            self.activation = relu
            self.a_prime = reluprime
        self.keep_prob = keep_prob
        # weight = (# of inputs,# of outputs)
        self.weights = np.random.randn(n_inputs ,n_neurons)*0.01
        self.biases = np.random.randn(1,n_neurons)*0.01 
        # self.weights = 0.1*np.ones((n_inputs ,n_neurons))
        # self.biases = 0.1*np.ones((1,n_neurons))
    def cal_output(self,input,train=False):        
        output = np.dot(input,self.weights) + self.biases
        
        if train == True:
            D = np.random.rand(1,self.n_neurons)
            self.D = (D<self.keep_prob).astype(int)
            output = np.multiply(output , self.D)  
        return output
    def forward(self,input):
        return self.activation(self.cal_output(input))
    def back_propagate(self,delta,ap,lr=1,keep_prob=1):
        dz =  delta
        self.weights -= np.multiply(np.dot(ap.T,dz),self.D) * (0.001*lr)
        self.biases -= (np.sum(dz,axis=0,keepdims=True)*self.D) * (0.001*lr)
        return np.multiply(np.dot(dz,self.weights.T),(1-ap**2))
        

class Neural_Network:
    def __init__(self,input,output):
        self.input=input
        self.output=output
        self.layers = []
    def Add_layer(self,n_neurons,activation="relu",keepprob=1):
        if len(self.layers) != 0:    
            newL = Layer_Dense(self.layers[-1].n_neurons,n_neurons,activation,keep_prob=keepprob)
        else:
            newL = Layer_Dense(self.input.shape[1],n_neurons,activation,keep_prob=keepprob)
        self.layers.append(newL)
    def predict(self,input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def cal_zs(self,input):
        self.activations = []
        self.activations.append(input)
        output = input
        for layer in self.layers:
            z = layer.cal_output(output,train=True)
            activation = layer.activation(z)
            self.activations.append(activation)
            output = activation
    def train(self,input=None,output=None,lr=10):
        if input is None:
            input=self.input
            output=self.output
            
        if len(input)>2000:
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)
            input = input[indices]
            output = output[indices]
            for _ in range(10):
                self.lr = lr
                for i in range(int(len(input)/64)):
                    self.lr *=0.99
                    self.train(input[i*64:i*64+64],output[i*64:i*64+64],self.lr)
            return
        self.cal_zs(input)
        for i in range(1,len(self.layers)+1):
            if i==1:
                delta = self.activations[-1] - output
                self.delta = self.layers[-1].back_propagate(delta,self.activations[-2],lr)
            else:
                self.delta = self.layers[-i].back_propagate(self.delta,self.activations[-i-1],lr)
    def MSE(self):
        predict = self.predict(self.input)
        error = (predict - self.output)**2
        mse = sum(sum(error))
        print(mse)
    def Logloss(self):
        predict = self.predict(self.input)
        error = np.multiply(self.output,np.log(predict)) + np.multiply(1-self.output,np.log(1-predict))
        logloss = -1*sum(sum(error))
        print(logloss)
    def accuracy(self):
        predict = self.predict(test_input)
        prediction = np.argmax(predict,axis=1)
        correct = np.mean(prediction == test_output)
        print(correct*100)
            
    # def train(self,input,output):
        
model = Neural_Network(iput,otput)
# model.Add_layer(4)
model.Add_layer(64)
model.Add_layer(16)
model.Add_layer(10,"sigmoid")
lrc= 6
for _ in range(10):
    model.accuracy()
    model.train(lr=lrc)
model.accuracy()

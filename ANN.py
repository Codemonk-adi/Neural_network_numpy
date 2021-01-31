import numpy as np 
import idx2numpy as idx
import matplotlib.pyplot as plt

np.random.seed(0)
img = r".\train-image.idx3-ubyte"
lbl = r'.\train-labels-idx1-ubyte'
t_lbl = r'.\t10k-labels.idx1-ubyte'
t_img = r'.\t10k-images.idx3-ubyte'
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
    return 1/(1+ np.exp(-x))
    # return np.where(x>0,x,0)
    # return x
def tanh(x):
    return np.tanh(x)
def relu(x):
    return np.where(x>0,x,0)

def reluprime(x):
    return (x>0).astype(x.dtype)

def sigmoid_prime(x):
    # return (1-sigmoid(x)**2)
    return sigmoid(x)*(1-sigmoid(x))
    # return (x>0).astype(x.dtype)    
    # return 1
def tanh_prime(x):
    return 1 - tanh(x)**2
class Layer_Dense:
    def __init__(self,n_inputs,n_neurons,activation="sigmoid"):
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
        # weight = (# of inputs,# of outputs)
        self.weights = np.random.randn(n_inputs ,n_neurons)
        self.biases = np.random.randn(1,n_neurons)
        # self.weights = 0.1*np.ones((n_inputs ,n_neurons))
        # self.biases = 0.1*np.ones((1,n_neurons))
    def cal_output(self,input):
        output = np.dot(input,self.weights) + self.biases  
        return output
    def forward(self,input):
        return self.activation(self.cal_output(input))
    def back_propagate(self,delta,ap,lr=1):
        dz =  delta
        self.weights -= 0.001*lr*np.dot(ap.T,dz)
        self.biases -= 0.001*lr*np.sum(dz,axis=0,keepdims=True)
        return np.multiply(np.dot(dz,self.weights.T),(1-ap**2))
        

class Neural_Network:
    def __init__(self,input,output):
        self.input=input
        self.output=output
        self.layers = []
    def Add_layer(self,n_neurons,activation="tanh"):
        if len(self.layers) != 0:    
            newL = Layer_Dense(self.layers[-1].n_neurons,n_neurons,activation)
        else:
            newL = Layer_Dense(self.input.shape[1],n_neurons,activation)
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
            z = layer.cal_output(output)
            activation = layer.activation(z)
            self.activations.append(activation)
            output = activation
    def train(self,input=None,output=None,lr=10):
        if input is None:
            input=self.input
            output=self.output
            
        if len(input)>1000:
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)
            input = input[indices]
            output = output[indices]
            for _ in range(100):
                self.lr = lr
                for i in range(int(len(input)/100)):
                    self.lr *=0.99
                    self.train(input[i*100:i*100+100],output[i*100:i*100+100],self.lr)
            return
        self.cal_zs(input)
        for i in range(1,len(self.layers)+1):
            if i==1:
                delta = self.activations[-1] - output
                self.delta = self.layers[-1].back_propagate(delta,self.activations[-2],lr)
            else:
                lr*=1.5
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
for _ in range(1):
    model.accuracy()
    model.Logloss()
    model.train(lr=lrc)
model.accuracy()
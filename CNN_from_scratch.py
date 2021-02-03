import numpy as np 
import idx2numpy as idx
import copy
import matplotlib.pyplot as plt

# unable to apply backpropagation using average pooling of stride 1
# using maxpool instead
np.random.seed(0)
img = r"C:\Users\Aaditya\OneDrive\Documents\ML\train-image"
lbl = r'C:\Users\Aaditya\OneDrive\Documents\ML\train-labels-idx1-ubyte'
t_lbl = r'C:\Users\Aaditya\OneDrive\Documents\ML\t10k-labels.idx1-ubyte'
t_img = r'C:\Users\Aaditya\OneDrive\Documents\ML\t10k-images.idx3-ubyte'
iput = idx.convert_from_file(img)
otput = np.eye(10)[idx.convert_from_file(lbl)]
test_input = idx.convert_from_file(t_img)
test_output = idx.convert_from_file(t_lbl)
# input = np.array([[0,1,1],[1,0,1],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0],[0,1,1],[1,0,1],[1,0,0],[0,1,0],[0,0,1],[1,1,1],[0,0,0]])
# output = np.array([[0,1],[0,0],[0,0],[0,0],[0,0],[1,1],[0,0],[0,1],[0,0],[0,0],[0,0],[0,0],[1,1],[0,0]])

def sigmoid(x):
    return 1/(1+ np.exp(-x))
    # return x

def relu(x):
    return np.where(x>0,x,0)

def reluprime(x):
    return (x>0).astype(x.dtype)

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
    # return 1

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.n_neurons=n_neurons
        self.weights = np.random.randn(n_inputs ,n_neurons)
        self.biases = np.random.randn(1,n_neurons)
        # self.weights = 0.1*np.ones((n_inputs ,n_neurons))
        # self.biases = 0.1*np.ones((1,n_neurons))
    def cal_output(self,input):
        output = np.dot(input,self.weights) + self.biases  
        return output
    def forward(self,input):
        return sigmoid(self.cal_output(input))
    def back_propagate(self,delta,ap,sp):
        sp = sigmoid_prime(sp)
        delta_true = delta*sp
        # weights = self.weights
        self.weights -= 0.003*np.dot(ap.T,delta_true)
        self.biases -= 0.003*np.sum(delta_true,axis=0)
        
        return np.dot(self.weights,np.transpose(delta_true)).T
    def activate(self,output):
        return sigmoid(output)


class C_neuron:
    def __init__(self,n_inputs,filter_shape):
        self.filter_shape=np.array(filter_shape)
        self.n_inputs = n_inputs
        # self.weights = np.ones((n_inputs , filter_shape[0], filter_shape[1]))
        # self.biases = np.ones(1)
        self.weights = np.random.randn(n_inputs , filter_shape[0], filter_shape[1])
        self.biases = np.random.randn(1)
        
    def cal_output(self,input,output,p):
        # o_shape = list(input.shape)
        # if len(o_shape)<4:
        #     o_shape.append(1)
        #     n_shape = tuple(o_shape)
        #     input = np.reshape(input,n_shape)
        (n_input,x,y,n_filter) = np.shape(input)
        assert n_filter == self.n_inputs,"Dimension error"
        (f_x,f_y) = self.filter_shape
        for i in range(n_input):
            for curr_f in range(self.n_inputs):
                curr_y=0
                while curr_y<=y-f_y:
                    curr_x=0
                    while curr_x<=x-f_x:
                        a=input[i,curr_y:f_y+curr_y,curr_x:f_x+curr_x,curr_f]
                        b=self.weights[curr_f]
                        sum = np.sum(a*b)+self.biases
                        output[i][curr_y][curr_x][p]+=sum
                        curr_x=curr_x+1
                    curr_y=curr_y+1
        return output
                
    # def forward(self,input):
    #     return relu(self.cal_output(input))
    def back_propagate(self,delta,ap):
        (_,f_y,f_x) = delta.shape
        ap_shape = list(ap.shape)
        if len(ap_shape) < 4:
            ap_shape.append(1)
            ap = ap.reshape(ap_shape)
        
        (n_input,y,x,_) = ap.shape
        delta_prev = np.zeros(ap.shape)
        for i in range(n_input):
            self.biases -= 0.01*np.sum(delta[i])
            for curr_f in range(self.n_inputs):
                curr_y=0
                while curr_y<=y-f_y:
                    curr_x=0
                    while curr_x<=x-f_x:
                        a=ap[i,curr_y:f_y+curr_y,curr_x:f_x+curr_x,curr_f]
                        b=delta[i]
                        sum = np.sum(a*b)
                        self.weights[curr_f][curr_y][curr_x] -= 0.01*sum
                        curr_x=curr_x+1
                    curr_y=curr_y+1
        delta = np.pad(delta,((0,0),(2,2),(2,2)),'constant',constant_values=0)
        temp_fil = np.rot90(self.weights,2,(1,2))
        (f_1_y,f_1_x) = self.filter_shape
        for i in range(n_input):
            for curr_f in range(self.n_inputs):
                curr_y=0
                while curr_y<f_y+2:
                    curr_x=0
                    while curr_x<f_x+2:
                        a=delta[i,curr_y:f_1_y+curr_y,curr_x:f_1_x+curr_x]
                        b=temp_fil[curr_f]
                        sum = np.sum(a*b)
                        delta_prev[i][curr_y][curr_x][curr_f]+=sum
                        curr_x=curr_x+1
                    curr_y=curr_y+1
        return delta_prev
        
        # self.weights -= 0.003*np.dot(ap.T,delta_true)
        # self.biases -= 0.003*np.sum(delta_true,axis=0)
        
        # return np.dot(self.weights,np.transpose(delta_true)).T  
    
    
class Convolution_layer:
    def __init__(self,n_inputs,n_neurons,filter_shape,stride=1,padding=0):
        self.n_neurons=n_neurons
        self.filter_shape=filter_shape
        self.stride=stride
        self.padding=padding 
        self.neurons = []
        for _ in range(n_neurons):
            self.neurons.append(C_neuron(n_inputs,filter_shape))
        # self.weights = 0.1*np.ones((n_inputs ,n_neurons))
        # self.biases = 0.1*np.ones((1,n_neurons))
    def cal_output(self,input):
        o_shape = np.array(input.shape)
        n_shape = copy.copy(o_shape)
        n_shape[1] -= 2
        n_shape[2] -= 2        
        if o_shape.size<4:
            n_shape = np.append(n_shape, self.n_neurons)
            input = np.reshape(input,np.append(o_shape,1))
        else:
            n_shape[3] =  self.n_neurons
        
        output = np.zeros(n_shape)
        i=0
        for n in self.neurons:
            output = n.cal_output(input,output,i)
            i+=1
        return np.asarray(output)
    def forward(self,input):
        return relu(self.cal_output(input))
    def back_propagate(self,delta,ap,sp):
        sp = reluprime(sp)
        delta_true = delta*sp
        delta_prev = np.zeros(ap.shape)
        i = 0 
        for n in self.neurons:
            delta_prev = np.add(n.back_propagate(delta_true[:,:,:,i],ap),delta_prev)
            i+=1
        return delta_prev
    def activate(self,output):
        return relu(output)
            

        # self.weights -= 0.003*np.dot(ap.T,delta_true)
        # self.biases -= 0.003*np.sum(delta_true,axis=0)
        
        # return np.dot(self.weights,np.transpose(delta_true)).T

class Pool:
    def __init__(self,n_neurons):
        self.size = 2
        self.n_neurons = n_neurons
        
    def forward(self, input):
        o_shape = list(np.shape(input))
        n_shape = copy.copy(o_shape)    
        inc_x = o_shape[1] + 1 - self.size
        inc_y = o_shape[2] + 1 - self.size
        n_shape[1] = inc_x
        n_shape[2] = inc_y   
        n_shape = tuple(int(i) for i in tuple(n_shape))
        output = np.zeros(n_shape)
         
        for k in range(o_shape[0]):    
            for i in range(inc_x):
                for j in range(inc_y):
                    for l in range(o_shape[3]):
                        ma = 0
                        ma = np.max(input[k,i:i+self.size,j:j+self.size,l])
                        # sum+=input[k][2*i][2*j][l]+input[k][2*i][2*j+1][l]+input[k][2*i+1][2*j][l]+input[k][2*i+1][2*j+1][l]
                        # sum /= 4
                        output[k][i][j][l]= ma
        return output    
    def back_propagate(self,delta,ap,sp):
        o_shape = ap.shape
        delta_new = np.zeros(o_shape)
        o_shape = list(o_shape)
        inc_x = o_shape[1] + 1 - self.size
        inc_y = o_shape[2] + 1 - self.size
        for k in range(o_shape[0]):    
            for i in range(inc_x):
                for j in range(inc_y):
                    for l in range(o_shape[3]):
                        ma = sp[k][i][j][l]
                        for p in range(self.size):
                            for q in range(self.size):
                                if ap[k][i+p][j+q][l] == ma:
                                    delta_new[k][i+p][j+q][l] += delta[k][i][j][l]
        
        return delta_new
                        # sum+=input[k][2*i][2*j][l]+input[k][2*i][2*j+1][l]+input[k][2*i+1][2*j][l]+input[k][2*i+1][2*j+1][l]
                        # sum /= 4
                        
        
        
        # weights = self.weights
        
    def cal_output(self,input):
        return self.forward(input)
    # since no updation of parameters is needed 
    def activate(self,output):
        return output
class flatten:
    def __init__(self,n_neurons):
        self.n_neurons  = n_neurons
    def forward(self, input):
        self.o_shape = np.array(input.shape)
        n_shape = tuple([self.o_shape[0],self.n_neurons])
        return input.reshape(n_shape)
    def cal_output(self,input):
        return self.forward(input)
    def activate(self,output):
        return output
    # since no updation of parameters is needed
    def back_propagate(self,delta,ap,sp):
        return delta.reshape(self.o_shape)
class Neural_Network:
    def __init__(self,input,output):
        self.input=input
        self.curr_shape = np.array(input.shape, dtype='int')
        self.curr_shape = self.curr_shape[1:]
        self.curr_shape = np.append(self.curr_shape,1)
        self.output=output
        self.layers = []
    def Add_Dlayer(self,n_neurons):
        if len(self.layers) != 0:    
            newL = Layer_Dense(self.layers[-1].n_neurons,n_neurons)
        else:
            newL = Layer_Dense(self.input.shape[1],n_neurons)
        self.layers.append(newL)
    def Add_Clayer(self,n_neurons):
        if len(self.layers) != 0:    
            newL = Convolution_layer(self.layers[-1].n_neurons,n_neurons,(3,3))
        else:
            newL = Convolution_layer(1,n_neurons,(3,3))
        self.curr_shape[0] -= 2
        self.curr_shape[1] -= 2
        self.curr_shape[2] = n_neurons
        self.layers.append(newL)
    def Add_Pool_layer(self):
        newL = Pool(self.layers[-1].n_neurons)
        self.curr_shape[0] -= 1
        self.curr_shape[1] -= 1
        self.layers.append(newL)
    def Add_Flayer(self):
        newL = flatten(np.prod(self.curr_shape))
        self.layers.append(newL)
    def predict(self,input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    def cal_zs(self,input):
        self.zs=[]
        self.activations = []
        self.activations.append(input)
        output = input
        for layer in self.layers:
            z = layer.cal_output(output)
            self.zs.append(z)
            activation = layer.activate(z)
            self.activations.append(activation)
            output = activation
    def train(self,input=None,output=None):
        if input is None:
            input=self.input
            output=self.output
        if len(input.shape) < 4:
            a =list(input.shape)
            a.append(1)
            input =input.reshape(a)
        if len(input)>1000:
            indices = np.arange(input.shape[0])
            np.random.shuffle(indices)
            input = input[indices]
            output = output[indices]
            for i in range(int(len(input)/10)):
                self.train(input[i*10:i*10+10],output[i*10:i*10+10])
            return
        self.cal_zs(input)
        for i in range(1,len(self.layers)+1):
            if i==1:
                delta = 2*(self.activations[-1]-output)
                self.delta = self.layers[-1].back_propagate(delta,self.activations[-2],self.zs[-1])
            else:
                self.delta = self.layers[-i].back_propagate(self.delta,self.activations[-i-1],self.zs[-i])
    def MSE(self):
        predict = self.predict(self.input)
        error = (predict - self.output)**2
        mse = sum(sum(error))
        print(mse)
    # def train(self,input,output):
        

input = np.array([[[1,2,3,4],[0,1,2,3],[2,3,4,5],[3,4,5,6]],[[1,2,3,4],[0,1,2,3],[2,3,4,5],[3,4,5,6]]])  
o = np.array([[1,0],[1,0]])
model = Neural_Network(iput,otput)
# model.Add_layer(4)
model.Add_Clayer(2)
model.Add_Pool_layer()
model.Add_Clayer(4)
model.Add_Pool_layer()
model.Add_Flayer()
model.Add_Dlayer(64)
model.Add_Dlayer(10)

predict = model.predict(iput[0:10])
print(predict)
model.train()
predict = model.predict(iput[0:10])
print(predict)
prediction = np.argmax(predict,axis=1)

# for i,j in zip(prediction,test_output):
#     if(i==j):
#         correct += 1 
# print(correct,"/",len(test_output))
# for _ in range(10):
#     model.train()
# predict = model.predict(test_input)
# prediction = np.argmax(predict,axis=1)
# correct = 0
# for i,j in zip(prediction,test_output):
#     if(i==j):
#         correct += 1 
# print(correct,"/",len(test_output))
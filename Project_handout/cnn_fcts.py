#Importations
import torch
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
import pandas as pd
import time
import h5py


from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
from matplotlib import gridspec
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

from __future__ import print_function
import numpy as np
import h5py
import tensorflow as tf
from keras.models import model_from_json, load_model
from keras.optimizers import SGD, Adam, RMSprop,Adagrad
import scipy.io
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


### Usefull variables
nb_freq = 256
nb_time = 360
nb_mics = 4
number_of_rooms = 1036

# Treating the Training Labels.

def prep_targets(index,label):
    target = torch.tensor(label)
    target = target.expand(nb_time,1)
    return target

#Function that applies the same shuffle to the data and its corresponding labels.    

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p] 

# Architecture 1.0 : vanilla CNN for DOA 

# In this part, we will focus on building a vanilla CNN in order to recognize the directions of arrival of the 
# sound in a specific room. Once done, we will then compare "by hand" the labelel dataset to the predicted values. This 
# is the most basic setup and will try to improve latter. 


# Weight sharing model with nb_time channels
class ws_Net(nn.Module):
    def __init__(self,nb_hidden = 50, n = 390 ):
        super(ws_Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=2)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=2)
        self.conv3 = nn.Conv2d(8 ,8, kernel_size=2)
        self.fc1 = nn.Linear(1008, nb_hidden)# (1x1008) being the dim of the censor obtained by flattening the output of the 3rd CL.
        self.fc2 = nn.Linear(nb_hidden*n,512)
        self.fc3 = nn.Linear(512,2)
        
    
    def forward(self, x, n):
        
        #print("What actually enters the model:",x.shape)
        test =  x[:,:,:,:,0].view(-1,1,4,129)
        #print("Shape of test:",test.shape)
        output = torch.zeros([1008,1])
        
        for i in range(n):
            y = x[:,:,:,:,i].view(-1,1,4,129)
            y = F.relu(self.conv1(y)) 
            y = F.relu(self.conv2(y))
            y = F.relu(self.conv3(y))
            y = F.relu(self.fc1(y.view(-1, 1008)))
            
            if (i==0):
                output = y
            else:
                output = torch.cat((output,y),1)
                
        #print("Shape of the out of fc1:",output.shape)        
                
        output = F.relu(self.fc2(output))
        
        #print("Shape of the out of fc2:",output.shape) 

        output = F.softmax(self.fc3(output))   
        
        return output    

# CNN model proposed in the paper.
class SimpleModel(nn.Module):
    
    def __init__(self, nb_hidden=512,phase_map=64):
        super(SimpleModel, self).__init__()
        self.cl1 = nn.Conv2d(1, phase_map, kernel_size=2)
        self.cl2 = nn.Conv2d(phase_map, phase_map, kernel_size=2)
        self.cl3 = nn.Conv2d(phase_map, phase_map, kernel_size=2)
        self.dropout = nn.Dropout(0.02)
        self.fc1 = nn.Linear((nb_freq-3)*phase_map,nb_hidden)
        self.dropout = nn.Dropout(0.02)
        self.fc2 = nn.Linear(nb_hidden,nb_hidden)
        self.fc3 = nn.Linear(nb_hidden,37)
 
 
    def forward(self, x):
       
        x = F.relu(self.cl1(x))
        x = F.relu(self.cl2(x))
        x = F.relu(self.cl3(x))
        x = self.dropout(x)
        x = F.relu(self.fc1(x.view(-1, (nb_freq-3)*64)))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Preparing the Data

def prep_input_vanilla(train_input):
    new_train_input = train_input.view(-1,1,nb_mics,nb_freq)
    return new_train_input


def prep_test_input(train_input):
    new_train_input = train_input.view(-1,1,nb_freq,nb_mics)
    return new_train_input

def prep_labels_vanilla(train_input):
    new_train_input = train_input.view(-1)
    return new_train_input




#Training function for the weights sharing model.

def train_model_3(model, optimizer1, nb_epochs, train_input, train_target ,mini_batch_size,n):

    start = time.time()
    for e in range(0,nb_epochs):
        start_ep = time.time()
        for b in range(0, train_input.size(0), mini_batch_size):
            start = time.time()
            #print("Shape of the input of the model:",train_input.narrow(0, b, mini_batch_size).shape)
            output = model(train_input.narrow(0, b, mini_batch_size),n)
            target = train_target.narrow(0, b, mini_batch_size)
            _,indices = output.max(1)
            #print("batch :",b," Output :",output,"Targets:", target)
            loss = criterion(output,target.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()   
        if (e == 0):
            end_ep = time.time()
            print("Time for 1 epochs is :{:5}".format(-(start_ep-end_ep)))
            
        print("Loss for epoch{:3} is {:5} ".format(e,loss.data.item()))
            
    end = time.time()
    print("Time the hole training is :{:5}".format(-(start-end)))

#Training function for the main model described in the paper.

def train_simple_model(model, optimizer, nb_epochs, train_input, train_target ,mini_batch_size):

    start = time.time()
    for e in range(0,nb_epochs):
        start_ep = time.time()
        train_input,train_target = unison_shuffled_copies(train_input,train_target)
        for b in range(0, train_input.size(0), mini_batch_size):
            start = time.time()
            #print("Shape of the input of the model:",train_input.narrow(0, b, mini_batch_size).shape)
            output = model((train_input.narrow(0, b, mini_batch_size)).float())
            target = train_target.narrow(0, b, mini_batch_size)
            _,indices = output.max(1)
            #print("batch :",b," Output :",indices,"Targets:", target)
            print(output.shape,target.shape)
            loss = criterion(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (e == 0):
            end_ep = time.time()
            print("Time for 1 epochs is :{:5}".format(-(start_ep-end_ep)))
            
        print("Loss for epoch{:3} is {:5} ".format(e,loss.data.item()))
            
    end = time.time()
    print("Time the hole training is :{:5}".format(-(start-end)))

#Prediction functions.

def predict(data):
    pred = loaded_model.predict(data)   
    pred = np.argmax(pred, axis=1)
    return pred

def get_labels(labels):
    labels = np.argmax(labels, axis=1)
    return labels








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

import numpy as np
import h5py
import tensorflow as tf
from keras.models import model_from_json, load_model
from keras.optimizers import SGD, Adam, RMSprop,Adagrad
import scipy.io
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import cnn_fcts

# The Data
Dataset = pd.read_pickle("/home/janjar/Dataset/Trainingset/Training_dataframe.pkl")

# Usefull variables
nb_freq = 256
nb_time = 360
nb_mics = 4
number_of_rooms = Dataset.shape[0]

## Treating the Training Data.

### Loading the phase matrices in tensors with respect to the indexing shown in the dataframe.
data_matrix = np.zeros([number_of_rooms,nb_mics,nb_freq,nb_time])
for i in range(number_of_rooms):
    fileName_matrix = Dataset.iloc[i]['Phase_Matrix']
    #print(fileName_matrix)
    fileObject2 = open(fileName_matrix, 'rb')
    matrix_loaded = pkl.load(fileObject2)
    fileObject2.close()
    data_matrix[i] = matrix_loaded
data_matrix.shape    

data_matrix = torch.from_numpy(data_matrix)
data_matrix = data_matrix.view(-1,nb_mics,nb_freq,nb_time)
data_matrix = data_matrix.type('torch.FloatTensor')
data_matrix.shape

final_data = torch.ones([number_of_rooms*nb_time,nb_mics,nb_freq], dtype=torch.float64)

i = 0
for setup in range(number_of_rooms):
     for nb_tf in range(nb_time):
        final_data[i] = data_matrix[setup,:,:,nb_tf]
        i = i+1   



## Treating the Training Labels.

data_targets = torch.zeros([number_of_rooms], dtype=torch.float64)
for i in range(number_of_rooms):
    data_targets[i] = Dataset.iloc[i]['Label']
data_targets.shape 

## Flatenning the timeframe dimension of the input.
final_targets = torch.ones([number_of_rooms*nb_time], dtype=torch.float64)
test = []
for i in range(number_of_rooms):
    curr = [data_targets[i].item()]*nb_time
    test.append(curr)
final_targets = torch.FloatTensor(test)
final_targets = final_targets.flatten()

#Schuffling the training data.
final_data,final_targets = unison_shuffled_copies(final_data,final_targets)

train_data = prep_input_vanilla(final_data)
train_data = train_data.float()

train_targets = prep_labels_vanilla(final_targets)
train_targets = train_targets.to(dtype=torch.int64)
train_targets2 =  prep_labels_vanilla(data_targets)
train_data, train_targets, train_data2, train_targets2 = Variable(train_data), Variable(train_targets), Variable(train_data2),Variable(train_targets2)


# Spliting the Training Data into Training/Validation.
training_data = train_data[0:365000]
training_targets = train_targets[0:365000]
validation_data = train_data[365000:]
validation_targets = train_targets[365000:]

# 1.2 Training the model

model2 = (SimpleModel()).float()

#Defining the training parameters.

mini_batch_size = 20
nb_epochs = 100
eta = 0.001 #learning rate
criterion = torch.nn.CrossEntropyLoss() #Cross Entropy
optimizer = torch.optim.Adam(model2.parameters(), lr = eta, weight_decay=0.001) #ADAM
optim_no_wd = torch.optim.Adam(model2.parameters(), lr = eta)

#Training the model.
train_simple_model(model2,optimizer,nb_epochs,training_data,training_targets,mini_batch_size)
















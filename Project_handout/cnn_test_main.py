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

show_plots = False


# Doing the prediction with the trained model provided by the paper.

# Treating the Test Data.
Test_data = h5py.File('DOA_test.hdf5')
X_test = Test_data['features']                   # These are the phase maps for each time frame
Y_test = Test_data['targets']
X_test = np.array(X_test)                        # size = (Number of time frames,1,256,4)
Y_test = np.array(Y_test)
X_test,Y_test = unison_shuffled_copies(X_test,Y_test)
Xtest = torch.from_numpy(X_test)
Ytest = torch.from_numpy(Y_test)

# Load trained model from the json file

json_file = open('Model_sin_CNN.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
print(type(loaded_model))
# Load weights 

loaded_model.load_weights('Weights_sin_CNN.h5')
print("Loaded model from disk")

# Define the optimizer and compile the model

lrate = 0.001
adam = Adam(lr =lrate,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
loaded_model.compile(loss='categorical_crossentropy', optimizer= adam)

# Doing the predictions on the test set.
pred = predict(X_test)
# Turning the labels for Hot-one to indexes.
labels = get_labels(Y_test)

#Calculation some metrics on the testing set.
report = classification_report(labels, pred, digits=4)
accuracy = accuracy_score(labels, pred)
print("The accuracy score the model on the test set is:",accuracy)


### Showing the confusion matrix of the classification.

cm = metrics.confusion_matrix(labels, pred)

if(show_plots==True):
	print(report)
	plt.figure(figsize=(14,14))
	sns.heatmap(cm, annot=True, fmt="d", linewidths=1.5, square = True, cmap = 'Blues_r');
	plt.ylabel('Actual label');
	plt.xlabel('Predicted label');
	all_sample_title = 'Accuracy Score: {0}'.format(accuracy)
	plt.title(all_sample_title, size = 15);


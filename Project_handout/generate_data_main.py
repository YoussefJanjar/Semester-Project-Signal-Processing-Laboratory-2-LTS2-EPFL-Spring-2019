#Importations.
import numpy as np
import matplotlib.pyplot as plt
import IPython
import pyroomacoustics as pra
import pickle as pkl
import pandas as pd
from scipy.io import wavfile
from scipy.signal import fftconvolve
from scipy import signal
from matplotlib import gridspec
import random
import generate_data_fcts

test_fcts = False

### Setting the different parameters for the rooms

#Here You should put the path to where the file wav is on the machine you are working on.
folder_path = "/home/janjar/Semester_Project/Dataset/GDWN/WGN.wav"
file_to_open = folder_path 

ang = (np.array([61.]) / 180.) * np.pi
distance = 1.0
a = np.array([5.,5.,3.])
b = np.array([6.,6.,3.])
dimensions = np.stack((a,b))
distances = np.array([1.,2.])
RT60 = np.array([0.2,0.3])

absorption_1 = compute_absorption(a[0],a[2],RT60[0])
absorption_2 = compute_absorption(b[0],b[2],RT60[1])
absorptions = np.array([absorption_1,absorption_2])

DOAs = np.linspace(0, 180, 37)
print(DOAs)
x_y_position = np.array([random.uniform(0.5, 4.5),random.uniform(0.5, 4.5),random.uniform(0.5, 2.5)])
print(x_y_position)



snr = 1000.
fs,signal_wav = wavfile.read(file_to_open)
snrs = np.array([5,10,15,20])
white_noise = np.random.normal(0.0,0.5,44100*2)


### Testing that generate_room works properly
if (test_fcts == True): 
	ang = (np.array([70.]) / 180.) * np.pi
	ang1 = (np.array([170.0]) / 180.) * np.pi
	room1,audio_signals,source_loc,t = generate_room_from_conditions(white_noise,x_y_position,DOAs[0],distances[1],absorptions,snr,dimensions,1)
	room2,audio_signals1,source_loc1,t1 = generate_room_from_conditions(white_noise,x_y_position,DOAs[1],distances[1],absorptions,snr,dimensions,1)

	#Plotting the RIR for each microphone
	room1.plot_rir()
	fig = plt.gcf()
	fig.set_size_inches(20, 10)

	#Plotting the RIR for each microphone
	room2.plot_rir()
	fig = plt.gcf()
	fig.set_size_inches(20, 10)

# Testing that generate_phasematrix works properly

	_,_,phase0 = generate_phasematrix_from_signals(audio_signals,0)
	_,_,phase1 = generate_phasematrix_from_signals(audio_signals1,1)

	eq = np.array_equal(phase0, phase1)

	if(eq==False):
		print("generate_phasematrix works properly")
	else:
		print("There is a bug in generate_phasematrix_from_signals")	

# Generating the entire Dataset.
Dataset = genere_dataset()

# Saving the Dataframe into a pickle to use it for the Training
Dataset.to_pickle("/home/janjar/Dataset/Trainingset/Training_dataframe.pkl")







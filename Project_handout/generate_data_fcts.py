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

# Function that computes the absorption out of the room dimensions and the RT60.

def compute_absorption(room_eadge,room_height,RT60):
    V = (room_eadge**2)*room_height
    S = 2*(room_eadge**2) + 4*(room_height**2)
    absr = 0.1611*(V/(S*RT60))
    return absr

# Methode that generates a room out of the conditions given in argument.
# Returns the audio_signals for the 4 mics, the location of the source and the absorption of the room .    

def generate_room_from_conditions(signal,array_position,ang,distance,absorptions,snr,dimensions,typeofroom):
   
    room_dim = dimensions[typeofroom]
    room_eadge = room_dim[0]

    sf = (0.03*np.sqrt(2)/4)
    corners = np.array([[0,0],[0,room_eadge],[room_eadge,room_eadge],[room_eadge,0]]).T
    
    absr = absorptions[typeofroom]
    room = pra.Room.from_corners(corners, fs=fs , max_order=8 ,absorption=absorptions[typeofroom])
    room.extrude(3.)

    # Add sources of 1 second duration
    #room_dim = np.array([5,5,3])
    rng = np.random.RandomState(23)
    duration_samples = int(fs)


    source_location = room_dim / 2 + (distance * np.array([np.cos(ang), np.sin(ang),0.0]))
    source_signal = rng.randn(duration_samples)
    room.add_source(source_location, signal = signal)
    #print('Here is the source at this step:',source_location) 

    #We initiate the point of the Tethra then we scale them
    #and translate the origin of the Tethra to the center of the room.
    R = np.array([(sf*np.array([1,1,-1,-1]))+array_position[0],(sf*np.array([1,-1,1,-1]))+array_position[1],(sf*np.array([1,-1,-1,1]))+array_position[2]])# [[x], [y], [z]]
    room.add_microphone_array(pra.MicrophoneArray(R,room.fs))
    room.image_source_model(use_libroom=True)
    #print("Here is the location of the mic-array:",R)

    #Visualization
    #fig, ax = room.plot(img_order=3)
    #fig.set_size_inches(18.5,10.5)

    room.simulate(snr = snr)
    #print('Here is the audio signal:',room.mic_array.signals[0][:5])
    return room,room.mic_array.signals,source_location,absr

# Method that compute the phasematrix out of a signal and store it in the training folder.
# Returns the paths to the audio_file and the phase_matrix.

def generate_phasematrix_from_signals(signals,j):
    
    #Here also the path should be the folder you need the audio files to be stored on.
        path = "/home/janjar/Dataset/Trainingset/"   
        name_signals = 'audio_signals/audio_signals-{}'.format(j)
        fileName_audio = path + name_signals
        fileObject = open(fileName_audio, 'wb')
        pkl.dump(signals, fileObject)
        fileObject.close()                
    
        phase_matrix = np.zeros((4, 256, 360))
        
        #print("Ici elle doit etre vide",phase_matrix)
        for i in range(4):
            f, t, stft_mic0 = signal.stft(signals[i,:].astype(np.float32), nperseg=512)
            spectrum = stft_mic0
            #print("shape of the STFT:",spectrum.shape)
            magnitude = np.abs(spectrum)
            phase = np.angle(spectrum)
            #print('Here is the phase:',phase.shape)
            #print("HERE IS THE SHAPE",phase.shape)
            #phase_matrix = np.empty((4, x, y))
            phase_matrix[i] = phase[:256,:360]
        
        
        #print("Matrice finale",phase_matrix)
        path = "/home/janjar/Dataset/Trainingset/"   
        name_matrix = 'phase_matrix/Phase_matrix-{}'.format(j)
        fileName_matrix = path + name_matrix
        fileObject = open(fileName_matrix, 'wb')
        pkl.dump(phase_matrix, fileObject)
        fileObject.close()                    
        return fileName_audio,fileName_matrix,phase_matrix


# Funtion to assign the right label to each angle.

def assign_label_to_anlges(angles):    
    bins = np.linspace(0, 180, 37)
    label = np.digitize(angles, bins)
    return label-1


## The main function that generate the entire Dataset set including the Dataframe to represent it.
## Return the Dataframe representing the training set.

def genere_dataset():
        index = 0
        df = pd.DataFrame(columns = ['Room','Array_position','Distance','Absorption','SNR','Audio_file','Phase_Matrix','Label'])
        snr = np.array([0,5,10,15,20]) #Comment this part to turn the snr into random assignment for each room
	#We iterate over the type of rooms (R1,R2).
        for i in range (2):          
	#We iterate over the distances from the microphones (1m,2m).	
            for j in range(2):
	#We iterate on the 7 random positions of the array.		
                for k in range(7):
	#We iterate on the 37 random positions of the source.		            
                    for l in range(37):
                    
                        print('Room:',index)
                        #angles = np.random.uniform(low=0, high=179, size=(7,))
                        #print(angles)
                        #print('Here are the angles for this Room:',angles) #Make the angles able to go to floats.
                        label = assign_label_to_anlges(DOAs[l])
                        #print('Here are the labels for this Room:',labels) #Check if the labels match the angles.
                        #azimuth = angles / 180. * np.pi 
                        snr = snrs[np.random.randint(4,size = 1)] # Uncomment this part to turn the snr into random assignment for each room
                        white_noise = np.random.normal(0.0,random.uniform(0, 1),44100*2)
                        array_position = np.array([random.uniform(0.5, 4.5),random.uniform(0.5, 4.5),random.uniform(0.5, 2.5)])
                        _,signals,array_position,absr = generate_room_from_conditions(white_noise,array_position,DOAs[l],distances[j],absorptions,snr,dimensions,i)
                        fileName_audio,fileName_matrix,_ = generate_phasematrix_from_signals(signals,index)
                        df = df.append({'Room':i,'Array_position':array_position,'Angle':DOAs[l],'Distance':distances[j],'Absorption':absr,'SNR':snr,'Audio_file':fileName_audio,'Phase_Matrix':fileName_matrix,'Label':label} , ignore_index=True)
                        index = index+1
        return df               



















import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
from scipy import signal
import pandas as pd


distances = [1.,2.]
RT =[0.3,0.2] # Dont know how to relate the RT the absotion and the max order
dimensions = [[5.,5.],[6.,6.]]

c = 343.    # speed of sound
fs , signal_wav = wavfile.read("/Users/youssef/Documents/EPFL/Semester_Project/Dataset/GDWN/audiocheck.net_whitenoisegaussian.wav")  # sampling frequency
nfft = 256  # FFT size
freq_range = [300, 3500]
center = [2, 1.5]
snr_db = 5.    # signal-to-noise ratio
sigma2 = 10**(-snr_db / 10) / (4. * np.pi * distance)**2
phase_matrix = np.empty((4, 129, 349))
data_matrix = np.empty((12,4, 129, 349))
df = pd.DataFrame(columns=['Room','Distance_Mic','RT60','SNR','Index_PM','label'])


def generate_room_from_conditions(ang,distances,RT,dimensions,typeofroom):
  # Create an anechoic room
	room_dim = dimensions[typeofroom]
	aroom = pra.ShoeBox(room_dim, fs=fs, max_order=0, sigma2_awgn=sigma2)
	echo = pra.circular_2D_array(center = room_dim*0.5, M=4, phi0=0, radius=0.0047746483)
	aroom.add_microphone_array(pra.MicrophoneArray(echo, aroom.fs))
	# Add sources of 1 second duration
	rng = np.random.RandomState(23)
	duration_samples = int(fs)

	source_location = room_dim / 2 + distance * np.r_[np.cos(ang), np.sin(ang)]
	source_signal = rng.randn(duration_samples)
	aroom.add_source(source_location, signal=source_signal)
	    
	# Run the simulation
	aroom.simulate()

	#Returns the signals for the 4 different micros in these conditions
	return aroom.mic_array.signals

def generate_phasematrix_from_signals(signals):
	phase_matrix = np.empty((4, 129, 349))
	for i in range(4):
    	f, t, stft_mic0 = signal.stft(signals[i,:].astype(np.float32), fs)
    	spectrum = stft_mic0
    	magnitude = np.abs(spectrum)
    	phase = np.angle(spectrum)
    	phase_matrix[i] = phase
    return phase_matrix	

def genere_dataset():
	index = 0
	#We iterate of the type of rooms (R1,R2).
	for i in range (2):
	#We iterate of the distances from the microphones (1m,2m).	
		for j in range(2):
	#We iterate on the 7 random positions of the source.		
			for k in range(1):
				angles = np.random.uniform(low=0, high=180, size=(7,)) #Make the angles able to go to floats.
				azimuth = angles / 180. * np.pi 
				singals = generate_room_from_conditions(azimuth[k],distances[j],RT,dimensions[i],i)
				data_matrix[index] = generate_phasematrix_from_signals(signals)
				df = df.append({'Room':i,'Distance_Mic':distances[j],'RT60':0.2,'SNR':1,'Index_PM':index,'label':assign_label_to_anlges(azimuth[k])} , ignore_index=True)
				index = index+1


def assign_label_to_anlges(angles):
	bins = np.linspace(0, 180, 37)
	label = np.digitize(angles, bins)
	return label




	#We iterate over the SNR but i don't know how to aproach it 

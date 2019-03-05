from scipy.signal import correlate
import numpy as np 
import matplotlib.pyplot as plt




def signals_cross_correlation(signal1, signal2):
    l1 = len(signal1)
    l2 = len(signal2)
    
    #normalize the input arrays
    signal1 = (signal1-signal1.mean()) / signal1.std()
    signal2 = (signal2-signal2.mean()) / signal2.std()
    
    if l1 > l2:
        
        corr = correlate(signal1, signal2)
    else:
        corr = correlate(signal2, signal1)
    
    return corr 

def plot_cross(corr):
    fig = plt.figure()
    fig.set_size_inches(10,4)
    plt.plot(corr)
    plt.title("Correlation")


def compute_spectrogram(fs, signal, title, low_pass=False):
    
    
    #signal = signal[:,0] # mono audio
    
    
    duration = len(signal)/fs
    
    #apply the lowpass filter if needed
    if low_pass:
        h_len = 50
        h = np.ones(h_len)
        h /= np.linalg.norm(h)

        
        # stft paramaters 
        fft_len = 512 
        block_size = fft_len - 50 + 1
        hop = block_size // 2
        window = pra.hann(block_size, flag='asymmetric', length='full') 

        stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, channels=1)
        stft.set_filter(h, zb=h.shape[0] - 1)  
        processed_audio = np.zeros(signal.shape)
        n = 0
        
        while  signal.shape[0] - n > hop:
            stft.analysis(signal[n:n+hop,])
            stft.process()  # apply the filter
            processed_audio[n:n+hop,] = stft.synthesis()
            n += hop
            
        fig = plt.figure()
        fig.set_size_inches(20,3)
        
        plt.subplot(2,1,1)
        #ax1.set_xlim([0, duration])
        spec_sig, _, _, _ = plt.specgram(signal[:n-hop].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title('Original Signal', fontsize=22, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(2,1,2)
        #ax2.set_xlim([0, duration])
        spec_lp, _, _, _ = plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title('Lowpass Filtered Signal', fontsize=22, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        return spec_sig, spec_lp
    else:
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_xlim([0, duration-2])
        fig.set_size_inches(20,3)
        spec, freqs, _, img = plt.specgram(signal.astype(np.float32), NFFT=256, Fs=fs,  vmin=-20, vmax=30) # dBFS scale
        plt.title(title, fontsize=15, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        return spec, freqs
    
    
def compute_spectrogram_phase(fs, signal, title, low_pass=False):
    
    
    #signal = signal[:,0] # mono audio
    
    
    duration = len(signal)/fs
    
    #apply the lowpass filter if needed
    if low_pass:
        h_len = 50
        h = np.ones(h_len)
        h /= np.linalg.norm(h)

        
        # stft paramaters 
        fft_len = 512 
        block_size = fft_len - 50 + 1
        hop = block_size // 2
        window = pra.hann(block_size, flag='asymmetric', length='full') 

        stft = pra.realtime.STFT(block_size, hop=hop, analysis_window=window, channels=1)
        stft.set_filter(h, zb=h.shape[0] - 1)  
        processed_audio = np.zeros(signal.shape)
        n = 0
        
        while  signal.shape[0] - n > hop:
            stft.analysis(signal[n:n+hop,])
            stft.process()  # apply the filter
            processed_audio[n:n+hop,] = stft.synthesis()
            n += hop
            
        fig = plt.figure()
        fig.set_size_inches(20,3)
        
        plt.subplot(2,1,1)
        #ax1.set_xlim([0, duration])
        spec_sig, _, _, _ = plt.specgram(signal[:n-hop].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title('Original Signal', fontsize=22, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(2,1,2)
        #ax2.set_xlim([0, duration])
        spec_lp, _, _, _ = plt.specgram(processed_audio[hop:n], NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title('Lowpass Filtered Signal', fontsize=22, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        return spec_sig, spec_lp
    else:
        fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.set_xlim([0, duration-2])
        fig.set_size_inches(20,3)
        spec, freqs, _, img = plt.specgram(signal.astype(np.float32), NFFT=256, Fs=fs, mode='phase',  vmin=-20, vmax=30) # dBFS scale
        plt.title(title, fontsize=15, fontweight="bold")
        plt.xlabel('Time [s]')
        plt.ylabel('Frequency [Hz]')
        return spec, freqs
    
    
def plot_spectrogramsOfWords(fs, s1, s2, s3, s4, s5, s6):  #s1 = (word, signal)
        fig = plt.figure()
        fig.set_size_inches(20,10)
        plt.subplot(3,2,1)
        plt.specgram(s1[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s1[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,2)
        plt.specgram(s2[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s2[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,3)
        plt.specgram(s3[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s3[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')

        
        plt.subplot(3,2,4)
        plt.specgram(s4[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s4[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,5)
        plt.specgram(s5[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s5[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,6)
        plt.specgram(s6[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s6[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
def plot_spectrogramsOfWords_phase(fs, s1, s2, s3, s4, s5, s6):  #s1 = (word, signal)
        fig = plt.figure()
        fig.set_size_inches(20,10)
        plt.subplot(3,2,1)
        plt.specgram(s1[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s1[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,2)
        plt.specgram(s2[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s2[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,3)
        plt.specgram(s3[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s3[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')

        
        plt.subplot(3,2,4)
        plt.specgram(s4[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s4[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,5)
        plt.specgram(s5[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s5[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(3,2,6)
        plt.specgram(s6[1].astype(np.float32), NFFT=256, Fs=fs, mode='phase', vmin=-20, vmax=30) # dBFS scale
        plt.title(s6[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
            
    
def plot_spectrogramsOfLetters(fs, s1, s2, s3, s4):  #s1 = (word, signal)
        fig = plt.figure()
        fig.set_size_inches(15,8)
        plt.subplot(2,2,1)
        plt.specgram(s1[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s1[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(2,2,2)
        plt.specgram(s2[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s2[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
        
        plt.subplot(2,2,3)
        plt.specgram(s3[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s3[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')

        
        plt.subplot(2,2,4)
        plt.specgram(s4[1].astype(np.float32), NFFT=256, Fs=fs, vmin=-20, vmax=30) # dBFS scale
        plt.title(s4[0], fontsize=11, fontweight="bold")
        plt.ylabel('Frequency [Hz]')
        
    
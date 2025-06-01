import librosa
import numpy as np
import matplotlib.pyplot as plt
import DataManager as dm
import scipy.signal as sig
import sounddevice as sd


def process_frame_methods1(signal,sr,M,N):
   upsampled=upsample(signal,M)
   filtered=filter_signal(upsampled,sr/2,M,N)
   downsampled=downsample(filtered,N)
   return downsampled
def process_frame_methods2(signal,_,M,N):
    return sig.resample(signal,int(len(signal)/(N/M)))
def upsample(signal,M):
    M_minus_1_zeros=(M-1)*[0]
    for i in range(1,len(signal)):
       next_sample_id=(i-1)*M+1
       signal=np.insert(signal,next_sample_id,M_minus_1_zeros)
    return signal
def filter_signal(signal,nyquist_frequency,M,N):
    order=5
    cutoff_frequency=min(nyquist_frequency/M,nyquist_frequency/N)
    normalized_cutoff=cutoff_frequency/nyquist_frequency
    b,a=sig.butter(order,normalized_cutoff,btype='low',analog=False)
    return sig.lfilter(b,a,signal)

def downsample(signal,N):
    return [signal[i] for i in range(0, len(signal), N)] 
    
def variable_speed_replay(signal,sr,M,N,process_frame_methods,framelength):
    if (N/M==1):
        return signal
    output=np.empty((0,))
    for i in range(0,len(signal),framelength):
        end=min(i+framelength,len(signal))
        processed=process_frame_methods(signal[i:end],sr,M,N)
        output=np.append(output,processed)
    return output
def play_signal(signal,sr,volume=1.0):
    sd.play(signal * volume, samplerate=sr)
    sd.wait()  # Attendre la fin de la lecture
    
data= dm.DataManager("Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav")
N=7
M=5 #v=N/M
sample_rate = data.sample_rate
signal = data.signal
transformed=variable_speed_replay(signal,sample_rate,M,N,process_frame_methods2,framelength=1024)
play_signal(transformed, sample_rate, volume=0.5)
data.plot_audio(transformed, sample_rate, title="Signal Transformé")
    
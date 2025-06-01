import numpy as np
import librosa 
import matplotlib.pyplot as plt
import soundfile as sf
import DataManager as dm

def coeff(f0,sr):
    f1=2*np.pi*f0/sr
    alpha=np.sin(f1)/2*0.707
    b0=(1-np.cos(f1))/2
    b1=1-np.cos(f1)
    b2=b0
    a0=1+alpha
    a1=-2*np.cos(f1)
    a2=1-alpha
    return[[b0/a0,b1/a0,b2/a0],[a1/a0,a2/a0]]
    
def EQ(input,output,coeff):
    output[0]=0
    output[1]=0
    for i in range(2,len(input)):
        output[i]=coeff[0][0]*input[i]+coeff[0][1]*input[i-1]+coeff[0][2]*input[i-2]-coeff[1][0]*output[i-1]-coeff[1][1]*output[i-2]
    return output

def main():
    INPUT_FILE ="Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav"
    data = dm.DataManager(INPUT_FILE)
    input=data.signal
    sample_rate=data.sample_rate
    output=np.zeros(input.size)
    coefficient=coeff(100,sample_rate)
    EQ(input,output,coefficient)
    data.plot_audio(input,sample_rate,"Input signal")
    data.plot_audio(output,sample_rate,"Output signal")
    librosa.util.normalize(output)
    sf.write("output.wav", output, sample_rate)
    print("Output written to output.wav")
if __name__ == "__main__":
    main()
    
import numpy as np
import librosa 
import matplotlib.pyplot as plt
import soundfile as sf



def load_audio(file_path):
    """Charge un fichier audio et retourne le signal + fréquence d'échantillonnage"""
   
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if audio.ndim > 1:
        audio=librosa.to_mono(audio)  # Convertir en mono si nécessaire
    audio = audio.astype(np.float32)  # Assurez-vous que le type de données est float32
    return audio,sr
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
def plot_audio(signal, sample_rate, title="Signal Audio"):
    """Visualise le signal audio"""
    plt.figure(figsize=(15, 5))
    
    # Version temps
    plt.subplot(1, 2, 1)
    time = np.arange(len(signal)) / sample_rate
    plt.plot(time, signal, alpha=0.7)
    plt.title(f"{title} (Domaine temporel)")
    plt.xlabel("Temps (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Version fréquentielle
    plt.subplot(1, 2, 2)
    fft = np.fft.rfft(signal)
    freq = np.fft.rfftfreq(len(signal), d=1/sample_rate)
    magnitude = np.abs(fft) / len(signal)
    plt.plot(freq, magnitude)
    plt.title("Spectre fréquentiel")
    plt.xlabel("Fréquence (Hz)")
    plt.ylabel("Magnitude")
    plt.xlim(0, 15000)  
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
def main():
    INPUT_FILE ="Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav"
    input=load_audio(INPUT_FILE)[0]
    sample_rate=load_audio(INPUT_FILE)[1]
    output=np.zeros(input.size)
    coefficient=coeff(100,sample_rate)
    EQ(input,output,coefficient)
    plot_audio(input,sample_rate,"Input signal")
    plot_audio(output,sample_rate,"Output signal")
    librosa.util.normalize(output)
    sf.write("output.wav", output, sample_rate)
    print("Output written to output.wav")
if __name__ == "__main__":
    main()
    
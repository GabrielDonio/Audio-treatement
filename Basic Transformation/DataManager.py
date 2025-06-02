import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.signal as sig

class DataManager:
    def __init__(self,filepath):
        self.filepath = filepath
        self.signal, self.sample_rate = self.load_audio()
        
    def load_audio(self):
        """Charge un fichier audio et retourne le signal + fréquence d'échantillonnage"""
        audio, sr = librosa.load(self.filepath, sr=None, mono=True)
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        audio = audio.astype(np.float32)
        return audio, sr
    @staticmethod
    def plot_audio(sig,sr,title="Signal Audio"):
        """Visualise le signal audio"""
        plt.figure(figsize=(15, 5))
        
        # Version temps
        plt.subplot(1, 2, 1)
        time = np.arange(len(sig)) / sr
        plt.plot(time, sig, alpha=0.7)
        plt.title(f"{title} (Domaine temporel)")
        plt.xlabel("Temps (s)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        
        # Version fréquentielle
        plt.subplot(1, 2, 2)
        fft = np.fft.rfft(sig)
        freq = np.fft.rfftfreq(len(sig), d=1/sr)
        magnitude = np.abs(fft) / len(sig)
        plt.plot(freq, magnitude)
        plt.title("Spectre fréquentiel")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Magnitude")
        plt.xlim(1, 5000)  
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    @staticmethod
    def spectrogram(signal, sample_rate, n_fft=2048, hop_length=512):
        """Affiche le spectrogramme d'un signal audio"""
        plt.figure(figsize=(10, 6))
        
        # Add small constant to avoid log(0)
        plt.specgram(signal, NFFT=n_fft, Fs=sample_rate, noverlap=hop_length, 
                     scale='dB', mode='magnitude')
        
        plt.title("Spectrogramme")
        plt.xlabel("Temps (s)")
        plt.ylabel("Fréquence (Hz)")
        plt.colorbar(label='Magnitude (dB)')
        plt.show()
    @staticmethod
    def save_audio(signal,sr, output_path):
        """Enregistre le signal audio dans un fichier WAV"""
        sf.write(output_path, signal, sr)
        print(f"Audio saved to {output_path}")
        
def main():
    filepath = "/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav"
    data_manager = DataManager(filepath)
    # Affichage du spectrogramme
    data_manager.spectrogram(data_manager.signal, data_manager.sample_rate)
if __name__ == "__main__":
    main()
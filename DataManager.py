import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

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
        plt.xlim(0, 15000)  
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
    def save_audio(self, output_path):
        """Enregistre le signal audio dans un fichier WAV"""
        sf.write(output_path, self.signal, self.sample_rate)
        print(f"Audio saved to {output_path}")
        
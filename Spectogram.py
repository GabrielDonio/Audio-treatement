import numpy as np
import librosa
import librosa.feature as lbf
import matplotlib.pyplot as plt

def load_audio(file_path):
    """Charge un fichier audio et retourne le signal + fréquence d'échantillonnage"""
    audio, sr = librosa.load(file_path, sr=None, mono=True)
    if audio.ndim > 1:
        audio = librosa.to_mono(audio)  # Convertir en mono si nécessaire
    audio = audio.astype(np.float32)  # Assurez-vous que le type de données est float32
    return audio, sr
def main():
    file_path = 'Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav'  # Remplacez par le chemin de votre fichier audio
    signal, sample_rate = load_audio(file_path)
    
if __name__ == "__main__":
    main()

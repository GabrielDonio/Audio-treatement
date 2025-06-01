import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import librosa
import DataManager as dm


def autocorrelation_basique(signal):
    """
    Calcule l'autocorrélation d'un signal de manière triviale (O(n²))
    
    Args:
        signal: Liste ou array 1D de valeurs numériques
        
    Returns:
        corr: Array des coefficients d'autocorrélation pour chaque décalage
    """
    n = len(signal)
    corr = np.zeros(n)  # Initialisation du résultat
    
    # Calcul pour chaque décalage possible τ
    for tau in range(n):
        somme = 0.0
        
        # Calcul de la somme des produits pour chaque position t
        for t in range(n - tau):
            produit = signal[t] * signal[t + tau]
            somme += produit
        
        # Normalisation par le nombre de points utilisés
        corr[tau] = somme / (n - tau)
    
    return corr
def find_pitch(signal, sr):
    """
    Trouve la fréquence fondamentale (pitch) d'un signal audio en utilisant l'autocorrélation.
    
    Args:
        signal: Signal audio mono
        sample_rate: Fréquence d'échantillonnage du signal
        
    Returns:
        pitch: Fréquence fondamentale en Hz
    """
    corr = autocorrelation_basique(signal)
    
    min_period = int(sr / 2000)  # Période min = 1/max_freq
    max_period = int(sr / 50)    # Période max = 1/min_freq
    
    # On cherche le premier maximum local dans la plage réaliste
    peak_index = min_period
    for i in range(min_period + 1, min(max_period, len(corr) - 1)):
        # Un pic est défini comme un point plus grand que ses voisins
        if corr[i] > corr[i-1] and corr[i] > corr[i+1]:
            if corr[i] > corr[peak_index]:
                peak_index = i
    
    # 4. Calcul de la fréquence fondamentale
    period = peak_index  # Période en échantillons
    f0 = sr / period     # Fréquence en Hz
    
    return f0

if __name__ == "__main__":
    # Configuration
    FILE_PATH = "Data audio/683572__trader_one__piano-loop-verse-5-90bpm-34.flac"  # Remplacez par votre fichier
    data= dm.DataManager(FILE_PATH) 
    # Chargement de l'audio
    sample_rate=data.sample_rate
    audio= data.signal[0:3000]
    #Informations de base
    duration = len(audio) / sample_rate
    print(f"\n{'-'*50}")
    print(f"Fichier audio chargé: {FILE_PATH}")
    print(f"Fréquence d'échantillonnage: {sample_rate} Hz")
    print(f"Durée: {duration:.2f} secondes")
    print(f"Nombre d'échantillons: {len(audio)}")
    print(f"Type de données: {audio.dtype}")
    print(f"Amplitude max: {np.max(audio):.4f}")
    print(f"{'-'*50}\n")
    print(find_pitch(audio, sample_rate))
    
    # Visualisation
    data.plot_audio(audio, sample_rate)
    
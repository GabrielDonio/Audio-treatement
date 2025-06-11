import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import BasicTransformation.DataManager as dm

def apply_multiple_delays(signal, sample_rate, delays, amplitudes):
    """
    Applique plusieurs échos au signal
    Args:
        signal: Signal d'entrée
        sample_rate: Fréquence d'échantillonnage
        delays: Liste des délais en secondes
        amplitudes: Liste des amplitudes pour chaque écho
    """
    # Vérification des paramètres
    if len(delays) != len(amplitudes):
        raise ValueError("Le nombre de delays et d'amplitudes doit être identique")
    
    # Calcul de la longueur maximale nécessaire
    max_delay_samples = round(max(delays) * sample_rate)  # Removed extra max()
    output_length = len(signal) + max_delay_samples
    
    # Initialisation du signal de sortie
    output = np.zeros(output_length, dtype=np.float32)
    output[:len(signal)] = signal  # Signal original
    
    # Application de chaque écho
    for delay, amp in zip(delays, amplitudes):
        delay_samples = round(delay * sample_rate)
        output[delay_samples:delay_samples + len(signal)] += signal * amp
    
    return output

# Paramètres
delays = [0.5, 0.9, 1.4, 1.8]  # Délais en secondes
amplitudes = [0.7, 0.5, 0.3, 0.1]  # Amplitude décroissante pour chaque écho

# Chargement et normalisation
data = dm.DataManager("/Users/macos/Documents/Audio python/Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav")
fs = data.sample_rate
signal = data.signal.astype(np.float32)

# Normalisation selon le type d'entrée
if signal.dtype == np.int16:
    signal = signal / 32768.0
elif signal.dtype == np.int32:
    signal = signal / 2147483648.0

# Application des échos multiples
output = apply_multiple_delays(signal, fs, delays, amplitudes)

# Normalisation finale
output = np.clip(output, -1.0, 1.0)
output = (output * 32767).astype(np.int16)

# Sauvegarde
wavfile.write("/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57_multiple_delays.wav", 
              fs, output)

print(f"Effet delay appliqué avec {len(delays)} échos!")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wavfile
import BasicTransformation.DataManager as dm

# Paramètres
echo_duration = 0.6  # Durée de l'écho en secondes
delay_amp = 0.7    # Amplitude de l'écho

# Chargement et normalisation
data = dm.DataManager("/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav")
fs = data.sample_rate
signal = data.signal
signal = signal.astype(np.float32)

# Normalisation selon le type d'entrée
if signal.dtype == np.int16:
    signal = signal / 32768.0
elif signal.dtype == np.int32:
    signal = signal / 2147483648.0

# Effet delay
delay_len_samples = round(echo_duration * fs)
leading_zero_padding = np.zeros_like(signal[:delay_len_samples])
delayed_signal = np.concatenate((leading_zero_padding, signal))
end_padding_len_sample = len(delayed_signal) - len(signal)
end_padding_sig = np.zeros_like(signal[:end_padding_len_sample])
signal = np.concatenate((signal, end_padding_sig))

# Mix + normalisation finale
summed_signal = signal + delay_amp * delayed_signal[:len(signal)]
summed_signal = np.clip(summed_signal, -1.0, 1.0)
summed_signal = (summed_signal * 32767).astype(np.int16)

# Sauvegarde
wavfile.write("/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57_delay1.wav", 
              fs, summed_signal)

print("Effet delay appliqué avec succès!")
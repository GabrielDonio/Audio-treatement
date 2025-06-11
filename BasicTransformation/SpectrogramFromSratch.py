import numpy as np
import matplotlib.pyplot as plt
import DataManager as dm

data= dm.DataManager("/Users/macos/Documents/Audio python/Data audio/Cymatics - Future Bass Drop Loop 6 - 160 BPM G Min.wav")
signal = data.signal
sr = data.sample_rate
window_size = 1024
overlap = 1000

def hanning_window(size):
    hanning=np.array([0.5 - 0.5 * np.cos(2 * np.pi * n / (size - 1)) for n in range(size)])
    return hanning
hanning= hanning_window(window_size)
windows=np.array([signal[i:i + window_size] * hanning for i in range(0, len(signal) - window_size, window_size - overlap)])

spectrogram = [np.abs(np.fft.rfft(window))**2 for window in windows]
spectrogram = np.array(spectrogram).T # transpose spectrogram
# Plot the spectrogram
frequencies = np.fft.rfftfreq(window_size, d=1.0/sr)
time = np.arange(len(spectrogram[0])) * (window_size - overlap) / sr
plt.pcolormesh(time, frequencies, 10 * np.log10(spectrogram))
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power spectral density (dB/Hz)")
plt.ylim([0, sr/2.])
plt.show()


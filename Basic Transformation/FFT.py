import numpy as np
import matplotlib.pyplot as plt
import DataManager as dm

def FFT(signal):
    """
    Implements the Fast Fourier Transform using the Cooley-Tukey algorithm
    Args:
        signal: Input signal (must have length that is power of 2)
    Returns:
        FFT of the signal
    """
    signal = np.asarray(signal, dtype=complex)
    N = len(signal)
    
    # Base case
    if N <= 1:
        return signal
    
    # Make sure N is a power of 2
    if N & (N-1) != 0:
        raise ValueError("Signal length must be a power of 2")
    
    # Recursive case
    even = FFT(signal[::2])  # Even indices
    odd = FFT(signal[1::2])  # Odd indices
    
    # Combine
    factor = np.exp(-2j * np.pi * np.arange(N) / N)
    return np.concatenate([
        even + factor[:N//2] * odd,
        even + factor[N//2:] * odd
    ])

# Main execution
Input = "/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav"
data = dm.DataManager(Input)

# Pad signal length to nearest power of 2
signal_length = len(data.signal)
next_power_2 = 2**int(np.ceil(np.log2(signal_length)))
signal = np.pad(data.signal, (0, next_power_2 - signal_length))
sr = data.sample_rate

# Plot original signal
data.plot_audio(signal, sr, title="Signal Audio")

# Compute FFT
X = FFT(signal)

# calculate the frequency
N = len(X)
n = np.arange(N)
T = N/sr
freq = n/T 

plt.figure(figsize = (12, 6))
plt.subplot(121)
plt.stem(freq, abs(X), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('FFT Amplitude |X(freq)|')

# Get the one-sided specturm
n_oneside = N//2
# get the one side frequency
f_oneside = freq[:n_oneside]

# normalize the amplitude
X_oneside =X[:n_oneside]/n_oneside

plt.subplot(122)
plt.stem(f_oneside, abs(X_oneside), 'b', \
         markerfmt=" ", basefmt="-b")
plt.xlabel('Freq (Hz)')
plt.ylabel('Normalized FFT Amplitude |X(freq)|')
plt.tight_layout()
plt.show()
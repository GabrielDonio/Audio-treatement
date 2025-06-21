import numpy as np
import scipy.io.wavfile as wav

def main():
    sample_rate = 44100  # Fréquence d'échantillonnage
    frequency= 440  # Fréquence de la note A4
    duration = 3.0  # Durée en secondes
    waveform=np.sin
    wavetable_length=64
    wave_table= np.zeros((wavetable_length, ))
    for n in range(wavetable_length):
        wave_table[n] = waveform(2 * np.pi * n / wavetable_length)
    output = np.zeros(int(sample_rate * duration))
    index=0
    index_increment = wavetable_length * frequency / sample_rate
    for n in range(output.shape[0]):
        #output[n] = wave_table[int(np.floor(index))]
        output[n]=interpolate(wave_table, index)
        index += index_increment
        index%= wavetable_length
    gain=-20
    amplitude = 10 ** (gain / 20)
    output *= amplitude
    wav.write('sine_wave.wav', sample_rate, output.astype(np.float32)) 
def interpolate(wave_table, index):
    trucated_index = int(np.floor(index))
    newtable_index = (trucated_index + 1)%wave_table.shape[0]
    next_index_weight = index - trucated_index
    truncated_index_weight = 1 - next_index_weight
    return truncated_index_weight * wave_table[trucated_index] + next_index_weight * wave_table[newtable_index]
    
main()
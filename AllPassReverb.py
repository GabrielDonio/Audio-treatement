import numpy as np
import BasicTransformation.DataManager as dm

class AllPassReverb:
    def __init__(self, input_signal,gain,delay_time,fs):
        self.input_signal = input_signal
        self.gain = gain
        self.delay_time = delay_time
        self.fs = fs
        self.delay_samples = int(self.delay_time * self.fs)
        
        def process_signal(self):
            # Initialize the output signal with zeros
            output_signal = np.zeros(len(self.input_signal))
            for i in range(self.delay_samples, len(self.input_signal)):
                # Apply the all-pass filter formula
                output_signal[i] = (-self.gain*self.input_signal[i] + self.input_signal[i - self.delay_samples] + self.gain * output_signal[i - self.delay_samples])
            for i in range(self.delay_samples):
                output_signal[i] = self.input_signal[i]
            return output_signal
        self.output_signal = process_signal(self)

data= dm.DataManager('/Users/macos/Documents/Audio python/Data audio/705944__josefpres__guitar-tones-005-string-b-22-tone-a57.wav')
signal=data.signal
fs=data.sample_rate
gain = 0.6  # Gain for the all-pass filter
delay_time = 0.001  # Delay time in seconds
all_pass_reverb = AllPassReverb(signal, gain, delay_time, fs)
output_signal = all_pass_reverb.output_signal

data.play(output_signal, fs, 0.8)  # Play the processed signal with volume 1.0
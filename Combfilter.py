import BasicTransformation.DataManager as dm
import numpy as np

class CombFilter:
    def __init__(self, signal, gain, delay_time, sample_rate):
        self.signal = signal
        self.gain = gain
        self.delay_time = delay_time
        self.sample_rate = sample_rate
        self.output_signal = self.process_signal()
        
    def process_signal(self):
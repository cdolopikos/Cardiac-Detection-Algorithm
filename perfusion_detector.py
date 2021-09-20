import data_reader
import numpy as np

class PerfusionDetector:
    def __init__(self):
        self.buffer = np.zeros(2000)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 150  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, perfusion: data_reader.Data_Reader):
        self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        raw=buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'same'))
        return out, raw
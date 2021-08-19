import numpy as np
import filters, data_reader


class ElectrogramDetector:
    def __init__(self):
        self.buffer = np.zeros(3000)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, electrogram: data_reader.Data_Reader):
        self.buffer[0:2800] = self.buffer[200:3000]
        self.buffer[2800:3000] = electrogram.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        out = filters.butter_lowpass_filter(data=buffer, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out = filters.butter_highpass_filter(data=out, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out_mean = np.mean(out)
        out = np.abs(out - out_mean)
        out = np.convolve(out, np.ones(111, dtype=np.int), 'same')
        raw = buffer[100:]
        return out, raw

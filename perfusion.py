import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')
import scipy
from matplotlib import animation
from scipy.optimize import leastsq
from sklearn import preprocessing
import PyQt6
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter  # Filter requirements.
from scipy import fftpack
import time

# ELECTROGRAM_PATH = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/Normal/normalbostoncrtd_11_11_2020_144740_/bpao.txt"
# PERFUSION_PATH = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/Normal/normalbostoncrtd_11_11_2020_144740_/qfin.txt"


PERFUSION_PATH = "/Traces_unzipped_examples/A Tach/Atach_CRTD_21_10_2020_164349_/qfin.txt"
ELECTROGRAM_PATH = "/Traces_unzipped_examples/A Tach/Atach_CRTD_21_10_2020_164349_/boxb.txt"


def butter_lowpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


def butter_highpass(cutoff, fs, order):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b1, a1 = butter(order, normal_cutoff, btype='high', analog=False)
    return b1, a1


def butter_highpass_filter(data, cutoff, fs, order):
    b1, a1 = butter_highpass(cutoff, fs, order=order)
    y1 = filtfilt(b1, a1, data)
    return y1


def dynamic_smoothing(x):
    start_window_length = len(x) // 2
    smoothed = []
    for i in range(len(x)):
        a = float(i) / len(x)
        w = int(np.round(a * start_window_length + (1.0 - a)))
        w0 = max(0, i - w)
        w1 = min(len(x), i + w)
        smoothed.append(sum(x[w0:w1]) / (w1 - w0))
    return np.array(smoothed)


class Data_Reader:

    def __init__(self, data_path):
        self.data_path = data_path
        self.data = self.read_data(self.data_path)
        self.data_len = len(self.data)
        self.data_index = 0

    # reads data
    def read_data(self, file_name):
        return np.genfromtxt(file_name, delimiter=',')

    # gets next 200ms
    def get_next_data(self, amount=200):
        if self.data_index + amount < self.data_len:
            self.data_index = self.data_index + amount
            return self.data[self.data_index: self.data_index + amount]
        else:
            raise Exception


class PerfusionDetector:
    def __init__(self):
        self.buffer = np.zeros(2000)
        # self.low_pass = low_pass
        # self.high_pass = high_pass
        # self.initial_threshold = initial_threshold
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 150  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, electrogram: Data_Reader):
        self.buffer[0:1800] = self.buffer[200:2000]
        # self.buffer[800:2600] = self.buffer[1000:2800]
        self.buffer[1800:2000] = electrogram.get_next_data(amount=200)
        # self.buffer[2600:2800] = electrogram.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'valid'))
        # fig = plt.figure()
        # plt.plot(out)
        # plt.show()
        # # out = preprocessing.scale(out.reshape(-1,1))
        return out

    def find_lows(self, per, ecg_pks):
        for i in ecg_pks:
            print(i)
            per_pks, _ = find_peaks(per)
            print(per_pks)
        return per_pks


def main():
    per_pks = []
    perfusion = Data_Reader(data_path=PERFUSION_PATH)
    perfusion_det = PerfusionDetector()
    win = pg.GraphicsWindow()
    p1 = win.addPlot(row=1, col=0)
    p2 = win.addPlot(row=2, col=0)
    curve1 = p1.plot()
    curve2 = p2.plot()
    dot1 = p1.plot(pen=None, symbol="o")
    dot2 = p1.plot(pen=None, symbol="o")
    dot3 = p2.plot(pen=None, symbol="o")
    count = 0
    while True:
        count = count + 1
        perfusion_det.load_new_data(perfusion)
        tmp = perfusion_det.detect_new_data()
        perfusion_det.load_new_data(perfusion)
        per_out = perfusion_det.detect_new_data()
        b = np.arange(len(per_out))
        min_per_peaks, properties = find_peaks(per_out * -1, prominence=0, width=20)
        per_peaks, properties = find_peaks(per_out, prominence=0, width=20)
        if per_peaks.size > 0:
            per_pks.append(per_peaks)
        # print("pr", per_peaks)
        # print("min", min_per_peaks)
        curve1.setData(x=b, y=per_out)
        dot1.setData(x=per_peaks, y=per_out[per_peaks])
        dot2.setData(x=min_per_peaks, y=per_out[min_per_peaks])
        curve2.setData(x=b, y=per_out)
        dot3.setData(x=per_peaks, y=per_out[per_peaks])
        QtGui.QGuiApplication.processEvents()
        time.sleep(1)
        for i in per_peaks:
            if len(min_per_peaks)>0:
                x2 = min_per_peaks[len(min_per_peaks)-1]
                y2 = per_out[x2]
                if x2 < i:
                    print("x1",i)
                    y1 = per_out[i]
                    print("x2",x2)
                    x1x2= abs(x2 - i)
                    print("x1x2",x1x2)
                    # print(x1x2)
                    print("y1",y1)
                    print("y2",y2)
                    y1y2 = abs(y1-y2)*1000
                    print(y1y2)
                    # print(y1y2)
                    gradientA = x1x2 / y1y2
                    gradientB = y1y2 / x1x2
                    print(gradientA)
                    print(math.degrees(math.atan(gradientA)))
                    alpha=math.degrees(math.atan(gradientA))
                    print(gradientB)
                    print(math.degrees(math.atan(gradientB)))
                    beta=math.degrees(math.atan(gradientB))
                    alpha_beta=alpha-beta
                    print("alpha_beta",alpha_beta)
                    if alpha_beta>0:
                        print("DANGER")



if __name__ == '__main__':
    main()

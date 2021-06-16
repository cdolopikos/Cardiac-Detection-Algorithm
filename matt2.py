import collections
import math
import statistics
import operator
from scipy.stats import skew, kurtosis
import pandas as pd
import sys
from PyQt6.QtWidgets import QTableWidget, QVBoxLayout, QTableWidgetItem, QApplication
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, lfilter  # Filter requirements.
from scipy import fftpack
import time

# pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/qfin.txt"
# ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/boxb.txt"
# bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/vfi0017_vvi180_01_27_04_2021_152246_/boxb.txt"

pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ADA003_03_12_01_2021_154719_/plethh.txt"
ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ADA003_03_12_01_2021_154719_/plethg.txt"
ee1 = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ADA003_03_12_01_2021_154719_/plethg.txt"
bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ADA003_03_12_01_2021_154719_/BP.txt"

DEBUG = False

STEP_SIZE = 200
BP_LAG = 200
PERFUSION_LAG = 200

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


class ElectrogramDetector:
    def __init__(self):
        self.buffer = np.zeros(2000)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, electrogram: Data_Reader):
        self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = electrogram.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        out = butter_lowpass_filter(data=buffer, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out = butter_highpass_filter(data=out, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out_mean = np.mean(out)
        out = np.abs(out - out_mean)
        out = np.convolve(out, np.ones(111, dtype=np.int), 'same')
        raw = buffer[100:]
        return out, raw


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

    def load_new_data(self, perfusion: Data_Reader):
        self.buffer = np.roll(self.buffer, -200)
        #self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'same'))
        return out


class BPDetector:

    V_to_mmHg_factor = 100

    def __init__(self):
        self.buffer = np.zeros(2000)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 150  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, perfusion: Data_Reader):
        self.buffer = np.roll(self.buffer, -200)
        #self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200) * self.V_to_mmHg_factor

    def detect_new_data(self):
        buffer = self.buffer.copy()
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'same'))
        return out


def main(electrogram_path, perfusion_path, bp_path, period):
    ELECTROGRAM_PATH = electrogram_path
    PERFUSION_PATH = perfusion_path
    BP_PATH = bp_path

    electrogram = Data_Reader(data_path=ELECTROGRAM_PATH)
    perfusion = Data_Reader(data_path=PERFUSION_PATH)
    bpdata = Data_Reader(data_path=BP_PATH)

    electrogram_detector = ElectrogramDetector()
    perfusion_det = PerfusionDetector()
    bp_det = BPDetector()

    if not DEBUG:
        win = pg.GraphicsWindow()
        #
        p1 = win.addPlot(row=1, col=0)
        p2 = win.addPlot(row=2, col=0)
        p3 = win.addPlot(row=0, col=0)
        p4 = win.addPlot(row=0, col=1)
        p5 = win.addPlot(row=1, col=1)
        p6 = win.addPlot(row=2, col=1)
        # p7 = win.addPlot(row=0, col=2) .  /

        curve1 = p1.plot()
        dot1 = p1.plot(pen=None, symbol="o")
        curve2 = p2.plot()
        curve3 = p3.plot()
        curve4 = p4.plot()
        curve5 = p5.plot()
        curve6 = p5.plot()
        curve7 = p5.plot()
        curve8 = p5.plot()
        curve9 = p5.plot()
        curve10 = p5.plot()
        curve11 = p6.plot()
        # curve12 = p7.plot()
        dot2 = p2.plot(pen=None, symbol="x")
        dot3 = p2.plot(pen=None, symbol="o")
        dot4 = p4.plot(pen=None, symbol="o")

        table = QTableWidget()
        table.setColumnCount(3)
        table.setRowCount(2)
        table.setHorizontalHeaderLabels(["BPM", "Perfusion-Gradient", "Decision"])

    mat = collections.deque(maxlen=6)
    mat_pks = collections.deque(maxlen=50)

    last_ecg_peak_time = 0

    # print(mat_pks)
    ecg_peaks_total = []
    ecg_pks = []
    per_pks = []
    count = 0
    # count=1200
    hrb = 0
    count = 0
    cons = []
    output = []

    stats = {"Beats per Second (1000ms)": 0,
             "BPM": 0,
             "EGM Mean RV": 0,
             "EGM STD RV": 0,
             "EGM Skewness RV": 0,
             "EGM Kurtosis RV": 0,
             "R-R Interval RV": 0,
             "BP": 0,
             "Max Actual BP": 0,
             "Mean Actual BP": 0,
             "Per Mean": 0,
             "Per STD": 0,
             "Per Skewness": 0,
             "Per Kurtosis": 0,
             "Current Perfusion Grad": 0,
             "Per Cum. Mean": 0,
             "Per Cum. STD": 0,
             "Per Cum. Skewness": 0,
             "Per Cum. Kurtosis": 0,
             "Cumulative Perfusion Grad": 0,
             "Decision": 0}

    start = 0
    rr_interval = 1000
    finish = 400
    # tmp=[]

    while True:
        print(f"count: {count}")
        try:
            electrogram_detector.load_new_data(electrogram)
            perfusion_det.load_new_data(perfusion)
            bp_det.load_new_data(bpdata)

            per_out = perfusion_det.detect_new_data()
            bp_out = bp_det.detect_new_data()
            ecg_out, raw = electrogram_detector.detect_new_data()
        except Exception as e:
            print("Out of data")
            break

        prom = np.mean(ecg_out)
        peaks, properties = find_peaks(ecg_out, prominence=prom, width=20)

        peak_pairs_to_process = []

        for p in list(peaks):
            # global index
            if p > 1400 and p <= 1600:
                peak_global_time = p + (count*STEP_SIZE)
                if len(mat_pks) > 0:
                    peak_pairs_to_process.append((mat_pks[0], peak_global_time))
                mat_pks.appendleft(peak_global_time)

        for start_global_time, end_global_time in peak_pairs_to_process:
            start_local_time = start_global_time - (count*200)
            end_local_time = end_global_time - (count*200)

            rr_interval = end_global_time - start_global_time
            bpm_interval = 60_000 / rr_interval

            mean_bp_interval = np.mean(bp_det.buffer[(start_local_time+BP_LAG):(end_local_time+BP_LAG)])
            max_bp_interval = np.max(bp_det.buffer[(start_local_time+BP_LAG):(end_local_time+BP_LAG)])
            min_bp_interval = np.min(bp_det.buffer[(start_local_time+BP_LAG):(end_local_time+BP_LAG)])

            egmMean_interval = np.mean(ecg_out[start_local_time:end_local_time])
            egmSTD_interval = np.std(ecg_out[start_local_time:end_local_time])
            egmSkew_interval = skew(ecg_out[start_local_time:end_local_time])
            egmKurtosis_interval = kurtosis(ecg_out[start_local_time:end_local_time])

            interval_stats = stats.copy()

            update_dict = {"Max Actual BP": max_bp_interval,
                           "Mean Actual BP": mean_bp_interval,
                           "BPM": bpm_interval,
                           "EGM Mean RV": egmMean_interval,
                           "EGM STD RV": egmSTD_interval,
                           "EGM Skewness RV": egmSkew_interval,
                           "EGM Kurtosis RV": egmKurtosis_interval,
                           "R-R Interval RV": rr_interval}

            interval_stats.update(update_dict)

            perfusion_cut = np.zeros(2000) * np.nan
            perfusion_cut[:rr_interval] = perfusion_det.buffer[(start_local_time+PERFUSION_LAG):(end_local_time+PERFUSION_LAG)]
            mat.appendleft(perfusion_cut)

            perfusion_mat = np.array(mat)
            perfusion_consensus = np.nanmean(perfusion_mat, axis=0)
            perfusion_consensus_mask = np.isnan(np.sum(perfusion_mat, axis=0))

            perfusion_mean = np.nanmean(perfusion_consensus)
            perfusion_sd = np.nanstd(perfusion_consensus)

            perfusion_consensus[perfusion_consensus_mask] = np.nan

            perfusion_consensus_argmax = np.nanargmax(perfusion_consensus)
            perfusion_consensus_max = perfusion_consensus[perfusion_consensus_argmax]
            perfusion_consensus_min = perfusion_consensus[0]

            theta = 1000 * (perfusion_consensus_max - perfusion_consensus_min) / perfusion_consensus_argmax
            tmpgrad = math.degrees(math.atan(theta))
            bp_inteerval = int(tmpgrad * perfusion_consensus_argmax * 0.00750062)

            update_dict = {"BP": bp_inteerval,
                           "Current Perfusion Grad": tmpgrad,
                           "Per Mean": perfusion_mean,
                           "Per STD": perfusion_sd}

            interval_stats.update(update_dict)

            output.append(interval_stats)

            if not DEBUG:
                curve1.setData(x=np.arange(len(ecg_out)), y=ecg_out)
                curve3.setData(x=np.arange(len(raw)), y=raw)
                dot1.setData(x=peaks, y=ecg_out[peaks])

                curve4.setData(x=np.arange(len(mat[0])), y=mat[0])

                try:
                    curve5.setData(x=np.arange(len(mat[0])), y=mat[0])
                except Exception as e:
                    pass

                try:
                    curve6.setData(x=np.arange(len(mat[1])), y=mat[1])
                except Exception as e:
                    pass

                try:
                    curve7.setData(x=np.arange(len(mat[2])), y=mat[2])
                except Exception as e:
                    pass

                try:
                    curve8.setData(x=np.arange(len(mat[3])), y=mat[3])
                except Exception as e:
                    pass

                try:
                    curve9.setData(x=np.arange(len(mat[4])), y=mat[4])
                except Exception as e:
                    pass

                try:
                    curve10.setData(x=np.arange(len(mat[5])), y=mat[5])
                except Exception as e:
                    pass

                try:
                    curve11.setData(x=np.arange(len(perfusion_consensus)), y=perfusion_consensus)
                except Exception as e:
                    pass

                try:
                    curve2.setData(x=np.arange(len(perfusion_det.buffer)), y=perfusion_det.buffer)
                except Exception as e:
                    pass


                QtGui.QGuiApplication.processEvents()



        count = count + 1
        time.sleep(0.2)

    return output


if __name__ == '__main__':
    output = main(perfusion_path=pp, bp_path=bpp, electrogram_path=ee, period=1)
    output_pd = pd.DataFrame(output)
    output_pd.to_csv("paok.csv")
    print("Done")

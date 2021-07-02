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

# pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/qfin.txt"
# ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/boxb.txt"
# bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/vfi0017_vvi180_01_27_04_2021_152246_/boxb.txt"

pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ADA003_03_12_01_2021_154719_/plethh.txt"
ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ADA003_03_12_01_2021_154719_/plethg.txt"
ee1 = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ADA003_03_12_01_2021_154719_/plethg.txt"
bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ADA003_03_12_01_2021_154719_/BP.txt"


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
        # self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'same'))
        return out


class BPDetector:
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
        # self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'same'))
        return out


#
def main(electrogram_path, perfusion_path, bp_path, period,decision):
    ELECTROGRAM_PATH = electrogram_path
    PERFUSION_PATH = perfusion_path
    BP_PATH = bp_path

    electrogram = Data_Reader(data_path=ELECTROGRAM_PATH)
    perfusion = Data_Reader(data_path=PERFUSION_PATH)
    bpdata = Data_Reader(data_path=BP_PATH)

    electrogram_detector = ElectrogramDetector()
    # electrogram_detector1 = ElectrogramDetector()
    perfusion_det = PerfusionDetector()
    bp_det = BPDetector()
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

    mat = collections.deque(6 * [0], 6)
    mat_pks = collections.deque(50 * [0], 50)
    # print(mat_pks)
    ecg_peaks_total = []
    ecg_pks = []
    per_pks = []
    count = 0
    # count=1200
    hrb = 0
    count = 0
    cons = []
    output = pd.DataFrame()
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
             "Decision": decision}
    start = 0
    rr_interval = 1000
    finish = 400
    # tmp=[]
    while True:
        try:
            electrogram_detector.load_new_data(electrogram)
            perfusion_det.load_new_data(perfusion)
            bp_det.load_new_data(bpdata)

            per_out = (perfusion_det.detect_new_data())
            bp_out = (bp_det.detect_new_data())
            ecg_out, raw = electrogram_detector.detect_new_data()

            # ecg_out1, raw1 = electrogram_detector1.detect_new_data()
            a = np.arange(len(ecg_out))
            curve1.setData(x=a, y=ecg_out)
            c = np.arange(len(raw))
            curve3.setData(x=c, y=raw)

            bpln = np.arange(len(bp_out))
            avgbp = np.mean(bp_det.buffer) * 100
            maxbp = max(bp_det.buffer) * 100

            # EGM
            prom = statistics.mean(ecg_out)
            peaks, properties = find_peaks(ecg_out, prominence=prom, width=20)
            dot1.setData(x=peaks, y=ecg_out[peaks])

            # print(peaks)
            # opote afto pou kaneis einai oti kathe fora pairneis to index 0 apo tin lista kai to 1 an iparxei pou tha iparxei
            # print(mat_pks)
            for p in list(peaks):
                # global index
                mat_pks.appendleft((p + (count * 200)))
            ecg_peaks_total.append(peaks)
            egmMean = sum(ecg_out) / len(ecg_out)
            egmSTD = np.std(ecg_out)
            egmSkew = skew(ecg_out)
            egmKurtosis = kurtosis(ecg_out)
            local_index1 = [mat_pks[0]]
            if len(peaks) > 1:
                rr_interval = abs(int(peaks[0]) - int(peaks[1]))
                if rr_interval == 0:
                    rr_interval = 200
            bps = len(local_index1)
            print(rr_interval)
            bpmtmp = len(peaks)
            bpm = (60000 / rr_interval)
            stats.update(
                {"Max Actual BP": maxbp, "Mean Actual BP": avgbp, "Beats per Second (1000ms)": bps, "BPM": bpm,
                 "EGM Mean RV": egmMean,
                 "EGM STD RV": egmSTD, "EGM Skewness RV": egmSkew, "EGM Kurtosis RV": egmKurtosis,
                 "R-R Interval RV": rr_interval, "Decision": decision})

            # Perfusion
            # promper = statistics.mean(per_out)

            min_per_peaks, properties = find_peaks(per_out * -1, prominence=-0.5, width=30)
            min_per_peaks = np.array(min_per_peaks)
            per_peaks, properties = find_peaks(per_out, prominence=0.3, width=30)
            per_peaks = np.array(per_peaks)
            tmp = []

            dot3.setData(x=min_per_peaks, y=per_out[min_per_peaks])
            dot2.setData(x=per_peaks, y=per_out[per_peaks])
            # if len(peaks)!=0:
            #     print("opa")
            #     if len(peaks)==1 :
            #         print("opa1")
            #         start = peaks[0]
            #         tmp=per_out[start:]
            #         print(len(tmp))
            #     else:
            #         print("opa2")
            for i in range(len(peaks) - 1):
                start = peaks[i]
                finish = peaks[i + 1]
                dx = int(abs(start - finish))
                start = start + int(dx * 0.65)
                start = min_per_peaks[np.abs(min_per_peaks - start).argmin()]
                # if abs(peaks[i] - start)<200:
                #     print(start)
                #     print(min_per_peaks)
                #     np.delete(min_per_peaks,np.where(min_per_peaks==start))
                #     start = min_per_peaks[np.abs(min_per_peaks - (start)).argmin()]
                if start < 0:
                    start = 0
                print("start", start)

                finish = finish + int(dx * 0.65)
                finish = min_per_peaks[np.abs(min_per_peaks - (finish)).argmin()]
                # if abs(peaks[i + 1] - finish)<200:
                #     print(finish)
                #     print(min_per_peaks)
                #     np.delete(min_per_peaks,np.where(min_per_peaks==finish))
                #     finish = min_per_peaks[np.abs(min_per_peaks - (finish)).argmin()]
                print("finish", finish)
                # if len(per_peaks)!=0:
                #     if start < per_peaks[np.abs(per_peaks-(finish)).argmin()]< finish:
                tmp = per_out[start:finish]
                continue
                #         print("parto",len(tmp))
                # else:
                #     tmp=per_out[peaks[i]:peaks[i+1]]
                #     print("to pira",len(tmp))

            if len(tmp) != 0:
                curve4.setData(x=np.arange(len(tmp)), y=tmp)
                perMean = sum(tmp) / len(tmp)
                perSTD = np.std(tmp)
                perSkew = skew(tmp)
                perKurtosis = kurtosis(tmp)
                xmax = np.where(tmp == max(tmp))[0]
                if len(np.where(tmp == max(tmp))[0]) > 1:
                    xmax = np.where(tmp == max(tmp))[0][0]
                    # print("y1", xmax)
                print("tmp", tmp)
                print("tmp[xmax]", tmp[xmax])
                print("tmp[0]", tmp[0])
                theta = abs(tmp[xmax] - tmp[0]) / xmax * 1000
                tmpgrad = math.degrees(math.atan(theta))
                bp = int(tmpgrad * (xmax) * 0.00750062)
                stats.update({"BP": bp, "Current Perfusion Grad": tmpgrad, "Per Mean": perMean, "Per STD": perSTD,
                              "Per Skewness": perSkew, "Per Kurtosis": perKurtosis})

                if len(tmp) < 2000:
                    mis = len(tmp)
                    for t in range(2000 - mis):
                        tmp = list(tmp)
                        tmp.append(np.nan)
                        tmp = np.array(tmp)

                mat.appendleft(tmp)

                cons = np.nanmean((mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]), axis=0)
                # cons = pd.DataFrame(cons)

                # print("CONS",cons)
            if len(cons) != 0:
                # cons.dropna()
                # print(cons)
                # print(len(cons))
                nan_array = np.isnan(cons)
                not_nan_array = ~ nan_array
                cons = cons[not_nan_array]
                # for i in cons:
                #     if np.isnan(i):
                #         print("i", i)
                #         # np.delete(cons, i)
                print(cons)
                print(len(cons))
                # cons_per_peaks, properties = find_peaks(cons, prominence=0.3, width=30)
                # print("asdfgh",cons_per_peaks)
                # min_cons_per_peaks, properties = find_peaks(cons*-1, prominence=0.3, width=30)
                # print("poiuhgf",min_cons_per_peaks)
                ys = abs(cons[0] - cons[np.where(cons == max(cons))[0][0]])
                # print("ys", ys)
                xs = abs(np.where(cons == max(cons))[0][0])
                # print("xs", xs)
                tan = ys / xs * 1000
                # print("tan", tan)
                grad = math.degrees(math.atan(tan))
                percumMean = sum(cons) / len(cons)
                percumSTD = np.std(cons)
                print("00000000000000000000000000000000", percumSTD)
                percumSkew = skew(cons)
                percumKurtosis = kurtosis(cons)
                # print(grad)
                # print("mat",mat[0])
                # print("corrrr",stats.spearmanr(mat[0], mat[1]))
                per_mean = statistics.mean(per_out)
                stats.update({"Cumulative Perfusion Grad": grad, "Per Cum. Mean": percumMean,
                              "Per Cum. STD": percumSTD, "Per Cum. Skewness": percumSkew,
                              "Per Cum. Kurtosis": percumKurtosis})

                for m in range(len(mat)):
                    if not any(elem is 0 for elem in mat):
                        curve5.setData(x=np.arange(len(mat[0])), y=mat[0])
                        curve6.setData(x=np.arange(len(mat[1])), y=mat[1])
                        curve7.setData(x=np.arange(len(mat[2])), y=mat[2])
                        curve8.setData(x=np.arange(len(mat[3])), y=mat[3])
                        curve9.setData(x=np.arange(len(mat[4])), y=mat[4])
                        curve10.setData(x=np.arange(len(mat[5])), y=mat[5])

            if not isinstance(cons, float):
                curve11.setData(x=np.arange(len(cons)), y=cons)

            # time.sleep(1)
            table.setItem(1, 0, QTableWidgetItem(bpm))
            b = np.arange(len(per_out))
            curve2.setData(x=b, y=(per_out))

            QtGui.QGuiApplication.processEvents()
            output = output.append(stats, ignore_index=True)

            count = count + 1
        except:
            print("An exception occurred")
            return output

    # return output


# #
# if __name__ == '__main__':
#     #     # plt.plot(np.genfromtxt(PERFUSION_PATH, delimiter=','))
#     #     # plt.show()
#     output = main(perfusion_path=pp, bp_path=bpp, electrogram_path=ee, period=1)
#     output.to_csv("paok.csv")

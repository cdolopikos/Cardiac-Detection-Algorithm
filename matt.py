import collections
import math
import statistics

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

#
#
PERFUSION_PATH = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/qfin.txt"
ELECTROGRAM_PATH = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/boxb.txt"
BP_PATH = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/vfi0017_vvi180_01_27_04_2021_152246_/boxb.txt"




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
        self.buffer = np.zeros(1200)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, electrogram: Data_Reader):
        self.buffer[0:1000] = self.buffer[200:1200]
        self.buffer[1000:1200] = electrogram.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        out = butter_lowpass_filter(data=buffer, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out = butter_highpass_filter(data=out, cutoff=self.cutoff, fs=self.fs, order=self.order)
        out_mean = np.mean(out)
        out = np.abs(out - out_mean)
        out = np.convolve(out, np.ones(111, dtype=np.int), 'valid')
        raw = buffer[100:1100]
        return out, raw


class PerfusionDetector:
    def __init__(self):
        self.buffer = np.zeros(1200)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 150  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, perfusion: Data_Reader):
        self.buffer[0:1000] = self.buffer[200:1200]
        self.buffer[1000:1200] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'valid'))
        return out

class BPDetector:
    def __init__(self):
        self.buffer = np.zeros(1200)
        self.T = 1.2  # Sample Period
        self.fs = 1000.0  # sample rate, Hz
        self.cutoff = 150  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
        self.nyq = 0.5 * self.fs  # Nyquist Frequency
        self.order = 3  # sin wave can be approx represented as quadratic
        self.n = int(self.T * self.fs)
        self.detected_qrs = []

    def load_new_data(self, perfusion: Data_Reader):
        self.buffer[0:1000] = self.buffer[200:1200]
        self.buffer[1000:1200] = perfusion.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'valid'))
        return out

def main():
    electrogram = Data_Reader(data_path=ELECTROGRAM_PATH)
    perfusion = Data_Reader(data_path=PERFUSION_PATH)
    bpdata= Data_Reader(data_path=BP_PATH)

    electrogram_detector = ElectrogramDetector()
    perfusion_det = PerfusionDetector()
    bp_det= BPDetector()
    win = pg.GraphicsWindow()

    p1 = win.addPlot(row=1, col=0)
    p2 = win.addPlot(row=2, col=0)
    p3 = win.addPlot(row=0, col=0)
    p4 = win.addPlot(row=0, col=1)
    p5 = win.addPlot(row=1, col=1)
    p6 = win.addPlot(row=2, col=1)
    p7 = win.addPlot(row=0, col=2)

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
    curve12 = p7.plot()
    dot2 = p2.plot(pen=None, symbol="o")
    dot3 = p2.plot(pen=None, symbol="o")
    dot4 = p4.plot(pen=None, symbol="o")

    table = QTableWidget()
    table.setColumnCount(3)
    table.setRowCount(2)
    table.setHorizontalHeaderLabels(["BPM", "Perfusion-Gradient", "Decision"])

    mat = collections.deque(6 * [0], 6)
    ecg_pks = []
    per_pks = []
    count = 0
    # count=1200
    hrb = 0
    count = 0
    cons = []
    output = pd.DataFrame()
    stats = {"EGM Mean": 0,
             "EGM STD": 0,
             "EGM Skewness": 0,
             "EGM Kurtosis": 0,
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
    while True:
        try:
            count = count + 1
            electrogram_detector.load_new_data(electrogram)
            perfusion_det.load_new_data(perfusion)
            bp_det.load_new_data(bpdata)
            per_out = (perfusion_det.detect_new_data())
            bp_out = (bp_det.detect_new_data())
            ecg_out, raw = electrogram_detector.detect_new_data()
            a = np.arange(len(ecg_out))
            b = np.arange(len(per_out))
            bpln=np.arange(len(bp_out))
            curve12.setData(x=bpln,y=bp_out)
            avgbp=np.mean(bp_det.buffer)
            maxbp = max(bp_det.buffer)
            # EGM
            prom=statistics.mean(ecg_out)
            peaks, properties = find_peaks(ecg_out, prominence=prom, width=20)
            # if peaks == None:
            #     peaks, properties = find_peaks(ecg_out, prominence=0.2, width=20)
            egmMean=sum(ecg_out)/len(ecg_out)
            egmSTD = np.std(ecg_out)
            egmSkew = skew(ecg_out)
            egmKurtosis = kurtosis(ecg_out)
            bpm = len(peaks)
            stats.update({"!Max Actual BP":maxbp,"!Mean Actual BP": avgbp,"Beats per Second (1000ms)":bpm, "EGM Mean": egmMean, "EGM STD": egmSTD, "EGM Skewness": egmSkew, "EGM Kurtosis": egmKurtosis})
            # print(bpm,"BPM")

            # if peaks.size > 0:
            #     ecg_pks.append(peaks)
            # else:
            #     print(len(ecg_pks[len(ecg_pks) - 1]))
            #     print(ecg_pks)
            #     x = int(ecg_pks[len(ecg_pks) - 1][len(ecg_pks[len(ecg_pks) - 1]) - 1]) - 200
            #     y = ecg_out[x]
            #     dec = y * 0.5
            #     print(dec)
            #     peaks, properties = find_peaks(ecg_out, prominence=dec, width=20)
            #     if peaks.size > 0:
            #         ecg_pks.append(peaks)
            curve1.setData(x=a, y=ecg_out)
            c = np.arange(len(raw))
            curve3.setData(x=c, y=raw)
            dot1.setData(x=peaks, y=ecg_out[peaks])
            # Perfusion
            promper=statistics.mean(per_out)
            min_per_peaks, properties = find_peaks(per_out * -1, prominence=-promper, width=30)
            min_per_peaks = list(min_per_peaks)
            for i in range(len(min_per_peaks)):
                if i != 0:
                    df = abs(min_per_peaks[i] - min_per_peaks[i - 1])
                    if df < 100:
                        min_per_peaks.remove(min_per_peaks[i])
                        break
            per_peaks, properties = find_peaks(per_out, prominence=0.1, width=20)
            per_peaks = list(per_peaks)
            for i in range(len(per_peaks)):
                if i != 0:
                    df = abs(per_peaks[i] - per_peaks[i - 1])
                    if df < 100:
                        per_peaks.remove(per_peaks[i])
                        break
            if len(min_per_peaks) > 1:
                x2 = min_per_peaks[len(min_per_peaks) - 1]
                x1=min_per_peaks[len(min_per_peaks) - 2]
                # print("X2",x2)
                # print("X1",min_per_peaks[len(min_per_peaks) - 2])
                y2 = per_out[x2]
                tmp = per_out[x1:x2]
                # print(max(tmp), np.where(tmp == max(tmp)))
                # print(0, tmp[0])
                perMean = sum(tmp) / len(tmp)
                perSTD = np.std(tmp)
                perSkew = skew(tmp)
                perKurtosis = kurtosis(tmp)
                xmax=np.where(tmp == max(tmp))[0]
                # print("y1",xmax)
                # print("y2",y2)
                # print("y0",tmp[0])
                if len(np.where(tmp == max(tmp))[0])>1:
                    xmax=np.where(tmp == max(tmp))[0][0]
                    print("y1", xmax)
                theta=abs(tmp[xmax]-tmp[0])/xmax *1000
                tmpgrad= math.degrees(math.atan(theta))
                bp= tmpgrad*(xmax)*0.00750062
                print(bp)
                stats.update({"!BP":bp,"Current Perfusion Grad":tmpgrad, "Per Mean":perMean, "Per STD": perSTD, "Per Skewness": perSkew, "Per Kurtosis": perKurtosis})
                curve4.setData(x=np.arange(len(tmp)), y=tmp)

                if len(tmp) < 700:
                    mis = len(tmp)
                    for t in range(700 - mis):
                        tmp = list(tmp)
                        tmp.append(0)
                        tmp = np.array(tmp)

                mat.appendleft(tmp[:700])

                for m in range(len(mat)):
                    if isinstance(mat[m], int):
                        print("")
                    else:
                        print(len(mat[m]))
                        if m != len(mat)-1:
                            # print(cons)
                            if cons == []:
                                cons = (mat[m] + mat[m - 1])/2
                            else:
                                cons = (cons + mat[m] + mat[m - 1])/3
                    if m<len(mat)-1:
                        s1=mat[m]
                        s2=mat[m+1]
                        # print(isinstance(s1, int) or isinstance(s2, int))
                        # print(isinstance(s1, int))
                        # print(type(s1))
                        # print(isinstance(s2, int))
                        # print(type(s2))
                        if isinstance(s1, int) or isinstance(s2, int):
                            ds=0
                            sim=0
                        else:
                            ds=abs(sum(s1-s2))
                            sim=1-(ds/sum(s1))
                            stats.update({"Per_similarity":sim})
                        # print("sim",sim)

                for m in range(len(mat)):
                    if not any(elem is 0 for elem in mat):
                        curve5.setData(x=np.arange(len(mat[0])), y=mat[0])
                        curve6.setData(x=np.arange(len(mat[1])), y=mat[1])
                        curve7.setData(x=np.arange(len(mat[2])), y=mat[2])
                        curve8.setData(x=np.arange(len(mat[3])), y=mat[3])
                        curve9.setData(x=np.arange(len(mat[4])), y=mat[4])
                        curve10.setData(x=np.arange(len(mat[5])), y=mat[5])
                        # print("cons",len(cons))


                if cons!=[]:
                    ys = abs(cons[0] - cons[np.where(cons == max(cons))[0][0]])
                    # print("ys", ys)
                    xs = abs(np.where(cons == max(cons))[0][0])
                    # print("xs", xs)
                    tan = ys / xs * 1000
                    # print("tan", tan)
                    grad = math.degrees(math.atan(tan))
                    percumMean = sum(tmp) / len(tmp)
                    percumSTD = np.std(tmp)
                    percumSkew = skew(tmp)
                    percumKurtosis = kurtosis(tmp)
                    # print(grad)
                    # print("mat",mat[0])
                    # print("corrrr",stats.spearmanr(mat[0], mat[1]))
                    per_mean=statistics.mean(per_out)
                    stats.update({"Per Mean":per_mean,"Cumulative Perfusion Grad": grad, "Per Cum. Mean": percumMean, "Per Cum. STD": percumSTD, "Per Cum. Skewness": percumSkew, "Per Cum. Kurtosis": percumKurtosis})
                curve11.setData(x=np.arange(len(cons)), y=cons)

                # curve5.setData(x=np.arange((cons)), y=cons)

                # if x2 < i:
                #     print("x1", i)
                #     y1 = per_out[i]
                #     print("x2", x2)
                #     x1x2 = abs(x2 - i)
                #     print("x1x2", x1x2)
                #     # print(x1x2)
                #     print("y1", y1)
                #     print("y2", y2)
                #     y1y2 = abs(y1 - y2) * 1000
                #     print(y1y2)
                #     # print(y1y2)
                #     gradientA = x1x2 / y1y2
                #     gradientB = y1y2 / x1x2
                #     print(gradientA)
                #     print(math.degrees(math.atan(gradientA)))
                #     alpha = math.degrees(math.atan(gradientA))
                #     print(gradientB)
                #     print(math.degrees(math.atan(gradientB)))
                #     beta = math.degrees(math.atan(gradientB))
                #     alpha_beta = alpha - beta
                #     print("alpha_beta", alpha_beta)
                #     if alpha_beta > 0:
                #         print("DANGER")
            # if per_peaks.size > 0:
            #     per_pks.append(per_peaks)
            #     if len(per_pks) >2:
            #         dt=per_out[per_pks[len(per_pks) - 2][len(per_pks[len(per_pks) - 2])-1]: per_pks[len(per_pks) - 1][len(per_pks[len(per_pks) - 1])-1]:]
            #         # print(dt)
            #         if len(dt)>0:
            #             mean = sum(dt) / len(dt)
            #             std = np.sqrt(math.pow((sum(dt) - mean), 2) / len(dt))
            #             cov = np.cov(dt)
            #             var = np.var(dt)
            #             print("mean", mean)
            #             print("std", std)
            #             print("cov", cov)
            #             print("var", var)

            table.setItem(1, 0, QTableWidgetItem(bpm))
            # print(stats)
            curve2.setData(x=b, y=(per_out))
            dot2.setData(x=per_peaks, y=per_out[per_peaks])
            dot3.setData(x=min_per_peaks, y=per_out[min_per_peaks])

            QtGui.QGuiApplication.processEvents()
            if stats.get("Perfusion Grad") != 0.0:
                # output=pd.DataFrame.from_dict(stats)
                output=output.append(stats, ignore_index=True)
                # print(output)
                # output=output.loc[(output!=0).any(axis=1)]
                # print(output.get("Perfusion Grad"))
                output.to_csv("out.csv")

            time.sleep(1)
            # if len(per_pks)>2:
            #     print(per_pks[len(per_pks)-2])
            #     print(len(per_pks)-1)
            #     dt=per_out[len(per_pks)-2:len(per_pks)-1]
        #     #     print(dt)
        except:
            print("An exception occurred")
            sys.exit()


if __name__ == '__main__':
    # plt.plot(np.genfromtxt(PERFUSION_PATH, delimiter=','))
    # plt.show()
    main()

# dtime milliseconds of peaks
# log of perfusion (peak to peak height),
# blood pressure
# a-b (+1)
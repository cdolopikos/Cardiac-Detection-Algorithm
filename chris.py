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

pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/vfi0017_vvi180_01_27_04_2021_152246_/qfin.txt"
ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/vfi0017_vvi180_01_27_04_2021_152246_/BP.txt"
bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/vfi0017_vvi180_01_27_04_2021_152246_/boxb.txt"


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
        out = np.convolve(out, np.ones(111, dtype=np.int), 'valid')
        raw = buffer[100:1100]
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
        self.buffer[0:1800] = self.buffer[200:2000]
        self.buffer[1800:2000] = perfusion.get_next_data(amount=200)

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

    def load_new_data(self, bp: Data_Reader):
        self.buffer[0:1000] = self.buffer[200:1200]
        self.buffer[1000:1200] = bp.get_next_data(amount=200)

    def detect_new_data(self):
        buffer = self.buffer
        buffer = (buffer - min(buffer)) / (max(buffer) - min(buffer))
        window_size = 100
        window = np.ones(window_size) / float(window_size)
        out = np.sqrt(np.convolve(buffer, window, 'valid'))
        return out


#
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
    win = pg.GraphicsWindow()

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
        # try:
        electrogram_detector.load_new_data(electrogram)
        perfusion_det.load_new_data(perfusion)
        bp_det.load_new_data(bpdata)
        per_out = (perfusion_det.detect_new_data())
        bp_out = (bp_det.detect_new_data())
        ecg_out, raw = electrogram_detector.detect_new_data()
        a = np.arange(len(ecg_out))

        bpln = np.arange(len(bp_out))
        avgbp = np.mean(bp_det.buffer)
        maxbp = max(bp_det.buffer)
        # EGM
        prom = statistics.mean(ecg_out)
        peaks, properties = find_peaks(ecg_out, prominence=prom, width=20)
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
        bpm = len(peaks)
        stats.update(
            {"!Max Actual BP": maxbp, "!Mean Actual BP": avgbp, "Beats per Second (1000ms)": bpm, "EGM Mean": egmMean,
             "EGM STD": egmSTD, "EGM Skewness": egmSkew, "EGM Kurtosis": egmKurtosis})
        curve1.setData(x=a, y=ecg_out)
        c = np.arange(len(raw))
        curve3.setData(x=c, y=raw)
        dot1.setData(x=peaks, y=ecg_out[peaks])
        local_index1 = mat_pks[0]
        # Perfusion
        promper = statistics.mean(per_out)
        # per_out1=per_out[local_index1:]
        min_per_peaks, properties = find_peaks(per_out * -1, prominence=-promper, width=30)
        # print(peaks)

        if mat_pks[1] != 0:
            # print(mat_pks)
            if mat_pks[0] != mat_pks[1]:
                if mat_pks[1] - mat_pks[0] > 200:
                    start = mat_pks[1] - (count * 200)
                    finish = mat_pks[0] - (count * 200)

            else:
                start = mat_pks[0]
                finish = 1000
            # elif len(peaks) == 1:
            #     start = mat_pks[0]
            #     finish = 2000

            # # Perfusion
            # promper = statistics.mean(per_out)
            # # per_out1=per_out[local_index1:]
            # min_per_peaks, properties = find_peaks(per_out * -1, prominence=-promper, width=30)
            #         min_per_peaks = list(min_per_peaks)
            #         # print("min_per_peaks",min_per_peaks)
            #         for i in range(len(min_per_peaks)):
            #             if i != 0:
            #                 dif = abs(min_per_peaks[i] - min_per_peaks[i - 1])
            #                 if dif < 100:
            #                     min_per_peaks.remove(min_per_peaks[i])
            #                     break
            #         # print("min_per_peaks",min_per_peaks)
            #         per_peaks, properties = find_peaks(per_out, prominence=0.1, width=20)
            #         per_peaks = list(per_peaks)
            #         # print("per_peaks",per_peaks)
            #         for i in range(len(per_peaks)):
            #             if (local_index1+200)<i:
            #                 if i != 0:
            #                     df = abs(per_peaks[i] - per_peaks[i - 1])
            #                     if df < 100:
            #                         per_peaks.remove(per_peaks[i])
            #                         break
            #         # print("per_peaks", per_peaks)
            #         tmpmn=[]
            #         tmpmx=[]
            #         for mn in min_per_peaks:
            #             diff=abs(local_index1-mn)
            #             tmpmn.append(diff)
            #         # print("tmpmn",tmpmn)
            #         if tmpmn!=[]:
            #             mincand=np.where(tmpmn == min(tmpmn))[0]
            #             # print("mincand",mincand)
            #             if len(mincand)>1:
            #                 min1 = tmpmn[int(np.where(tmpmn == min(tmpmn))[0][0])]
            #             else:
            #                 min1=tmpmn[mincand[0]]
            #             # print("min",min1)
            #         for mx in per_peaks:
            #             diff = abs(local_index1-mx)
            #             tmpmx.append(diff)
            #         if tmpmx!=[]:
            #             max1 = per_peaks[int(np.where(tmpmx == min(tmpmx))[0])]
            #             # print("max",max1)
            #         if len(min_per_peaks) > 1:
            #             diff_per= abs(per_out[min1]-per_out[max1])
            #             if diff_per>(promper/10):
            #                 x2 = min_per_peaks[len(min_per_peaks) - 1]
            #                 x1 = min_per_peaks[len(min_per_peaks) - 2]
            #                 print("X2",x2)
            #                 # print("X1",min_per_peaks[len(min_per_peaks) - 2])
            #                 y2 = per_out[x2]
            #
            #                 if x1 < max1 < x2:
            tmp = per_out[start:finish]
            #                     # tmpmn.sort()
            #                     # x2 =min_per_peaks[len(min_per_peaks) - 1]
            #                     # x1 = int(min1)
            #                     # print("x1",x1)
            #                     # print("X2",x2)
            #                     # # print("X1",min_per_peaks[len(min_per_peaks) - 2])
            #                     # y2 = per_out[x2]
            #                     # tmp = per_out[x1:x2]
            #                     # print(max(tmp), np.where(tmp == max(tmp)))
            #                     # print(0, tmp[0])
            if len(tmp) != 0:
                perMean = sum(tmp) / len(tmp)
                perSTD = np.std(tmp)
                perSkew = skew(tmp)
                perKurtosis = kurtosis(tmp)
                xmax = np.where(tmp == max(tmp))[0]
                if len(np.where(tmp == max(tmp))[0]) > 1:
                    xmax = np.where(tmp == max(tmp))[0][0]
                    # print("y1", xmax)
                theta = abs(tmp[xmax] - tmp[0]) / xmax * 1000
                tmpgrad = math.degrees(math.atan(theta))
                bp = tmpgrad * (xmax) * 0.00750062
                # print(bp)
                stats.update({"!BP": bp, "Current Perfusion Grad": tmpgrad, "Per Mean": perMean, "Per STD": perSTD,
                              "Per Skewness": perSkew, "Per Kurtosis": perKurtosis})
                curve4.setData(x=np.arange(len(tmp)), y=tmp)
                # print(np.arange(len(tmp)))
                # print(tmp[x1])
                # dot4.setData(x=np.arange(len(tmp)), y=tmp[x1])

                if len(tmp) < 2000:
                    mis = len(tmp)
                    for t in range(2000 - mis):
                        tmp = list(tmp)
                        tmp.append(np.nan)
                        tmp = np.array(tmp)

                mat.appendleft(tmp[:2000])

                cons = np.nanmean((mat[0], mat[1], mat[2], mat[3], mat[4], mat[5]), axis=0)
            # print("CONS",cons)
            # for m in range(len(mat)):
            #     if isinstance(mat[m], int):
            #         print("")
            #     else:
            #         # print(len(mat[m]))
            #         if m != len(mat) - 1:
            #             tmp1 = np.array(mat[m])
            #             tmp2 = np.array(mat[m - 1])
            #             # print(cons)
            #             if cons == []:
            #                 if len(tmp1)<len(tmp2):
            #                     sm1=(tmp1 + tmp2[:len(tmp1)])
            #                     tmp2[:len(tmp1)]=sm1
            #                     cons = ((tmp2) / 2)
            #                 elif len(tmp1)>len(tmp2):
            #                     sm1 = (tmp2 + tmp1[:len(tmp2)])
            #                     tmp1[:len(tmp2)] = sm1
            #                     cons = ((tmp1) / 2)
            #                 else:
            #                     cons= (tmp1+tmp2)/2
            #
            #             else:
            #                 if len(tmp1)<len(tmp2):
            #                     sm1=(tmp1 + tmp2[:len(tmp1)])
            #                     tmp2[:len(tmp1)]=sm1
            #                     if len(cons)<len(tmp2):
            #                         sm2=(cons+tmp2[:len(cons)])
            #                     elif len(cons)>len(tmp2):
            #                         sm2 = (tmp2+cons[len(tmp2)])
            #                     else:
            #                         sm2=tmp2+cons
            #                 elif len(tmp1)<len(tmp2):
            #
            #                     cons = ((tmp2) / 2)
            #                 cons = (cons + tmp1 + tmp2) / 3

            # if m < len(mat) - 1:
            #     s1 = mat[m]
            #     s2 = mat[m + 1]
            #     # print(isinstance(s1, int) or isinstance(s2, int))
            #     # print(isinstance(s1, int))
            #     # print(type(s1))
            #     # print(isinstance(s2, int))
            #     # print(type(s2))
            #     if isinstance(s1, int) or isinstance(s2, int):
            #         ds = 0
            #         sim = 0
            #     else:
            #         ds = abs(np.mean(np.array([s1,s2]),axis=0))
            #         sim = 1 - (ds / sum(s1))
            #         stats.update({"Per_similarity": sim})
            # # print("sim",sim)
            #
            for m in range(len(mat)):
                if not any(elem is 0 for elem in mat):
                    curve5.setData(x=np.arange(len(mat[0])), y=mat[0])
                    curve6.setData(x=np.arange(len(mat[1])), y=mat[1])
                    curve7.setData(x=np.arange(len(mat[2])), y=mat[2])
                    curve8.setData(x=np.arange(len(mat[3])), y=mat[3])
                    curve9.setData(x=np.arange(len(mat[4])), y=mat[4])
                    curve10.setData(x=np.arange(len(mat[5])), y=mat[5])
            #         # print("cons",len(cons))
            #
            if cons != []:
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
                per_mean = statistics.mean(per_out)
                stats.update({"Per Mean": per_mean, "Cumulative Perfusion Grad": grad, "Per Cum. Mean": percumMean,
                              "Per Cum. STD": percumSTD, "Per Cum. Skewness": percumSkew,
                              "Per Cum. Kurtosis": percumKurtosis})
            if isinstance(cons, float):
                print("")
            else:
                curve11.setData(x=np.arange(len(cons)), y=cons)
        #
        #                 # curve5.setData(x=np.arange((cons)), y=cons)
        #
        #                 # if x2 < i:
        #                 #     print("x1", i)
        #                 #     y1 = per_out[i]
        #                 #     print("x2", x2)
        #                 #     x1x2 = abs(x2 - i)
        #                 #     print("x1x2", x1x2)
        #                 #     # print(x1x2)
        #                 #     print("y1", y1)
        #                 #     print("y2", y2)
        #                 #     y1y2 = abs(y1 - y2) * 1000
        #                 #     print(y1y2)
        #                 #     # print(y1y2)
        #                 #     gradientA = x1x2 / y1y2
        #                 #     gradientB = y1y2 / x1x2
        #                 #     print(gradientA)
        #                 #     print(math.degrees(math.atan(gradientA)))
        #                 #     alpha = math.degrees(math.atan(gradientA))
        #                 #     print(gradientB)
        #                 #     print(math.degrees(math.atan(gradientB)))
        #                 #     beta = math.degrees(math.atan(gradientB))
        #                 #     alpha_beta = alpha - beta
        #                 #     print("alpha_beta", alpha_beta)
        #                 #     if alpha_beta > 0:
        #                 #         print("DANGER")
        #                 # if per_peaks.size > 0:
        #                 #     per_pks.append(per_peaks)
        #                 #     if len(per_pks) >2:
        #                 #         dt=per_out[per_pks[len(per_pks) - 2][len(per_pks[len(per_pks) - 2])-1]: per_pks[len(per_pks) - 1][len(per_pks[len(per_pks) - 1])-1]:]
        #                 #         # print(dt)
        #                 #         if len(dt)>0:
        #                 #             mean = sum(dt) / len(dt)
        #                 #             std = np.sqrt(math.pow((sum(dt) - mean), 2) / len(dt))
        #                 #             cov = np.cov(dt)
        #                 #             var = np.var(dt)
        #                 #             print("mean", mean)
        #                 #             print("std", std)
        #                 #             print("cov", cov)
        #                 #             print("var", var)
        time.sleep(1)
        table.setItem(1, 0, QTableWidgetItem(bpm))
        # print(stats)
        b = np.arange(len(per_out))
        curve2.setData(x=b, y=(per_out))
        # dot2.setData(x=per_peaks, y=per_out[per_peaks])
        # dot3.setData(x=min_per_peaks, y=per_out[min_per_peaks])

        QtGui.QGuiApplication.processEvents()
        # if stats.get("Perfusion Grad") != 0.0:
        # output=pd.DataFrame.from_dict(stats)
        output = output.append(stats, ignore_index=True)
        output.to_csv("paok.csv")
        count = count + 1
    # except:
    #     print("An exception occurred")
    #     break

    return output
    # print(output)
    # output=output.loc[(output!=0).any(axis=1)]
    # print(output.get("Perfusion Grad"))

    # if len(per_pks)>2:
    #     print(per_pks[len(per_pks)-2])
    #     print(len(per_pks)-1)
    #     dt=per_out[len(per_pks)-2:len(per_pks)-1]
    #     #     print(dt)
    # except:
    #     print("An exception occurred")
    #     # sys.exit()

    # break


#
if __name__ == '__main__':
    #     # plt.plot(np.genfromtxt(PERFUSION_PATH, delimiter=','))
    #     # plt.show()
    main(perfusion_path=pp, bp_path=bpp, electrogram_path=ee, period=1)

import numpy
import numpy as np
import data_reader
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, lfilter  # Filter requirements.
T = 1.0         # Sample Period
fs = 1000.0       # sample rate, Hz
lowcut = 500.0
highcut = 750.0

cutoff = 20      # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 3       # sin wave can be approx represented as quadratic
n = int(T * fs) # total number of samples

# def butter_lowpass_filter(data, cutoff, fs, order):
#     normal_cutoff = cutoff / nyq
#     # Get the filter coefficients
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     y = filtfilt(b, a, data)
#     return y

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

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y



data=data_reader.ecg
data=data[0:1000]
# data=data
# y=butter_highpass_filter(data, cutoff, fs, order)
y=butter_bandpass_filter(data, lowcut, highcut, fs)
# a=butter_lowpass_filter(y, cutoff, fs, order)


# y = butter_lowpass_filter(data, cutoff, fs, order)
# a=butter_lowpass_filter(y, cutoff, fs, order)
# for i in range(3):
#     a=butter_lowpass_filter(a, cutoff, fs, order)
#
# y1 = butter_highpass_filter(data, cutoff, fs, order)
# a1=butter_highpass_filter(y, cutoff, fs, order)



# peaks, _ = find_peaks(y,height=0)
# peaks1, _ = find_peaks(y1,height=0)
# peakss, _ = find_peaks(a,height=0)
# peakss1, _ = find_peaks(a1,height=0)
#
#
# # for i,z in zip(data,a):
#     # print(i)
#     # print(z)
plt.plot(data)
plt.plot(y)
# plt.plot(y1)
# plt.plot(a)
# plt.plot(a1)
# plt.plot(peaks, y[peaks], "x")
# plt.plot(peakss, a[peakss], "o")
plt.show()

# print("mono", peaks)
# print("diplo", peakss)
# print(a[peaks])
# ms=600
# suma=0
#
# def findPeak(mat,pk):
#     temp=[]
#     # print(mat)
#     ftistarxidiasoukaitrexa=0
#     for i in range(0,len(mat)):
#         # print(mat[i])
#         if float(mat[i])>ftistarxidiasoukaitrexa:
#             ftistarxidiasoukaitrexa = mat[i]
#     # print("re", ftistarxidiasoukaitrexa)
#     tilestr = np.where(mat == ftistarxidiasoukaitrexa)
#     # print(pk[tilestr[0]])
#     gtxm=pk[tilestr[0]]
#     print("gtxm",gtxm)
#     return ftistarxidiasoukaitrexa,gtxm
#
# def sld():
#     tmp=[]
#
#     for i in range(0,len(a),1200):
#         print(i)
#         x=(a[i:i+1200])
#         tmp.append(i)
#         # if i >= 1200:
#         peaks, _ = find_peaks(a[i:i+1200], height=0)
#         # print(peaks)
#         if peaks.size !=0:
#             # plt.plot(x)
#             # plt.plot(peaks, x[peaks], "x")
#             # plt.show()
#             # print(tsapou)
#             # print(tsapou[0])
#             mat = numpy.array(x[peaks])
#             f,indefix=findPeak(mat,peaks)
#             # print(np.where(peaks == f))
#             qtmp=x[int(indefix)-600:int(indefix)]
#             # todo: find the lowest between th previous two--> ara tha breis pali ta peaks
#             #  i kapoio for loop pou aplws checkarei poia apo ta ipoloipa exei tin mikroteri diafora see auto to miso
#             #  kai meta anamesa sto peak(R) kai sto deftero peak(P) orizeis to Q pou einai h xamiloteri timi
#             tempo = []
#             pk, _ = find_peaks(qtmp, height=0)
#             for j in pk:
#                 diff = abs(j-int(indefix))
#                 tempo.append(diff)
#             q=0
#             if len(tempo) != 0:
#                 qlim=tempo.index(min(tempo))
#                 qrange=x[qlim:int(indefix)]
#                 q=(min(qrange))
#                 q=numpy.where(qtmp==q)
#                 print("q",(q[0]))
#                 plt.plot(x)
#                 plt.plot(peaks, x[peaks], "x")
#                 plt.plot(q[0], x[q[0]], "o")
#                 plt.show()
#                 print("qtmp",qtmp)
#                 # stmp=a[peaks[ind[0]]:i]
#     # print(mat.max())
#         # for j in x[peaks]:
#         #     if j
#
#         # print(max(tmp1))
#                 # for j in tmp:
#             #     for x in
#             #     if j
#
#
#
# sld()


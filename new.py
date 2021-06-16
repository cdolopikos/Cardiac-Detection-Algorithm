import math

import numpy
import numpy as np
import data_reader
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, lfilter  # Filter requirements.

T = 1.0  # Sample Period
fs = 1000.0  # sample rate, Hz
# lowcut = 100.0
# highcut = 250.0

cutoff = 20  # desired cutoff frequency of the filter, Hz ,      slightly higher than actual 1.2 Hz
nyq = 0.5 * fs  # Nyquist Frequency
order = 3  # sin wave can be approx represented as quadratic
n = int(T * fs)  # total number of samples
data = data_reader.ecg



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


def second_max(pk):
    # print(pk)
    count = 0
    m1 = m2 = float('-inf')
    for x in pk:
        count += 1
        if x > m2:
            if x >= m1:
                m1, m2 = x, m1
            else:
                m2 = x
    return m2


diffc = 0


# def diff_low_hig(lwmx,higmx,lwmx2,highmx2):
# diffA = abs(lwmx / higmx)
# print("xyn", diffA)
# # diffA=abs(lwmx)
# # print(diffA)
# diffB = abs(lwmx2 / highmx2)
# print("xyn12", diffB)
# # diffB=abs(lwmx2)
# # print(diffB)
# # if diffA>diffB:
# #     angle = (diffA / diffB)
# #     print(angle)
# # else:
# angle = float(diffA) / float(diffB)
# diffc = float(diffA) - float(diffB)
# print(angle, "!")
# print(diffc, "paok")
# return angle
def decay(t):
    dec=math.pow(2,((-t/(50 * math.log(10)))))
    return dec


def diff_low_hig(pk, pk2, lwdiffsignal, lwmx, lwmx2):
    #     sum of list elements between index x and y
    # print(pk)
    # print(pk2)
    # print(lwmx2,"kjjgffs")
    # print(lwdiffsignal)
    sum = 0
    for i in range(pk, pk2):
        sum = sum + abs(lwdiffsignal[i])
    lwNoise = sum / abs(pk - pk2)
    # print(lwNoise)
    thresholdLow = lwNoise + ((1 / 4) * (lwmx - lwNoise))
    # if pk > pk2:
    #     dec=pk+decay(pk2)*pk
    #     if dec > pk2:
    #         idk=True
    # else:
    #     dec=pk2+decay(pk)*pk2
    #     if dec>pk:
    #         idk=True
    # tmpdf=abs(pk-pk2)
    # print("pk",pk)
    # print("pk2",pk2)
    # print("dec",dec)
    # print("th",thresholdLow)
    if lwmx2 >= thresholdLow:
        print("true")
        return True, thresholdLow
    else:
        print("false")
        return False, thresholdLow


count = 0
pks = list()
xs = list()
for i in range(0, len(data), 6000):
    print(i)
    data1 = data[i:i + 6000]

    low_pass_filteredSignal = butter_lowpass_filter(data1, cutoff, fs, order)
    # print("low-pas",low_pass_filteredSignal)
    high_pass_filteredSignal = butter_highpass_filter(data1, cutoff, fs, order)
    differentiated_signal = numpy.diff(low_pass_filteredSignal)
    fig = plt.figure(figsize=(9, 4), dpi=150)
    plt.plot(differentiated_signal)
    title = "EGM 6000ms"
    plt.title(title)
    plt.show()
    loc = title
    fig.tight_layout()
    fig.savefig(loc, dpi=200)
    # print("diff-signal", differentiated_signal)
    differentiated_signal1 = numpy.diff(high_pass_filteredSignal)
    peaks, _ = find_peaks(differentiated_signal)
    # print("peaks",peaks)
    peaks1, _ = find_peaks(differentiated_signal1, height=0)
    # print(max(differentiated_signal[peaks]))
    mx = max(differentiated_signal[peaks])
    mh = max(differentiated_signal1[peaks1])
    # print(mx)
    # print(differentiated_signal[peaks])
    dfpk = list(differentiated_signal[peaks])
    # print(len(dfpk))
    # print(peaks)
    pk = list(peaks)
    # print(pk)
    idn = dfpk.index(mx)
    # print(dfpk.index(mx))
    # print(list(pk).__getitem__(idn))
    my = second_max(differentiated_signal[pk])
    # print(differentiated_signal[pk])
    # idn1 = dfpk.index(my)
    # liner1 = list(pk).__getitem__(idn1)
    liner = list(pk).__getitem__(idn)
    liner1=0
    liner2=0
    # ela, th = diff_low_hig(liner, liner1, differentiated_signal, mx, my)
    # print(my)
    pks.append(mx)
    xs.append(idn)
    # my = second_max(differentiated_signal[pk])
    idn1 = dfpk.index(my)
    idn2 = 0
    tmppk = list(peaks)
    liner1 = list(peaks).__getitem__(idn1)
    liners=list()
    liners.append(liner)
    # dec=decay(liner)
    # print(decay(liner1))
    # plt.plot(decay(liner1))
    # plt.show()
    for j in range(len(peaks)):
        if j == 0:
            ela, th = diff_low_hig(liner, liner1, differentiated_signal, mx, my)
            if ela:
                liners.append(liner1)
                pks.append(idn1)
                tmppk.remove(liner)
                idn2 = idn1
            else:
                break
        else:
            # print("asdfg",j, " tmpj", len(tmppk))
            my = second_max(differentiated_signal[tmppk])
            ela, th = diff_low_hig(liner, liner1, differentiated_signal, mx, my)
            if ela:
                idn3 = dfpk.index(my)
                liner2 = list(peaks).__getitem__(idn3)
                liners.append(liner2)
                pks.append(my)
                tmppk.remove(liner1)
                idn1 = idn2
                idn2 = idn3
                liner1=liner2
            else:
                break
    fig = plt.figure(figsize=(9, 4), dpi=350)
    # plt.plot(high_pass_filteredSignal)
    # plt.plot(data1)
    plt.plot(differentiated_signal)
    # plt.plot(differentiated_signal1)
    # plt.plot(y1)
    # plt.plot(a)
    # plt.plot(a1)
    plt.plot(liners, differentiated_signal[liners], "x")
    # plt.plot(peaks, data[peaks], "x")
    # plt.plot(peakss, a[peakss], "o")
    for z in liners:
        plt.axvline(x=z)
    # plt.axvline(x=liner1)
    # plt.axhline(y=th,color='r')
    plt.axhline(y=differentiated_signal[second_max(peaks)])
    title = "Algo-EGM 6000ms"
    # title = str(i) +' '
    plt.title(title)
    plt.show()
    # loc='det'+str(i)+''
    loc=title
    fig.tight_layout()
    fig.savefig(loc, dpi=200)

        # else:
#
#     if ela:
#         # print("a",second_max(pk))
#         # plt.plot(data1)
#         # print("II", i)
#         count = count + 1
#         fig = plt.figure(figsize=(6, 3), dpi=150)
#         # plt.plot(high_pass_filteredSignal)
#         # plt.plot(data1)
#         plt.plot(differentiated_signal)
#         # plt.plot(differentiated_signal1)
#         # plt.plot(y1)
#         # plt.plot(a)
#         # plt.plot(a1)
#         plt.plot(peaks, differentiated_signal[peaks], "x")
#         # plt.plot(peaks, data[peaks], "x")
#         # plt.plot(peakss, a[peakss], "o")
#         plt.axvline(x=liner)
#         plt.axvline(x=liner1)
#         plt.axhline(y=th,color='r')
#         plt.axhline(y=differentiated_signal[second_max(peaks)])
#         title = str(i) +' '+ str(ela)
#         plt.title(title)
#         plt.show()
#         loc='det'+str(i)+''
#         fig.tight_layout()
#         fig.savefig(loc, dpi=200)
#     else:
#         fig = plt.figure(figsize=(6, 3), dpi=150)
#         # plt.plot(high_pass_filteredSignal)
#         # plt.plot(data1)
#         plt.plot(differentiated_signal)
#         # plt.plot(differentiated_signal1)
#         # plt.plot(y1)
#         # plt.plot(a)
#         # plt.plot(a1)
#         plt.plot(peaks, differentiated_signal[peaks], "x")
#         # plt.plot(peaks, data[peaks], "x")
#         # plt.plot(peakss, a[peakss], "o")
#         plt.axvline(x=liner)
#         # plt.axvline(x=liner1)
#         plt.axhline(y=th,color='r')
#         y=differentiated_signal[second_max(peaks)]
#         title = str(i) + ' ' + str(ela) + ' '+str(y)
#         plt.title(title)
#         plt.show()
#         loc = 'det' + str(i) + ''
#         fig.tight_layout()
#         fig.savefig(loc, dpi=200)
# # # # diff_low_hig(mx,mh, differentiated_signal[second_max(peaks)],differentiated_signal[second_max(peaks1)])
#

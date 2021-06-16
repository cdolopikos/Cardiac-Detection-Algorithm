import numpy

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
# data=data[0:1200]

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


def diff_low_hig(pk, pk2, lwdiffsignal, lwmx, lwmx2):
    sum = 0
    for i in range(pk, pk2):
        sum = sum + abs(lwdiffsignal[i])
    lwNoise = sum / abs(pk - pk2)
    print(lwNoise)
    thresholdLow = lwNoise + ((1 / 4) * (lwmx - lwNoise))
    if lwmx2 >= thresholdLow:
        return True, thresholdLow
    else:
        return False, thresholdLow

# def buff(inpt, turn, prev):
#     bf = []
#     if turn == 0:
#         for i in range(len(inpt)):
#             if i < 1200:
#                 bf.append(inpt[i])
#                 print(i)
#         prev=bf[700:1200]
#         print(len(prev))
#         bf = bf + prev
#     else:
#         for i in range(len(inpt)):
#             if i < 1200:
#                 bf.append(inpt[i])
#                 bf=bf+prev
#     return bf, prev

    # else:
    #     print(prev)
    #     prev = bf[700:1200]
    #     bf = bf + prev
    # print(len(bf))
    # return bf

def find_r_peak(filtered_signal):
    r_peak=max(filtered_signal[peaks])
    r_peak_index=list(numpy.where(filtered_signal == numpy.amax(filtered_signal)))
    return r_peak,r_peak_index


def lin_thres(filtered_signal,r_pk):
    sum=0
    for i in filtered_signal:
        sum = sum + i
    signal_noise= sum/len(filtered_signal) + 0.45*r_pk
    return signal_noise

def find_rest_pks(filtered_signal,linear_thres):
    peaks, _ = find_peaks(filtered_signal,height=linear_thres)
    return peaks


# def decay(pk1, pk2ind, pk2,linear_threshold):
#     n = pk1 - linear_threshold + numpy.exp((-pk2) * (r_peak_init))
#     print(n,"    ", pk2)
#     return pk2>n
# def find_other_peak(others):
#     max=
count = 0
# pks = list()
xs = list()
for i in range(0, len(data), 1200):
    pks = list()
    if i == 0:
        data1 = data[i:i + 1200]
    else:
        data1 = data[i:i + 1200]
    # print(i)
    low_pass_filteredSignal = butter_lowpass_filter(data1, cutoff, fs, order)
    high_pass_filteredSignal = butter_highpass_filter(low_pass_filteredSignal, cutoff, fs, order)
    out = numpy.abs(high_pass_filteredSignal)
    peaks, _ = find_peaks(out)
    r_peak_init, r_peak_index =list(find_r_peak(out))
    print(r_peak_index[0])
    pks.append(int(r_peak_index[0]))
    # find all values above a threshold
    linear_threshold=lin_thres(out,r_peak_init)
    tmp=list(find_rest_pks(out,linear_threshold))
    # from them find the second highest
    tmp.remove(r_peak_index)
    tmp_len=len(tmp)

    for j in range(0,tmp_len):
        if len(tmp) > 0:
            second= list(numpy.where(out == numpy.amax(out[tmp])))

            if second[0]>r_peak_index:

                dec_threshold= r_peak_init *0.65 +numpy.exp((-(second[0]))*(r_peak_init))

            else:
                dec_threshold= r_peak_init+numpy.exp(((second[0]))*(r_peak_init))

            if high_pass_filteredSignal[second[0]]>dec_threshold and r_peak_init>dec_threshold:
                print(i)
                print("pass")
                print(r_peak_index, r_peak_init)
                print(second[0], high_pass_filteredSignal[second[0]])
                print(dec_threshold)
                r_peak_2_indx=second[0]
                print(r_peak_2_indx[0])
                pks.append(r_peak_2_indx[0])
                r_peak_init=r_peak_2_indx[0]
                tmp.remove(second[0])
            else:
                tmp.remove(second[0])
    # print(second)
    # for i in range(0,1200):
    #     if i<380:
    #         xs.append(0)
    #     else:
    #         print(i)
    #         n = r_peak_init*0.65 + numpy.exp((-i) *(r_peak_init))
    #         xs.append(n)
    # poutsa=decay(r_peak_init,second,numpy.amax(high_pass_filteredSignal[tmp]),linear_threshold)
    # find if it is above the decay data
    fig = plt.figure()
    plt.plot(high_pass_filteredSignal)
    plt.title(i)
    # plt.plot(xs)
    plt.plot(pks,high_pass_filteredSignal[pks],"x")
    # plt.plot(peaks,high_pass_filteredSignal[peaks],"x")
    # plt.axvline(x=r_peak_index)
    # plt.axhline(y=dec_threshold)
    # print(r_peak_init-linear_threshold+numpy.exp((-(second[0]))*r_peak_init))
    # plt.axhline(y=linear_threshold)
    # plt.plot(low_pass_filteredSignal)
    # plt.plot(differentiated_signal1)
    # plt.plot(data1)
    plt.show()
    loc = 'det' + str(i) + ''
    fig.tight_layout()
    fig.savefig(loc, dpi=200)

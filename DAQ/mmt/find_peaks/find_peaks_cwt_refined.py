import numpy as np
import scipy
import scipy.signal

import DAQ.mmt as mmt

def find_peaks_cwt_refined(vector, widths, decimate, decimate_factor=10, noise_perc=10):
    # print("vector", vector)
    # print("widths", widths)
    if decimate:
        decimate_factor = int(decimate_factor)
        vector_decimate = scipy.signal.decimate(vector, decimate_factor, zero_phase=True)
        # print("vector_decimate",vector_decimate)
        widths_decimate = np.array(list(set(np.array(widths)//decimate_factor)))
        # print("widths_decimate",widths_decimate)
        peaks_cwt = np.array(scipy.signal.find_peaks_cwt(vector_decimate, widths_decimate, noise_perc=noise_perc)) * decimate_factor
        # print("peaks_cwt",peaks_cwt)
    else:
        peaks_cwt = np.array(scipy.signal.find_peaks_cwt(vector, widths, noise_perc=noise_perc))

    tmp=[]
    # print("aaaaa",len(peaks_cwt))
    if len(peaks_cwt)<2:
        tmp.append(peaks_cwt[0])
        tmp.append(peaks_cwt[0]+200)
        peaks_cwt=np.array(tmp)
    # print("aaaaa", len(peaks_cwt))
    # print("peaks_cwt", peaks_cwt)

    peaks_cwt_interval = np.diff(peaks_cwt)
    # print("peaks_cwt_interval",peaks_cwt_interval)
    peaks_search_dist = max(int(np.percentile(peaks_cwt_interval, 20)//2),3)
    peaks_local = scipy.signal.argrelmax(vector, order=peaks_search_dist)[0]

    refined_peaks = np.zeros_like(peaks_cwt)

    for i, peak in enumerate(peaks_cwt):
        refined_peaks[i] = mmt.find_nearest_value(peaks_local,peak)

    return np.sort(list(set(refined_peaks)))

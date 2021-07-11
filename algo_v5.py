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
from pickle import load
import filters, data_reader, perfusion_detector, bp_detector,electrogram_detector
from scipy import fftpack
import time

# pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/qfin.txt"
# ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/A Tach/Haem/Atach_CRTD_21_10_2020_164349_/boxb.txt"
# bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/vfi0017_vvi180_01_27_04_2021_152246_/boxb.txt"

# pp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/VTFI0014_VVI_200_01_09_02_2021_123325_/BP.txt"
pp = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/ADA003_03_12_01_2021_154719_/plethh.txt"
# ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/vtfi0017_test_aai_his_27_04_2021_150217_/BP.txt"
# ee = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/VTFI0014_VVI_200_01_09_02_2021_123325_/ecg.txt"
ee = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/ADA003_03_12_01_2021_154719_/plethg.txt"
# ee1 = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ADA003_03_12_01_2021_154719_/plethg.txt"
# bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/vtfi0017_test_aai_his_27_04_2021_150217_/boxb.txt"
# bpp = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/VTFI0014_VVI_200_01_09_02_2021_123325_/bpao.txt"
bpp = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/ADA003_03_12_01_2021_154719_/BP.txt"

DEBUG = False

STEP_SIZE = 200
BP_LAG = 200
PERFUSION_LAG = 100

def lag_calc(egm_start_time, egm_end_time, signal_with_lag):
    promper = statistics.mean(signal_with_lag)
    min_per_peaks, properties = find_peaks(signal_with_lag * -1, prominence=-promper, width=20)
    min_per_peaks = list(min_per_peaks)
    lag=0
    for i in min_per_peaks:
        if egm_start_time<=i<=egm_end_time:
            lag1=abs(egm_start_time-i)
            # print(lag1)
            lag2=abs(egm_end_time-i)
            # print(lag2)
            if abs(lag1-lag2)>400:
                lag=400
            else:
                lag=int(np.mean([lag2,lag1])/2)

    return lag



def main(electrogram_path, perfusion_path, bp_path, period, decision):
    ELECTROGRAM_PATH = electrogram_path
    PERFUSION_PATH = perfusion_path
    BP_PATH = bp_path
    model = load(open('/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/model.pkl', 'rb'))
    electrogram = data_reader.Data_Reader(data_path=ELECTROGRAM_PATH)
    perfusion = data_reader.Data_Reader(data_path=PERFUSION_PATH)
    bpdata = data_reader.Data_Reader(data_path=BP_PATH)

    electrogram_det = electrogram_detector.ElectrogramDetector()
    perfusion_det = perfusion_detector.PerfusionDetector()
    bp_det = bp_detector.BPDetector()


    if not DEBUG:
        win = pg.GraphicsWindow()
        #
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

    stats = {"BPM": 0,
             "EGM Mean RV": 0,
             "EGM STD RV": 0,
             "EGM Skewness RV": 0,
             "EGM Kurtosis RV": 0,
             "R-R Interval RV": 0,
             # "BP": 0,
             # "Max Actual BP": 0,
             # "Mean Actual BP": 0,
             "Per Mean": 0,
             "Per STD": 0,
             "Per Skewness": 0,
             "Per Kurtosis": 0,
             "Current Perfusion Grad": 0,
             "Quality of Perfusion":0,
             "Perfusion Amplitude":0,
             # "Per Cum. Mean": 0,
             # "Per Cum. STD": 0,
             # "Per Cum. Skewness": 0,
             # "Per Cum. Kurtosis": 0,
             # "Cumulative Perfusion Grad": 0,
             "Decision": decision}

    start = 0
    rr_interval = 1000
    finish = 400
    # tmp=[]
    ecg_data = [None] * 20
    while True:
        # print(f"count: {count}")
        # Setting up the data instreams
        try:
            # electrogram_det.load_new_data(electrogram)
            electrogram_det.load_new_data(electrogram=electrogram)
            ecg_out, raw = electrogram_det.detect_new_data()

            perfusion_det.load_new_data(perfusion)
            per_out = perfusion_det.detect_new_data()
            if np.mean(per_out)<2:
                per_out=per_out*100
            # per_out=np.log(per_out)
            bp_det.load_new_data(bpdata)
            bp_out = bp_det.detect_new_data()
            if np.mean(bp_out)<2:
                bp_out=bp_out*100


        except Exception as e:
            # print("Out of data")
            break
        # try:
        # EGM Peak detection
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
            start_local_time = start_global_time - (count*STEP_SIZE)
            if start_local_time<0:
                start_local_time=0
            end_local_time = abs(end_global_time - (count*STEP_SIZE))

            rr_interval = end_local_time - start_local_time
            bpm_interval = 60000 / rr_interval



            # problem
            # print("??????????",bpm_interval)
            BP_lag=lag_calc(start_local_time,end_local_time,bp_out)
            mean_bp_interval = np.mean(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])

            # print("bp lag time",BP_lag)

            # print("len bp",len(bp_out))
            # print(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])
            max_bp_interval = np.max(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])
            min_bp_interval = np.min(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])

            egmMean_interval = np.mean(ecg_out[start_local_time:end_local_time])
            egmSTD_interval = np.std(ecg_out[start_local_time:end_local_time])
            egmSkew_interval = skew(ecg_out[start_local_time:end_local_time])
            egmKurtosis_interval = kurtosis(ecg_out[start_local_time:end_local_time])

            interval_stats = stats.copy()

            update_dict = {
                # "Max Actual BP": max_bp_interval,
                #            "Mean Actual BP": mean_bp_interval,
                           "BPM": bpm_interval,
                           "EGM Mean RV": egmMean_interval,
                           "EGM STD RV": egmSTD_interval,
                           "EGM Skewness RV": egmSkew_interval,
                           "EGM Kurtosis RV": egmKurtosis_interval,
                           "R-R Interval RV": rr_interval}

            interval_stats.update(update_dict)

            # Perfusion
            perfusion_cut = np.zeros(2000) * np.nan
            PERFUSION_lag=lag_calc(start_local_time,end_local_time,per_out)
            # print(PERFUSION_lag)
            # print("end_global_time",end_local_time)
            # print("start_global_time",start_local_time)
            # print("rr_interval",rr_interval)
            # print(len(per_out[(start_local_time+PERFUSION_LAG):(end_local_time+PERFUSION_LAG)]))

            perfusion_cut[:rr_interval] = per_out[(start_local_time+PERFUSION_lag):(end_local_time+PERFUSION_lag)]
            # print("perfusion_cut",perfusion_cut)
            mat.appendleft(perfusion_cut)

            perfusion_mat = np.array(mat)
            # print("perfusion_mat",perfusion_mat)
            perfusion_consensus = np.nanmean(perfusion_mat, axis=0)
            # print("perfusion_consensus",perfusion_consensus)
            perfusion_consensus_mask = np.isnan(np.sum(perfusion_mat, axis=0))
            # print(perfusion_consensus_mask)
            perfusion_mean = np.nanmean(perfusion_consensus)
            # print(perfusion_mean)
            perfusion_sd = np.nanstd(perfusion_consensus)
            # print(perfusion_sd)

            perfusion_consensus[perfusion_consensus_mask] = np.nan
            sim_scores = []
            for i in mat:
                for a in range(len(perfusion_consensus)):
                    if not np.isnan(perfusion_consensus[a]) and not np.isnan(i[a]):
                        dif=np.power(perfusion_consensus[a],2)-np.power(i[a],2)
                        similarity = np.sqrt(dif)

                # similarity = np.nansum(np.sqrt(np.power((a-b),2)) for a, b in zip(perfusion_consensus,i))
                        sim_scores.append(similarity)
            sim_score = np.nansum(sim_scores)/len(sim_scores)

            print("!!!!!!!!!!!!!!!!",sim_score)


            try:
                perfusion_consensus_argmax = np.nanargmax(perfusion_consensus)
                perfusion_consensus_argmin = np.nanargmin(perfusion_consensus)
            except:
                continue
            # print(perfusion_consensus_argmax)
            perfusion_consensus_max = perfusion_consensus[perfusion_consensus_argmax]
            # print(perfusion_consensus_max)
            perfusion_consensus_min = perfusion_consensus[0]
            perfusion_amplitude = perfusion_consensus_max-perfusion_consensus_min
            per_cons=[]
            for p in perfusion_consensus:
                if not np.isnan(p):
                    per_cons.append(p)
            perSkew = skew(per_cons)
            perKurtosis = kurtosis(per_cons)
            # print(perfusion_consensus_min)
            if perfusion_consensus_argmax!=0:
                # print(perfusion_consensus_max)
                # print(perfusion_consensus_min)
                theta = ((perfusion_consensus_max-perfusion_consensus_min) /abs(perfusion_consensus_argmax-perfusion_consensus_argmin)) *10
                # print("!!!!!!!!!!!!!!Theta",theta)
                tmpgrad = math.degrees(math.atan(theta))

                # print("!!!!!!!!!!!!!!! tmpgrad", tmpgrad
                bp_inteerval = model.predict([[tmpgrad]])[0]
                # bp_inteerval = float(tmpgrad * perfusion_consensus_argmax * 0.00750062)
                # print("!!!!!!!!!!!!!!! bp", max_bp_interval, bp_inteerval)
                # bp_inteerval = float((bpm_interval*(tmpgrad))*rr_interval* 0.00750062)
                # print("!!!!!!!!!!!!!!! bp", max_bp_interval, bp_inteerval)
            else:
                bp_inteerval=0
                tmpgrad=0


            update_dict = {
                # "BP": bp_inteerval,
                           "Quality of Perfusion":sim_score,
                           "Perfusion Amplitude": perfusion_amplitude,
                           "Current Perfusion Grad": tmpgrad,
                           "Per Mean": perfusion_mean,
                           "Per STD": perfusion_sd,
                           "Per Skewness": perSkew,
                           "Per Kurtosis": perKurtosis}

            interval_stats.update(update_dict)
            ecg_data.append([bpm_interval,rr_interval])
            # todo
            # my_decision = getDecision(bpm=bpm_interval,rr_interval=rr_interval)
            output.append(interval_stats)
        # except Exception as e:
        #     print("Out of")
        #     pass

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
                    curve2.setData(x=np.arange(len(per_out)), y=per_out)
                except Exception as e:
                    pass


                QtGui.QGuiApplication.processEvents()



        count = count + 1
        # time.sleep(0.5)
    # if 0<bpm_interval<50:
    #     print(start_local_time)
    #     print(start_global_time)
    #     print(end_local_time)
    #     print(end_global_time)
    #     print(bpm_interval)
    #     time.sleep(60)
    output=pd.DataFrame(output)
    output=output.iloc[1: , :]
    return output

#
# if __name__ == '__main__':
#     output = main(perfusion_path=pp, bp_path=bpp, electrogram_path=ee, period=1,decision=1)
#     output_pd = pd.DataFrame(output)
#     output_pd.to_csv("paok.csv")
#     print("Done")

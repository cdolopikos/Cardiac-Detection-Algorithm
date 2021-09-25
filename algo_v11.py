import collections
import statistics
from scipy.stats import skew, kurtosis
import pandas as pd
from PyQt6.QtWidgets import QTableWidget
from PyQt6 import QtGui
import pyqtgraph as pg
import numpy as np
from scipy.signal import find_peaks # Filter requirements.
import  data_reader, perfusion_detector, bp_detector,electrogram_detector


ecg = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/ecg.txt"
lsr1 = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/plethh.txt"
lsr2 = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/qfin.txt"
bp = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/bpao.txt"

DEBUG = True

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





def main(electrogram_path, perfusion_path,perfusion_path2, bp_path, period,num_lasers,extra, treat, flname,patient, bp1, hrs):
    ELECTROGRAM_PATH = electrogram_path
    PERFUSION_PATH = perfusion_path
    PERFUSION_PATH2 = perfusion_path2
    BP_PATH = bp_path
    electrogram = data_reader.Data_Reader(data_path=ELECTROGRAM_PATH)
    perfusion = data_reader.Data_Reader(data_path=PERFUSION_PATH)
    bpdata = data_reader.Data_Reader(data_path=BP_PATH)
    print(ELECTROGRAM_PATH)
    print(PERFUSION_PATH)
    try:
        perfusion2 = data_reader.Data_Reader(data_path=PERFUSION_PATH2)
    except:
        pass


    electrogram_det = electrogram_detector.ElectrogramDetector()
    perfusion_det = perfusion_detector.PerfusionDetector()
    perfusion_det2 = perfusion_detector.PerfusionDetector()
    bp_det = bp_detector.BPDetector()


    if not DEBUG:
        # pg.setConfigOption('background', 'w')
        win = pg.GraphicsWindow()

        #
        p1 = win.addPlot(row=1, col=0)
        p2 = win.addPlot(row=2, col=0)
        p3 = win.addPlot(row=0, col=0)
        p4 = win.addPlot(row=0, col=1)
        p5 = win.addPlot(row=1, col=1)
        p6 = win.addPlot(row=2, col=1)
        p7 = win.addPlot(row=0, col=2)
        p8 = win.addPlot(row=1, col=2)
        p9 = win.addPlot(row=2, col=2)

        curve1 = p1.plot(pen=pg.mkPen('g'))
        dot1 = p1.plot(pen=None, symbol="o")
        curve2 = p2.plot(pen=pg.mkPen('r'))
        curve3 = p3.plot(pen=pg.mkPen('g'))
        curve4 = p4.plot(pen=pg.mkPen('r'))
        curve5 = p5.plot(pen=pg.mkPen('r'))
        curve6 = p5.plot(pen=pg.mkPen('b'))
        curve7 = p5.plot(pen=pg.mkPen('c'))
        curve8 = p5.plot(pen=pg.mkPen('m'))
        curve9 = p5.plot(pen=pg.mkPen('y'))
        curve10 = p5.plot(pen=pg.mkPen('k'))
        curve11 = p6.plot(pen=pg.mkPen('w'))
        curve12 = p8.plot(pen=pg.mkPen('r'))
        curve13 = p8.plot(pen=pg.mkPen('b'))
        curve14 = p8.plot(pen=pg.mkPen('c'))
        curve15 = p8.plot(pen=pg.mkPen('m'))
        curve16 = p8.plot(pen=pg.mkPen('y'))
        curve17 = p8.plot(pen=pg.mkPen('k'))
        curve18 = p9.plot(pen=pg.mkPen('w'))
        curve19 = p7.plot(pen=pg.mkPen('r'))
        dot2 = p2.plot(pen=None, symbol="x")
        dot3 = p2.plot(pen=None, symbol="o")
        dot4 = p4.plot(pen=None, symbol="o")

        table = QTableWidget()
        table.setColumnCount(3)
        table.setRowCount(2)
        table.setHorizontalHeaderLabels(["BPM", "Perfusion-Gradient", "Diagnosis"])

    mat = collections.deque(maxlen=6)
    mat2 = collections.deque(maxlen=6)
    mat_pks = collections.deque(maxlen=50)


    count = 0
    output = []

    stats = {
        "Filename":flname,
        "Patient":patient,
        "Global Time": 0,
         "BPM": 0,
         "EGM Mean RV": 0,
         "EGM STD RV": 0,
         "EGM Skewness RV": 0,
         "EGM Kurtosis RV": 0,
         "EGM Quality":0,
         "R-R Interval RV": 0,
         "BP Estimat": 0,
         # "BP": 0,
         "Max Actual BP": 0,
         "Mean Actual BP": 0,
         "Per Mean": 0,
         "Per STD": 0,
         "Per Skewness": 0,
         "Per Kurtosis": 0,
         "Current Perfusion Grad": 0,
         "Quality of Perfusion":0,
         "Perfusion Amplitude":0,
         "Magic Laser":0,
         "Diagnosis": period,
        "Treatment":treat,
        "Rhythm":treat,
    "BP1":bp1,
    "HRS":hrs}

    start = 0
    rr_interval = 1000
    finish = 400
    ecg_data = [None] * 20
    while True:
        # Setting up the data instreams
        try:
            # ECG data
            electrogram_det.load_new_data(electrogram=electrogram)
            ecg_out, raw= electrogram_det.detect_new_data()
            # PPG 1
            perfusion_det.load_new_data(perfusion)
            per_out,per_raw = perfusion_det.detect_new_data()
            if np.mean(per_out)<2:
                per_out=per_out*100
            # PPG 2
            if num_lasers>1:
                perfusion_det2.load_new_data(perfusion2)
                per_out2,per_raw1 = perfusion_det2.detect_new_data()
                if np.mean(per_out2)<2:
                    per_out2=per_out2*100
            # BP
            bp_det.load_new_data(bpdata)
            bp_out = bp_det.detect_new_data()
            if np.mean(bp_out)<2:
                bp_out=bp_out*100

        except Exception as e:
            # print("Out of data")
            break
        # EGM Peak detection
        prom = np.mean(ecg_out)
        peaks, properties = find_peaks(ecg_out, prominence=prom, width=20)
        min_peaks, properties = find_peaks(-1*ecg_out, prominence=prom, width=20)

        peak_pairs_to_process = []
        for p in list(peaks):
            # global index
            if p > 1400 and p <= 1600:
                peak_global_time = p + (count*STEP_SIZE)
                if len(mat_pks) > 0:
                    peak_pairs_to_process.append((mat_pks[0], peak_global_time))
                mat_pks.appendleft(peak_global_time)
        # Setting the time position of ecg peaks
        ecg_history = []
        for i in range(len(peaks)):
            start = peaks[i] - 250
            if start < 0:
                start = 0
            finish = peaks[i] + 250
            if finish > 2000:
                finish = 2000
            tmp_ecg = ecg_out[start:finish]
            ecg_history.append(tmp_ecg)

        # ECG similarity score
        ecg_sim_scores = []
        similarity=0
        for i in range(len(ecg_history)):
            if i < len(ecg_history)-1:
                tmp_ecg=ecg_history[i]
                tmp_ecg1=ecg_history[i+1]
                if len(tmp_ecg) <= len(tmp_ecg1):
                    for a in range(len(tmp_ecg)):
                        if not np.isnan(tmp_ecg[a]) and not np.isnan(tmp_ecg1[a]):
                            dif = np.power(np.log(tmp_ecg[a]), 2) - np.power(np.log(tmp_ecg1[a]), 2)
                            similarity = np.sqrt(dif)
                elif len(tmp_ecg1)<len(tmp_ecg):
                    for a in range(len(tmp_ecg1)):
                        if not np.isnan(tmp_ecg1[a]) and not np.isnan(tmp_ecg[a]):
                            dif = np.power(np.log(tmp_ecg1[a]), 2) - np.power(np.log(tmp_ecg[a]), 2)
                            similarity = np.sqrt(dif)
                ecg_sim_scores.append(similarity)
        ecg_sim_score = np.nansum(ecg_sim_scores) / len(ecg_sim_scores)
        if np.isnan(ecg_sim_score) or np.isinf(ecg_sim_score):
            ecg_sim_score=0
        # Setting global timing
        for start_global_time, end_global_time in peak_pairs_to_process:
            start_local_time = start_global_time - (count*STEP_SIZE)
            if start_local_time<0:
                start_local_time=0
            end_local_time = abs(end_global_time - (count*STEP_SIZE))
            # Calculating the R-R interval
            # Estimating the BPM
            rr_interval = end_local_time - start_local_time
            bpm_interval = 60000 / rr_interval
            # Lag time calculation
            BP_lag=lag_calc(start_local_time,end_local_time,bp_out)
            # BP values for validation purposes
            mean_bp_interval = np.mean(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])
            max_bp_interval = np.max(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])
            min_bp_interval = np.min(bp_out[(start_local_time+BP_lag):(end_local_time+BP_lag)])
            # ECG stats
            egmMean_interval = np.mean(ecg_out[start_local_time:end_local_time])
            egmSTD_interval = np.std(ecg_out[start_local_time:end_local_time])
            egmSkew_interval = skew(ecg_out[start_local_time:end_local_time])
            egmKurtosis_interval = kurtosis(ecg_out[start_local_time:end_local_time])

            global_time_of_beat = peaks[len(peaks)-1] + (count*STEP_SIZE)
            interval_stats = stats.copy()


            update_dict = {
                "Global Time": (global_time_of_beat + extra),
                "Max Actual BP": max_bp_interval,
                "Mean Actual BP": mean_bp_interval,
                "BPM": bpm_interval,
                "EGM Mean RV": egmMean_interval,
                "EGM STD RV": egmSTD_interval,
                "EGM Skewness RV": egmSkew_interval,
                "EGM Kurtosis RV": egmKurtosis_interval,
                "EGM Quality": abs(ecg_sim_score),
                "R-R Interval RV": rr_interval}


            interval_stats.update(update_dict)

            # PPG
            if num_lasers==1:
                perfusion_cut = np.zeros(2000) * np.nan
                PERFUSION_lag=lag_calc(start_local_time,end_local_time,per_out)


                perfusion_cut[:rr_interval] = per_out[(start_local_time+PERFUSION_lag):(end_local_time+PERFUSION_lag)]
                mat.appendleft(perfusion_cut)

                perfusion_mat = np.array(mat)
                perfusion_consensus = np.nanmean(perfusion_mat, axis=0)
                perfusion_consensus_mask = np.isnan(np.sum(perfusion_mat, axis=0))
                perfusion_mean = np.nanmean(perfusion_consensus)
                perfusion_sd = np.nanstd(perfusion_consensus)

                perfusion_consensus[perfusion_consensus_mask] = np.nan
                sim_scores = []
                for i in mat:
                    for a in range(len(perfusion_consensus)):
                        if not np.isnan(perfusion_consensus[a]) and not np.isnan(i[a]):
                            dif=np.power(np.log(perfusion_consensus[a]),2)-np.power(np.log(i[a]),2)
                            similarity = np.sqrt(dif)

                            sim_scores.append(similarity)
                sim_score = np.nansum(sim_scores)/len(sim_scores)



                try:
                    perfusion_consensus_argmax = np.nanargmax(perfusion_consensus)
                    perfusion_consensus_argmax_magic = np.nanargmax(np.log10(perfusion_consensus))
                    perfusion_consensus_argmin = np.nanargmin(perfusion_consensus)
                    perfusion_consensus_argmin_magic = np.nanargmin(np.log10(perfusion_consensus))
                except:
                    continue
                perfusion_consensus_max = perfusion_consensus[perfusion_consensus_argmax]
                perfusion_consensus_max_magic = np.log10(perfusion_consensus[perfusion_consensus_argmax_magic])
                perfusion_consensus_min = perfusion_consensus[0]
                perfusion_consensus_min_magic = np.log10(perfusion_consensus[0])
                perfusion_amplitude = (perfusion_consensus_max)-(perfusion_consensus_min)
                perfusion_amplitude_laser = (perfusion_consensus_max_magic*10)-np.log(perfusion_consensus_min_magic)
                if np.isinf(perfusion_amplitude):
                    perfusion_amplitude = 0
                if np.isinf(perfusion_amplitude_laser):
                    perfusion_amplitude_laser=0
                per_cons=[]
                for p in perfusion_consensus:
                    if not np.isnan(p):
                        per_cons.append(p)

                perSkew = skew(per_cons)
                perKurtosis = kurtosis(per_cons)
                if perfusion_consensus_argmax!=0:
                    theta = ((perfusion_consensus_max-perfusion_consensus_min) /abs(perfusion_consensus_argmax-perfusion_consensus_argmin)) *10
                    if np.isinf(theta):
                        theta=0

                    bp_inteerval = np.log10(bpm_interval/ (theta*rr_interval*sim_score))

                else:
                    bp_inteerval=0
                    theta=0


                update_dict = {
                    "BP Estimat": bp_inteerval,
                               "Quality of Perfusion":abs(sim_score),
                               "Perfusion Amplitude": (perfusion_amplitude),
                               "Magic Laser": ((perfusion_amplitude_laser)),
                               "Current Perfusion Grad": (theta),
                               "Per Mean": perfusion_mean,
                               "Per STD": perfusion_sd,
                               "Per Skewness": perSkew,
                               "Per Kurtosis": perKurtosis}

            else:
                perfusion_cut = np.zeros(2000) * np.nan
                perfusion_cut2 = np.zeros(2000) * np.nan
                PERFUSION_lag = lag_calc(start_local_time, end_local_time, per_out)
                PERFUSION_lag2 = lag_calc(start_local_time, end_local_time, per_out2)
                perfusion_cut[:rr_interval] = per_out[
                                              (start_local_time + PERFUSION_lag):(end_local_time + PERFUSION_lag)]
                perfusion_cut2[:rr_interval] = per_out2[
                                               (start_local_time + PERFUSION_lag2):(end_local_time + PERFUSION_lag2)]
                mat.appendleft(perfusion_cut)
                mat2.appendleft(perfusion_cut2)

                perfusion_mat = np.array(mat)
                perfusion_mat2 = np.array(mat2)
                perfusion_consensus = np.nanmean(perfusion_mat, axis=0)
                perfusion_consensus2 = np.nanmean(perfusion_mat2, axis=0)
                perfusion_consensus_mask = np.isnan(np.sum(perfusion_mat, axis=0))
                perfusion_consensus_mask2 = np.isnan(np.sum(perfusion_mat2, axis=0))
                perfusion_mean = np.nanmean(perfusion_consensus)
                perfusion_mean2 = np.nanmean(perfusion_consensus2)
                perfusion_sd = np.nanstd(perfusion_consensus)
                perfusion_sd2 = np.nanstd(perfusion_consensus2)
                perfusion_consensus[perfusion_consensus_mask] = np.nan
                perfusion_consensus2[perfusion_consensus_mask2] = np.nan
                sim_scores = []
                for i in mat:
                    for a in range(len(perfusion_consensus)):
                        if not np.isnan(perfusion_consensus[a]) and not np.isnan(i[a]):
                            dif = np.power(np.log(perfusion_consensus[a]), 2) - np.power(np.log(i[a]), 2)
                            similarity = np.sqrt(dif)
                            sim_scores.append(similarity)
                sim_score = np.nansum(sim_scores) / len(sim_scores)

                sim_scores2 = []
                for i in mat2:
                    for a in range(len(perfusion_consensus2)):
                        if not np.isnan(perfusion_consensus2[a]) and not np.isnan(i[a]):
                            dif2 = np.power(np.log(perfusion_consensus2[a]), 2) - np.power(np.log(i[a]), 2)
                            similarity2 = np.sqrt(dif2)
                            sim_scores2.append(similarity2)
                sim_score2 = np.nansum(sim_scores2) / len(sim_scores2)



                try:
                    perfusion_consensus_argmax_magic = np.nanargmax(np.log10(perfusion_consensus))
                    perfusion_consensus_argmax = np.nanargmax(perfusion_consensus)
                    perfusion_consensus_argmin_magic = np.nanargmin(np.log10(perfusion_consensus))
                    perfusion_consensus_argmin = np.nanargmin(perfusion_consensus)
                except:
                    continue
                try:
                    perfusion_consensus_argmax2 = np.nanargmax(perfusion_consensus2)
                    perfusion_consensus_argmax2_magic = np.nanargmax(np.log10(perfusion_consensus2))
                    perfusion_consensus_argmin2 = np.nanargmin(perfusion_consensus2)
                    perfusion_consensus_argmin2_magic = np.nanargmin(np.log10(perfusion_consensus2))
                except:
                    continue

                perfusion_consensus_max = perfusion_consensus[perfusion_consensus_argmax]
                perfusion_consensus_max_magic = perfusion_consensus[perfusion_consensus_argmax_magic]
                perfusion_consensus_max2 = perfusion_consensus2[perfusion_consensus_argmax2]
                perfusion_consensus_max2_magic = perfusion_consensus2[perfusion_consensus_argmax2_magic]
                perfusion_consensus_min = perfusion_consensus[0]
                perfusion_consensus_min_magic = np.log10(perfusion_consensus[0])
                perfusion_consensus_min2 = perfusion_consensus2[0]
                perfusion_consensus_min2_magic = np.log10(perfusion_consensus2[0])
                perfusion_amplitude = perfusion_consensus_max - perfusion_consensus_min
                perfusion_amplitude_laser = (perfusion_consensus_max_magic*10) - (perfusion_consensus_min_magic)
                perfusion_amplitude2 = perfusion_consensus_max2 - perfusion_consensus_min2
                perfusion_amplitude2_laser = (perfusion_consensus_max2_magic*10) - (perfusion_consensus_min2_magic)
                if np.isinf(perfusion_amplitude):
                    perfusion_amplitude = 0
                if np.isinf(perfusion_amplitude2):
                    perfusion_amplitude2 = 0
                per_cons = []
                for p in perfusion_consensus:
                    if not np.isnan(p):
                        per_cons.append(p)
                perSkew = skew(per_cons)
                perKurtosis = kurtosis(per_cons)

                per_cons2 = []
                for p in perfusion_consensus2:
                    if not np.isnan(p):
                        per_cons2.append(p)
                perSkew2 = skew(per_cons2)
                perKurtosis2 = kurtosis(per_cons2)
                if perfusion_consensus_argmax != 0:
                    theta = ((perfusion_consensus_max - perfusion_consensus_min) / abs(
                        perfusion_consensus_argmax - perfusion_consensus_argmin)) * 10
                    if np.isinf(theta):
                        theta = 0

                    bp_inteerval = np.log10(bpm_interval / (theta * rr_interval * sim_score))

                else:
                    bp_inteerval = 0
                    theta = 0

                if perfusion_consensus_argmax2 != 0:

                    theta2 = ((perfusion_consensus_max2 - perfusion_consensus_min2) / abs(
                        perfusion_consensus_argmax2 - perfusion_consensus_argmin2)) * 10
                    if np.isinf(theta2):
                        theta2 = 0

                    bp_inteerval2 = np.log10(bpm_interval / (theta2 * rr_interval * sim_score2))

                else:
                    bp_inteerval2 = 0
                    theta2 = 0

                if np.isinf(perfusion_amplitude):
                    perfusion_amplitude = 0
                if np.isinf(perfusion_amplitude_laser):
                    perfusion_amplitude_laser=0
                if np.isinf(perfusion_amplitude2):
                    perfusion_amplitude = 0
                if np.isinf(perfusion_amplitude2_laser):
                    perfusion_amplitude2_laser=0

                if sim_score2>sim_score:
                    sim_score = sim_score2
                    bp_inteerval=bp_inteerval2
                    perfusion_amplitude= perfusion_amplitude2
                    perfusion_amplitude_laser= perfusion_amplitude2_laser
                    theta=theta2
                    perfusion_mean =perfusion_mean2
                    perSkew=perSkew2
                    perfusion_sd=perfusion_sd2
                    perKurtosis=perKurtosis2

                update_dict = {
                    "BP Estimat": bp_inteerval,
                    "Quality of Perfusion": (sim_score),
                    "Perfusion Amplitude": (perfusion_amplitude),
                    "Magic Laser": (perfusion_amplitude_laser),
                    "Current Perfusion Grad": (theta),
                    "Per Mean": perfusion_mean,
                    "Per STD": perfusion_sd,
                    "Per Skewness": perSkew,
                    "Per Kurtosis": perKurtosis}

            interval_stats.update(update_dict)
            ecg_data.append([bpm_interval,rr_interval])

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
                    curve12.setData(x=np.arange(len(mat2[0])), y=mat2[0])
                except Exception as e:
                    pass

                try:
                    curve13.setData(x=np.arange(len(mat2[1])), y=mat2[1])
                except Exception as e:
                    pass

                try:
                    curve14.setData(x=np.arange(len(mat2[2])), y=mat2[2])
                except Exception as e:
                    pass

                try:
                    curve15.setData(x=np.arange(len(mat2[3])), y=mat2[3])
                except Exception as e:
                    pass

                try:
                    curve16.setData(x=np.arange(len(mat2[4])), y=mat2[4])
                except Exception as e:
                    pass

                try:
                    curve17.setData(x=np.arange(len(mat2[5])), y=mat2[5])
                except Exception as e:
                    pass

                try:
                    curve19.setData(x=np.arange(len(mat2[0])), y=mat2[0])
                except Exception as e:
                    pass

                try:
                    curve18.setData(x=np.arange(len(perfusion_consensus2)), y=perfusion_consensus2)
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

# # # #
# if __name__ == '__main__':
#     output = main(perfusion_path=lsr1, perfusion_path2=lsr2, bp_path=bp, electrogram_path=ecg, period=1,extra=0, num_lasers=2,treat="IDK",flname="d",patient="a",bp1="1",hrs="q")
#     output_pd = pd.DataFrame(output)
#     output_pd.to_csv("paok.csv")
#     print("Done")

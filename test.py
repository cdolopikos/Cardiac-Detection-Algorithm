import data_reader
import electrogram_detector
import perfusion_detector
import matplotlib.pyplot as plt
import sys
import filters
import numpy as np

ecg = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/ecg.txt"
lsr1 = "/Users/cmdgr/OneDrive - Imperial College London/VTFI0015_VVI_SET01_140_16_02_2021_120646_/plethh.txt"

electrogram = data_reader.Data_Reader(data_path=ecg)
perfusion = data_reader.Data_Reader(data_path=lsr1)

electrogram_det = electrogram_detector.ElectrogramDetector()
perfusion_det = perfusion_detector.PerfusionDetector()
count = 0

while True:
    try:
        electrogram_det.load_new_data(electrogram=electrogram)
        ecg_out, raw = electrogram_det.detect_new_data()
        # PPG 1
        perfusion_det.load_new_data(perfusion)
        per_out, per_raw = perfusion_det.detect_new_data()

        f, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(ecg_out, per_out)
        ax1.set_title("ECG vs PPG")
        ax2.plot(ecg_out)
        ax2.set_title("ECG")
        ax3.plot(per_out)
        ax3.set_title("PPG")
        f.savefig('/Users/cmdgr/Dropbox/AAD-Documents/ECG vs PPG/VTFI0015_' + str(count) + '1000_ms.png')
        count += 1

    except:
        print("Out of data")
        sys.exit()

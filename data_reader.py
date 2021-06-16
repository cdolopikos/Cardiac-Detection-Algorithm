# ## detect QRS complex from ECG time series

import numpy as np
import math
from numpy import genfromtxt
import matplotlib.pyplot as plt


def read_ecg(file_name):
    return genfromtxt(file_name, delimiter=',')


# print(read_ecg('/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_/ecg.txt'))
ecg = read_ecg("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_/boxb.txt")
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/Normal/boston_test01_11_11_2020_140516_/boxa.txt")
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/Normal/boston_test01_11_11_2020_140516_/ecg.txt")
    #
    # '/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_/ecg.txt')
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/Normal/boston_test01_11_11_2020_140516_/ecg.txt")
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/Normal/AAD_RVLEADFRACTURE_MB01_27_10_2020_153348_/ecg.txt")
    #
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_/boxa.txt")
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_/boxb.txt")
    #
    #
    #
    #
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/Normal/AAD_RVLEADFRACTURE_MB01_27_10_2020_153348_/ecg.txt")
    # "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/CT Lead noise/ada001_30_07_2020_093706_/ecg.txt")
    # )
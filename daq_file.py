import zipfile

from DAQ.hdy import DAQ_File

# x=DAQ_File.fast_load_txt('/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_')

import pandas as pd

# read_file = pd.read_csv (r'/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_')
# read_file.to_csv (r'Path where the CSV will be saved\File name.csv', index=None)
import os
from os import walk

f = []
mypath='/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_'
for (dirpath, dirnames, filenames) in walk(mypath):
    f.extend(filenames)
    break

outputpath='/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_csv'
for i in f:
    filename, file_extension = os.path.splitext(i)
    if file_extension == ".txt":
        read_file = pd.read_csv(mypath+'/'+i)
        read_file.to_csv(outputpath+'/'+filename+'.csv', index=None)

import shutil
shutil.make_archive(outputpath, 'zip', outputpath)
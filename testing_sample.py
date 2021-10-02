import shutil

import pandas as pd
import os
import glob
import pandas as pd
import numpy as np
import random


os.chdir("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/laser_1")
cwd = os.getcwd()
print(cwd)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(all_filenames)
print(len(all_filenames))
sample = int(0.1 * len(all_filenames))
print(sample)
sample_files = []
for i in range(sample):
    n = random.randint(0,len(all_filenames))
    sample_files.append(n)
print(sample_files)

for j in sample_files:
    print(cwd+"/"+all_filenames[j])
    shutil.move(cwd+"/"+all_filenames[j], cwd+"_testing"+"/"+all_filenames[j])
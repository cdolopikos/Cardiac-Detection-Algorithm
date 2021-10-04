import shutil

import pandas as pd
import os
import glob
import pandas as pd
import numpy as np
import random

def randomizer():
    n = random.randint(0, len(all_filenames))
    return n
os.chdir("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/instances")
cwd = os.getcwd()
print(cwd)
extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
print(all_filenames)
print(len(all_filenames))
sample = int(0.1 * len(all_filenames))
print(sample)
sample_files = []
for i in range((sample)):
    print(i)
    n = randomizer()
    if n not in sample_files:
        sample_files.append(n)
    else:
        i -=1
print(sample_files)

for j in sample_files:
    print(cwd+"/"+all_filenames[j])
    shutil.move(cwd+"/"+all_filenames[j], cwd+"_testing"+"/"+all_filenames[j])
import os
import glob
import pandas as pd
import numpy as np
def concat():
    os.chdir("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data")
    cwd = os.getcwd()
    print(cwd)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    print(all_filenames)
    unique_names=[]
    for i in all_filenames:
        if "Baseline" in i:
            f=i.lower().split(".csv")[0]
            print(f.split('baseline')[0])
            print(f,"ok")
            if f.split('baseline')[0] not in unique_names:
                unique_names.append(f.split('baseline')[0])

    tmp=[]
    print(unique_names)
    for i in unique_names:
        for j in all_filenames:
            if i in j:
             tmp.append(j)
        combined_csv = pd.concat([pd.read_csv(f) for f in tmp])
        combined_csv.to_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/combined/"+str(i)+".csv",
                            index=False, encoding='utf-8-sig')
        tmp.clear()

    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])

concat()
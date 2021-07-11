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
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    combined_csv=combined_csv.fillna(0)
    # for f in all_filenames:
    #     print(f)
    #     pd.read_csv(f)
    #export to csv
    tmp = combined_csv[(combined_csv.index > np.percentile(combined_csv.index, 33)) & (combined_csv.index <= np.percentile(combined_csv.index, 66))]
    print(len(combined_csv))
    print(len(combined_csv.iloc[:int(len(combined_csv)*0.5)]))
    print(len(combined_csv.iloc[int(len(combined_csv)*0.5):]))
    testing_combined_csv = combined_csv.iloc[:int(len(combined_csv)*0.5)]
    testing_combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/testing_combined_csv.csv", index=False, encoding='utf-8-sig')

    training_combined_csv = combined_csv.iloc[int(len(combined_csv)*0.5):]
    training_combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/training_combined_csv.csv", index=False, encoding='utf-8-sig')

    # combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/combined_csv.csv", index=False, encoding='utf-8-sig')

concat()
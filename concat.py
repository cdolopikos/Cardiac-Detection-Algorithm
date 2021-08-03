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
    # df = combined_csv[["Current Perfusion Grad","Quality of Perfusion"]]
    # lista=[]
    # for i in range(len(df)):
    #     print("aaa",df.iloc[i])
    #     tmp = df.iloc[i][0]*df.iloc[i][1]
    #     blood_pressure_pred = 0.0002*np.power(tmp,3) + 0.0225*np.power(tmp,2)-0.2238*tmp +87.138
    #     lista.append(blood_pressure_pred)
    # combined_csv.insert(2,"blood_pressure_pred", lista, True)
    # testing_combined_csv = pd.concat([pd.read_csv(f).iloc[:int(len(f)*0.5)] for f in all_filenames ])
    # training_combined_csv = pd.concat([pd.read_csv(f).iloc[int(len(f)*0.5):] for f in all_filenames ])
    test=[]
    train=[]
    # for f in all_filenames:
    #     dt = pd.read_csv(f)
    #     training=dt[:int(len(dt)*0.5)]
    #     testing = dt[int(len(dt)*0.5):]
    #     test.append(testing)
    #     train.append(training)
    # testing_combined_csv = pd.concat(testing)
    # training_combined_csv = pd.concat(training)
    # combined_csv=combined_csv.fillna(0)
    # for f in all_filenames:
    #     print(f)
    #     pd.read_csv(f)
    #export to csv

    # testing_combined_csv = combined_csv.iloc[:int(len(combined_csv)*0.66)]
    # testing_combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/testing_combined_csv.csv", index=False, encoding='utf-8-sig')
    #
    # training_combined_csv = combined_csv.iloc[int(len(combined_csv)*0.33):]
    # training_combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/training_combined_csv.csv", index=False, encoding='utf-8-sig')

    combined_csv.to_csv( "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/combined_csv.csv", index=False, encoding='utf-8-sig')

concat()
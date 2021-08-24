import pandas as pd
import numpy as np

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/testing/combined_csvlaser_1_testing.csv")
# mgl=df["Magic Laser"]
# print(type(df["Magic Laser"].iloc[16]))
# print(type(df["Magic Laser"].iloc[15]))
# print((df["Magic Laser"].iloc[15]))
# print(isinstance(df["Magic Laser"].iloc[16],str))
# print(()

from os import walk

f = []
mypath="/Volumes/mc1/Preprocessed_data/laser_2/"
for (dirpath, dirnames, filenames) in walk("/Volumes/mc1/Preprocessed_data/laser_2/"):
    for f in filenames:
        print(mypath+str(f))
        df = pd.read_csv(mypath+str(f))
        print(df)
        mgl = df["Magic Laser 2"]
        print(mgl)
        for i in range(len(mgl)):
            # print(i)
            if df["Magic Laser 2"].iloc[i] == "#NAME?":
                df["Magic Laser 2"].iloc[i] = 0
                print(df["Magic Laser 2"].iloc[i])

        df.to_csv(mypath+str(f))
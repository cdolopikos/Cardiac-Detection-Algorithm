import math
import sys
import pandas as pd
import zipfile
import glob, os
import matplotlib.pyplot as plt

path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk"
count = 0
os.chdir(path)
actual=[]
estimated=[]
for f in glob.glob("*"):
    flname = f.lower().split(".csv")[0]
    print(flname)
    if flname !="combined_csv":
        df=pd.read_csv(f)
        print(df.columns)
        est=df["BP"]
        act=df["Max Actual BP"]*100
        actual.append(act)
        estimated.append(est)

        plt.plot(act, est)
        plt.plot(act, est,'ro')
        plt.show()

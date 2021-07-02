import math
import sys
import pandas as pd
import zipfile
import glob, os
import matplotlib.pyplot as plt
import numpy as np

path= "/Preprocessed_data"
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
        print(len(est))
        act=df["Max Actual BP"]*100
        for i in range(len(est)):
            if est[i]==0:
                est.drop(i)
                act.drop(i)
        print(len(est))
        act = np.log(act)
        actual.append(act)
        estimated.append(est)

        # plt.plot(act, est)
        plt.plot(act,est,'o')
        plt.xlabel('actual')
        plt.ylabel('estimated')
        plt.title(flname)
        # plt.plot(est,'bo')
        # plt.plot(act,np.arange(len(act)),'ro')
        plt.show()

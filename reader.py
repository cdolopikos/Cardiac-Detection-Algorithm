import math
import sys

import numpy as np

import matt2 as platform
import pandas as pd
import zipfile
import glob, os


database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/database.csv")
path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1"
count = 0
os.chdir(path)

for f in glob.glob("*"):
    flname = f.lower().split(".zip")[0]
    count += 1
    df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "Period", "Begin", "End","BP", "Laser1"]]
    print("///",len(df))
    if df.empty:
        continue
    else:
        if len(df)>1:
            time_stamp = []
            for i in range(len(df)):
                print(df.iloc[i])
                deci= df.iloc[i].loc["Period"]
                end= df.iloc[i].loc["End"]
                begin= df.iloc[i].loc["Begin"]
                if not np.isnan(end) and not np.isnan(begin):
                    time_stamp.append([deci,int(begin),int(end)])
                print(time_stamp)

            for i in range(len(df)):
                if isinstance(df.iloc[i].loc["LeadRV"], str):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                    leadShock = str(df.iloc[i].loc["LeadShock"]) + ".txt"
                else:
                    leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                # leadShock=str(leadShock)+".txt"
                bp=str(bp)+".txt"
                laser1=str(laser1)+".txt"
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/"+str(f))
                # ldrv=(zf.open(leadRV))
                ldrv=pd.read_csv(zf.open(leadRV))
                s=time_stamp[i][1]
                x=time_stamp[i][2]
                ldrv=ldrv.iloc[s:x]
                ldrv.to_csv("ldrv.csv",index=False)
                ldrv="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ldrv.csv"
                # lsr1=(zf.open(laser1))
                lsr1=pd.read_csv(zf.open(laser1))
                lsr1 = lsr1.iloc[s:x]
                lsr1.to_csv("lsr1.csv", index=False)
                lsr1 = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/lsr1.csv"
                blood_pressure=pd.read_csv(zf.open(bp))
                blood_pressure = blood_pressure.iloc[s:x]
                blood_pressure.to_csv("blood_pressure.csv", index=False)
                blood_pressure = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/blood_pressure.csv"

                # blood_pressure=(zf.open(bp))
                out = platform.main(electrogram_path=ldrv, perfusion_path=lsr1, bp_path=blood_pressure, period=time_stamp[i][0])
                # os.remove("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/ldrv.csv")
        else:
            print("paok")
        # out = platform.main(electrogram_path=ldrv,perfusion_path=lsr1,bp_path=blood_pressure,period=period)
        #
        # print(out)
        # csv="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/out"+str(flname)+str(period)+".csv"
        # out.to_csv(csv, index=False)



sys.exit()
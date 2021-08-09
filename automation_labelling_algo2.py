import math
import sys

# import klaus as platform
import algo_v7 as platform
import pandas as pd
import zipfile
import glob, os

database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/database2.csv")
path= "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped"
count = 0
os.chdir(path)

for f in glob.glob("*"):
    tmp = []
    flname = f.lower().split(".zip")[0]
    print("f", f)
    count += 1
    df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "ECG","Period", "Begin", "End","BP", "Laser1", "Laser2"]]
    print(df)
    if df.empty:
        continue
    else:
        if len(df)<=1:
            num_lasers=1
            for i in range(len(df)):
                if  isinstance(df.iloc[i].loc["LeadRV"], str):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                else:
                    leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
                period = df.iloc[i].loc["Period"]
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                laser2 = df.iloc[i].loc["Laser2"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                if not isinstance(laser2, str):
                    continue
                else:
                    num_lasers = 2
                bp=str(bp)+".txt"
                laser1=str(laser1)+".txt"
                laser2=str(laser2)+".txt"
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/"+str(f))
                ldrv=(zf.open(leadRV))
                lsr1=(zf.open(laser1))
                if num_lasers!=2:
                    lsr2=(zf.open(laser2))
                else:
                    lsr2="lol"
                blood_pressure=(zf.open(bp))
                print("???????????????????????????????", period)
                if "Lead Noise" in period:
                    label = 0
                    print("noise")
                elif "Normal" in period:
                    label = 1
                    print("normal")
                elif "Baseline" in period:
                    label = 1
                    print("baseline")
                elif "VVI" in period:
                    label = 2
                    print("vvi")
                elif "AAI" in period:
                    label = 3
                    print("aai")
                elif "VT" in period:
                    label = 4
                    print("slow")
                elif "NSR" in period:
                    label = 1
                    print("nsr")
                out = platform.main(electrogram_path=ldrv,perfusion_path=lsr1, perfusion_path2=lsr2,bp_path=blood_pressure,period=label,num_lasers=num_lasers,extra=0)

                csv="/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/out"+str(flname)+str(period)+".csv"
                out.to_csv(csv, index=False)
        # else:
        #
        #
        #     for i in range(len(df)):
        #         zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/" + str(f))
        #         if  isinstance(df.iloc[i].loc["LeadRV"], str):
        #             leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
        #         else:
        #             if isinstance(df.iloc[i].loc["LeadShock"],str):
        #                 leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
        #             else:
        #                 leadRV = str(df.iloc[i].loc["ECG"]) + ".txt"
        #         period = df.iloc[i].loc["Period"]
        #         bp = df.iloc[i].loc["BP"]
        #         laser1 = df.iloc[i].loc["Laser1"]
        #         if not isinstance(bp, str):
        #             continue
        #         if not isinstance(laser1, str):
        #             continue
        #         begin=int(df.iloc[i].loc["Begin"])
        #         print("BEGIN",begin)
        #         end = int(df.iloc[i].loc["End"])
        #         print("END", end)
        #         bp = str(bp) + ".txt"
        #         laser1 = str(laser1) + ".txt"
        #
        #         ldrv = (pd.read_csv(zf.open(leadRV)))[begin:end]
        #         ldrv = ldrv.to_csv("ldrv.txt",index=False)
        #
        #         lsr1 = (pd.read_csv(zf.open(laser1)))[begin:end]
        #         lsr1 = lsr1.to_csv("lsr1.txt",index=False)
        #         blood_pressure = (pd.read_csv(zf.open(bp)))[begin:end]
        #         blood_pressure = blood_pressure.to_csv("blood_pressure.txt",index=False)
        #         print("???????????????????????????????",period)
        #         if "Lead Noise" in period:
        #             label=0
        #             print("noise")
        #         elif "Normal" in period:
        #             label=1
        #             print("normal")
        #         elif "Baseline" in period:
        #             label=1
        #             print("baseline")
        #         elif "VVI" in period:
        #             label=2
        #             print("vvi")
        #         elif "AAI" in period:
        #             label=3
        #             print("aai")
        #         elif "VT" in period:
        #             label = 4
        #             print("slow")
        #         elif "NSR" in period:
        #             label =1
        #             print("nsr")
        #         out = platform.main(electrogram_path="ldrv.txt",perfusion_path="lsr1.txt",bp_path="blood_pressure.txt",period=label,extra=begin)
        #         tmp_out =out
        #         tmp.append(tmp_out)
        #         print("aaaaaaaa",len(tmp))
        #     if len(tmp)>0:
        #         print(len(tmp))
        #         combined_csv = pd.concat(tmp)
        #
        #     csv = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/out" + str(flname) + ".csv"
        #     combined_csv.to_csv(csv, index=False)

sys.exit()

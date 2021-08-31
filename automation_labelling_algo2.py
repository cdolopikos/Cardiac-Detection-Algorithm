import math
import sys

# import klaus as platform
import algo_v9 as platform
import numpy as np
import pandas as pd
import zipfile
import glob, os

database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/BIG_DB.csv")
path= "/Users/cmdgr/OneDrive - Imperial College London/pr_data/New_unzipped"
count = 0
os.chdir(path)
score =0
names = database["File"]
tempo=[]
for name in names:
    if name not in tempo:
        tempo.append(name)

for f in glob.glob("*"):
    tmp = []
    flname = f.lower().split(".zip")[0]
    # print("f", f)
    count += 1
    df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "ECG","Period", "Begin", "End","BP", "Laser1", "Laser2", "Rhythm"]]
    # print(df["Rhythm"])

    if df.empty:
        continue
    else:
        if len(df)<=1:
            num_lasers=1
            for i in range(len(df)):
                shock = str(df.iloc[i]["Rhythm"])
                print(shock)
                # if "No shock" in shock:
                #     shock="No shock"
                # elif "Shock" in shock:
                #     shock = "Shock"
                print("AAAAAAAAAAAAA",shock)
                if isinstance(df.iloc[i].loc["LeadRV"],str) or not np.isnan(df.iloc[i].loc["LeadRV"]):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                elif isinstance(df.iloc[i].loc["LeadShock"],str) or not np.isnan(df.iloc[i].loc["LeadShock"]):
                    leadRV = str(df.iloc[i].loc["LeadShock"]) + ".txt"
                else:
                    leadRV = str(df.iloc[i].loc["ECG"]) + ".txt"
                period = df.iloc[i].loc["Period"]
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                laser2 = df.iloc[i].loc["Laser2"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                if isinstance(laser2, str):
                    num_lasers=2
                else:
                    num_lasers = 1
                bp=str(bp)+".txt"
                laser1=str(laser1)+".txt"
                laser2=str(laser2)+".txt"
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/New_unzipped/"+str(f))
                ldrv=(zf.open(leadRV))
                lsr1=(zf.open(laser1))
                if num_lasers==2:
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
                print(num_lasers)
                out = platform.main(electrogram_path=ldrv,perfusion_path=lsr1, perfusion_path2=lsr2,bp_path=blood_pressure,period=label,num_lasers=num_lasers,extra=0, treat = shock)
                if num_lasers ==1:
                    csv="/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/laser_1/out_"+str(flname)+"_laser"+str(num_lasers)+".csv"
                else:
                    csv="/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/laser_2/out_"+str(flname)+"_laser"+str(num_lasers)+".csv"
                out.to_csv(csv, index=False)
        else:
            # todo: rule based with perfusion
            # todo: svm + rule
            num_lasers = 1
            for i in range(len(df)):
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/New_unzipped/" + str(f))
                if  isinstance(df.iloc[i].loc["LeadRV"], str) or not np.isnan(df.iloc[i].loc["LeadRV"]):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                else:
                    if isinstance(df.iloc[i].loc["LeadShock"],str) or not np.isnan(df.iloc[i].loc["LeadShock"]):
                        leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
                    else:
                        leadRV = str(df.iloc[i].loc["ECG"]) + ".txt"
                shock = str(df.iloc[i]["Rhythm"])
                print(shock)
                # if "No shock" in shock:
                #     shock="No shock"
                # elif "Shock" in shock:
                #     shock = "Shock"
                period = df.iloc[i].loc["Period"]
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                laser2 = df.iloc[i].loc["Laser2"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                if isinstance(laser2, str):
                    num_lasers=2
                else:
                    num_lasers = 1
                if not np.isnan(df.iloc[i].loc["Begin"]):
                    begin=int(df.iloc[i].loc["Begin"])
                else:
                    begin=0
                print("BEGIN",begin)
                if not np.isnan(df.iloc[i].loc["End"]):
                    end = int(df.iloc[i].loc["End"])
                else:
                    end = len(leadRV)
                print("END", end)
                bp = str(bp) + ".txt"
                laser1 = str(laser1) + ".txt"
                laser2 = str(laser2) + ".txt"
                ldrv = (pd.read_csv(zf.open(leadRV)))[begin:end]
                ldrv = ldrv.to_csv("ldrv.txt",index=False)

                lsr1 = (pd.read_csv(zf.open(laser1)))[begin:end]
                lsr1 = lsr1.to_csv("lsr1.txt",index=False)
                if num_lasers==2:
                    lsr2 = (pd.read_csv(zf.open(laser2)))[begin:end]
                    lsr2 = lsr2.to_csv("lsr2.txt", index=False)
                else:
                    lsr2="lol"
                blood_pressure = (pd.read_csv(zf.open(bp)))[begin:end]
                blood_pressure = blood_pressure.to_csv("blood_pressure.txt",index=False)
                print("???????????????????????????????",period)
                if "Lead Noise" in period:
                    label=0
                    print("noise")
                elif "Normal" in period:
                    label=1
                    print("normal")
                elif "Baseline" in period:
                    label=1
                    print("baseline")
                elif "VVI" in period:
                    label=2
                    print("vvi")
                elif "AAI" in period:
                    label=3
                    print("aai")
                elif "VT" in period:
                    label = 4
                    print("slow")
                elif "NSR" in period:
                    label =1
                    print("nsr")
                out = platform.main(electrogram_path="ldrv.txt",perfusion_path="lsr1.txt", perfusion_path2="lsr2.txt",bp_path="blood_pressure.txt",period=label,extra=begin,num_lasers=num_lasers, treat=shock)
                tmp_out =out
                tmp.append(tmp_out)
                # print("aaaaaaaa",len(tmp))
            if len(tmp)>0:
                print(len(tmp))
                combined_csv = pd.concat(tmp)
            if num_lasers == 1:
                csv = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/laser_1/out_"+str(flname)+"_laser"+str(num_lasers)+".csv"
            else:
                csv = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/laser_2/out_"+str(flname)+"_laser"+str(num_lasers)+".csv"
            combined_csv.to_csv(csv, index=False)
    score = score+1
    print(score,"/",len(tempo))
    print(score/len(tempo))


sys.exit()

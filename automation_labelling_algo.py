# import math
# import sys
#
# import numpy as np
#
# import klaus as platform
# import pandas as pd
# import zipfile
# import glob, os
#
#
# database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces_unzipped_examples/database.csv")
# path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped"
# count = 0
# os.chdir(path)
# end, begin = np.nan, np.nan
# for f in glob.glob("*"):
#     flname = f.lower().split(".zip")[0]
#     count += 1
#     df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "ECG","Period", "Begin", "End","BP", "Laser1"]]
#     print("///",len(df))
#     if df.empty:
#         continue
#     else:
#         time_stamp = []
#         for i in range(len(df)):
#             print(df.iloc[i])
#             deci= df.iloc[i].loc["Period"]
#             if isinstance(df.iloc[i].loc["End"], int):
#                 end= df.iloc[i].loc["End"]
#
#             if isinstance(df.iloc[i].loc["Begin"], int):
#                 begin= df.iloc[i].loc["Begin"]
#             if not np.isnan(end) and not np.isnan(begin):
#                 time_stamp.append([deci,int(begin),int(end)])
#             else:
#                 time_stamp.append([deci,0, len(df.iloc[i])])
#             print("time_stamp",time_stamp)
#             print(time_stamp)
#             if "Noise" in deci:
#                 label=0
#             elif "Normal" in deci:
#                 label=1
#             elif "Baseline" in deci:
#                 label=1
#             elif "VVI" in deci:
#                 label=2
#             elif "AAI" in deci:
#                 label=2
#             elif "Slow" in deci:
#                 label = 3
#             elif "NSR" in deci:
#                 label =4
#
#         for i in range(len(df)):
#             if isinstance(df.iloc[i].loc["LeadRV"], str):
#                 leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
#                 leadShock = str(df.iloc[i].loc["LeadShock"]) + ".txt"
#             else:
#                 if isinstance(df.iloc[i].loc["LeadShock"], str):
#                     leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
#                 else:
#                     leadRV = str(df.iloc[i].loc["ECG"]) + ".txt"
#
#             bp = df.iloc[i].loc["BP"]
#             laser1 = df.iloc[i].loc["Laser1"]
#             if not isinstance(bp, str):
#                 continue
#             if not isinstance(laser1, str):
#                 continue
#             # leadShock=str(leadShock)+".txt"
#             bp=str(bp)+".txt"
#             laser1=str(laser1)+".txt"
#             zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/"+str(f))
#             # ldrv=(zf.open(leadRV))
#             ldrv=pd.read_csv(zf.open(leadRV))
#
#             s=time_stamp[i][1]
#             print(s)
#             x=time_stamp[i][2]
#             ldrv=ldrv.iloc[s:x]
#             ldrv.to_csv("ldrv.csv",index=False)
#             ldrv="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ldrv.csv"
#             # lsr1=(zf.open(laser1))
#             lsr1=pd.read_csv(zf.open(laser1))
#             lsr1 = lsr1.iloc[s:x]
#             lsr1.to_csv("lsr1.csv", index=False)
#             lsr1 = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/lsr1.csv"
#             blood_pressure=pd.read_csv(zf.open(bp))
#             blood_pressure = blood_pressure.iloc[s:x]
#             blood_pressure.to_csv("blood_pressure.csv", index=False)
#             blood_pressure = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/blood_pressure.csv"
#
#             # blood_pressure=(zf.open(bp))
#             out = platform.main(electrogram_path=ldrv, perfusion_path=lsr1, bp_path=blood_pressure, period=time_stamp[i][0], decision=label)
#             csv = "/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Preprocessed_data/out" + str(flname) + str(label) + ".csv"
#             out.to_csv(csv, index=False)
#             os.remove("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/ldrv.csv")
#
#
#
#
#
# sys.exit()

import math
import sys

# import klaus as platform
import algo_v5 as platform
import pandas as pd
import zipfile
import glob, os


# zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/A Tach/Atach_CRTD_21_10_2020_164349_csv.zip")
# df = pd.read_csv(zf.open('intfile.csv'))
# print(zf.infolist())
database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/database.csv")
# database.set_index("File", inplace=True)
# # print(database.columns)
# for i in database["File"]:
#     path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_unzipped_examples/"+i
path= "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped"
count = 0
os.chdir(path)
# for file in glob.glob("*"):
#     if file == "database.csv":
#         continue
#     else:
#         os.chdir("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces_zipped/" + file)

for f in glob.glob("*"):
    flname = f.lower().split(".zip")[0]
    print("f", f)
    count += 1
    df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "ECG","Period", "Begin", "End","BP", "Laser1"]]
    # print(database.loc[database["File"]==f,["LeadRV","LeadShock", "Period", "BP", "Laser1"]])
    print(df)
    if df.empty:
        continue
    else:
        if len(df)<=1:
            for i in range(len(df)):
                if  isinstance(df.iloc[i].loc["LeadRV"], str):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                else:
                    leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
                # leadShock = df.iloc[i].loc["LeadShock"]
                period = df.iloc[i].loc["Period"]
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                # leadShock=str(leadShock)+".txt"
                bp=str(bp)+".txt"
                laser1=str(laser1)+".txt"
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/"+str(f))
                ldrv=(zf.open(leadRV))
                # ldrv=pd.read_csv(zf.open(leadRV))
                lsr1=(zf.open(laser1))
                # print((lsr1))
                # lsr1=pd.read_csv(zf.open(laser1))
                # blood_pressure=pd.read_csv(zf.open(bp))
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
                elif "Slow VT" in period:
                    label = 4
                    print("slow")
                elif "NSR" in period:
                    label = 1
                    print("nsr")
                out = platform.main(electrogram_path=ldrv,perfusion_path=lsr1,bp_path=blood_pressure,period=label)

                # print(out)
                csv="/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/out"+str(flname)+str(period)+".csv"
                # out.to_csv(os.get(),index=False, encoding='utf-8-sig')
                out.to_csv(csv, index=False)
        else:
            for i in range(len(df)):
                zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Traces_zipped/" + str(f))
                if  isinstance(df.iloc[i].loc["LeadRV"], str):
                    leadRV = str(df.iloc[i].loc["LeadRV"])+".txt"
                else:
                    if isinstance(df.iloc[i].loc["LeadShock"],str):
                        leadRV=str(df.iloc[i].loc["LeadShock"])+".txt"
                    else:
                        leadRV = str(df.iloc[i].loc["ECG"]) + ".txt"
                # print(leadRV)
                period = df.iloc[i].loc["Period"]
                bp = df.iloc[i].loc["BP"]
                laser1 = df.iloc[i].loc["Laser1"]
                if not isinstance(bp, str):
                    continue
                if not isinstance(laser1, str):
                    continue
                # if isinstance(df.iloc[i].loc["Begin"], int):
                begin=int(df.iloc[i].loc["Begin"])
                # if isinstance(begin)
                #     begin=0
                print("BEGIN",begin)
                # if isinstance(df.iloc[i].loc["End"],int):
                end = int(df.iloc[i].loc["End"])
                # else:
                #     end=len(pd.read_csv(zf.open(leadRV)))
                print("END", end)
                # print(end)
                bp = str(bp) + ".txt"
                laser1 = str(laser1) + ".txt"

                ldrv = (pd.read_csv(zf.open(leadRV)))[begin:end]
                ldrv = ldrv.to_csv("ldrv.txt",index=False)

                # ldrv=pd.read_csv(zf.open(leadRV))
                # lsr1 = (zf.open(laser1))
                lsr1 = (pd.read_csv(zf.open(laser1)))[begin:end]
                lsr1 = lsr1.to_csv("lsr1.txt",index=False)
                # lsr1=pd.read_csv(zf.open(laser1))
                # blood_pressure=pd.read_csv(zf.open(bp))
                # blood_pressure = (zf.open(bp))
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
                elif "Slow VT" in period:
                    label = 4
                    print("slow")
                elif "NSR" in period:
                    label =1
                    print("nsr")
                # print(pd.read_csv("ldrv.txt"))
                out = platform.main(electrogram_path="ldrv.txt",perfusion_path="lsr1.txt",bp_path="blood_pressure.txt",period=label)
                csv = "/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/out" + str(flname) + str(
                    period) + ".csv"
                print("SHAPE",out.shape)
                # out.to_csv(os.get(),index=False, encoding='utf-8-sig')
                out.to_csv(csv, index=False)


sys.exit()

            # if (out.to_csv(csv , index=False)):
            #     print("??????????????????????NAI GTXM??????????????????")
            # else:
            #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # csv.to_csv(output, )

        # print(str(leadRV),"opaaa!!!!!!!!!!!!!!!!!!!!!!!!!")

        # if f.lower().endswith('.zip'):
        #     flname = f.lower().split(".zip")[0]
        #     print("fname",f.lower().split(".zip")[0])
        #     cwd = os.getcwd()
        #     # print(cwd+"/"+f)
        #     zf = zipfile.ZipFile(cwd+"/"+f)
        #     # print(database["File"])
        #     if f in database["File"]:
        #         print("ok")
cfghv
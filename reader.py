import math
import sys

import chris
import pandas as pd
import zipfile
import glob, os


# zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/A Tach/Atach_CRTD_21_10_2020_164349_csv.zip")
# df = pd.read_csv(zf.open('intfile.csv'))
# print(zf.infolist())
database = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/database.csv")
# database.set_index("File", inplace=True)
# # print(database.columns)
# for i in database["File"]:
#     path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces/"+i
path="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1"
count = 0
os.chdir(path)
# for file in glob.glob("*"):
#     if file == "database.csv":
#         continue
#     else:
#         os.chdir("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/" + file)
for f in glob.glob("*"):
    flname = f.lower().split(".zip")[0]
    print("f", f)
    count += 1
    df = database.loc[database["File"] == f, ["LeadRV", "LeadShock", "Period", "BP", "Laser1"]]
    # print(database.loc[database["File"]==f,["LeadRV","LeadShock", "Period", "BP", "Laser1"]])
    print(df)
    if df.empty:
        continue
    else:
        print(len(df))
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
            zf = zipfile.ZipFile("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/Traces1/"+str(f))
            ldrv=(zf.open(leadRV))
            # ldrv=pd.read_csv(zf.open(leadRV))
            lsr1=(zf.open(laser1))
            # lsr1=pd.read_csv(zf.open(laser1))
            # blood_pressure=pd.read_csv(zf.open(bp))
            blood_pressure=(zf.open(bp))
            out = chris.main(electrogram_path=ldrv,perfusion_path=lsr1,bp_path=blood_pressure,period=period)

            print(out)
            csv="/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/out"+str(flname)+str(period)+".csv"
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

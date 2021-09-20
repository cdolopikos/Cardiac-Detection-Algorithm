import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set01_200_16_02_2021_121627_.csv")
df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/combined/out_vtfi0015_vvi_set02_140_16_02_2021_125253__laser2.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0014_vvi_160_01_09_02_2021_122951_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0014_vvi_180_01_09_02_2021_122759_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0014_vvi_200_01_09_02_2021_123325_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set01_120_16_02_2021_120139_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set01_140_16_02_2021_120646_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set01_160_16_02_2021_121140_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set01_180_16_02_2021_115521_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set02_120_16_02_2021_124811_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set02_140_16_02_2021_125253_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set02_160_16_02_2021_125848_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0015_vvi_set02_180_16_02_2021_124323_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0016_vvi120_01_16_03_2021_151232_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0016_vvi140_01_16_03_2021_151622_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0016_vvi160_01_16_03_2021_152035_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0016_vvi180_01_16_03_2021_152428_.csv")
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0016_vvi200_01_16_03_2021_152736_.csv")
# todo df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0017_rv120_01_27_04_2021_150734_.csv")
# todo df= pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0017_test_aai_his_27_04_2021_150217_.csv")
# todo df= pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0017_test_aai120_27_04_2021_145039_.csv")
# todo df= pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0017_test_aai140_27_04_2021_145324_.csv")

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/combined_csv.csv")
x=df["BPM"]
y=df["BP Estimat"]
df=df[[ "BP Estimat","BPM", "Per STD", "Diagnosis"]]

# plt.scatter(df['Max Actual BP'], df['BP Estimat'])
# plt.scatter(df['Max Actual BP'], np.log10(df["Current Perfusion Grad"] * df["Quality of Perfusion"] * df["Perfusion Amplitude"] * df["R-R Interval RV"]))
# plt.show()
groups = df.groupby('Diagnosis')
for name, group in groups:
    print(name)
    print(group["BPM"])
    plt.plot(group["BPM"] , np.log10((group["BP Estimat"])), marker='o', linestyle='', markersize=1, label=name)

#
plt.legend()
plt.show()
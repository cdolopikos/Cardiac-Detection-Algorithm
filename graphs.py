import bp_detector
import data_reader
import electrogram_detector
import perfusion_detector
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from matplotlib.collections import LineCollection

df=pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/combined/out_vtfi0015_vvi_set02_140_16_02_2021_125253__laser2.csv")
bpm=df["BPM"]
rr_int=df["R-R Interval RV"]
ecgmean=df["EGM Mean RV"]
ecgstd = df["EGM STD RV"]
ecgskew=df["EGM Skewness RV"]
ecgkur=df["EGM Kurtosis RV"]
ecg_quality=df['EGM Quality']
bp_estimat=df['BP Estimat']
per_mean=df["Per Mean"]
per_std=df["Per STD"]
per_skew=df["Per Skewness"]
per_kurt=df["Per Kurtosis"]
per_grad=df["Current Perfusion Grad"]
qual_per=df["Quality of Perfusion"]
per_ampl=df["Perfusion Amplitude"]
magic=df["Magic Laser"]
diag=df["Diagnosis"]
figure, axs = plt.subplots(nrows=16,figsize=(15,30),constrained_layout=True)
figure.suptitle('Statistical values extracted', fontsize=16)
# axs[0].plot(bpm)

def plot_colourline(x,y,c,index,name):
    c = cm.jet((c-np.min(c))/(np.max(c)-np.min(c)))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        axs[index].plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
        axs[index].set_xlabel('Instance Number (Single Heartbeat)')
        axs[index].set_ylabel(name)
# fig = plt.figure(1, figsize=(5,5))
count=-1
for i in df.columns[:-2]:
    print(i)

    count = count + 1
    print(count)
    plot_colourline(x=np.arange(len(df[i])),y=df[i],c=diag,index=count,name=i)

    print(count)

# groups = df.groupby('Diagnosis')
# for name, group in groups:
#         # axs[0].plot(group["BPM"],label=name,linestyle='solid')
#         axs[0].plot(group["BPM"],marker='.')
        # axs[1].plot(rr_int,'r')
        # axs[2].plot(ecgmean,'r')
        # axs[3].plot(ecgstd,'r')
        # axs[4].plot(ecgskew,'r')
        # axs[5].plot(ecgkur,'r')
        # axs[6].plot(ecg_quality,'r')
        # axs[7].plot(bp_estimat,'r')
        # axs[8].plot(per_mean,'r')
        # axs[9].plot(per_std,'r')
        # axs[10].plot(per_skew,'r')
        # axs[11].plot(per_kurt,'r')
        # axs[12].plot(per_grad,'r')
        # axs[13].plot(qual_per,'r')
        # axs[14].plot(per_ampl,'r')
        # axs[15].plot(magic,'r')
        # axs[0].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[1].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[2].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[3].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[4].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[5].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[6].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[7].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[8].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[9].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[10].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[11].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[12].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[13].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[14].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[15].set_xlabel('Instance Number (Single Heartbeat)')
        # axs[0].set_ylabel('BPM')
        # axs[1].set_ylabel('R-R Interval RV')
        # axs[2].set_ylabel('EGM Mean RV')
        # axs[3].set_ylabel('EGM STD RV')
        # axs[4].set_ylabel('EGM Skewness RV')
        # axs[5].set_ylabel('EGM Kurtosis RV')
        # axs[6].set_ylabel('EGM Quality')
        # axs[7].set_ylabel('BP Estimat')
        # axs[8].set_ylabel('Per Mean')
        # axs[9].set_ylabel('Per STD')
        # axs[10].set_ylabel('Per Skewness')
        # axs[11].set_ylabel('Per Kurtosis')
        # axs[12].set_ylabel('Current Perfusion Grad')
        # axs[13].set_ylabel('Quality of Perfusion')
        # axs[14].set_ylabel('Perfusion Amplitude')
        # axs[15].set_ylabel('Magic Laser')
    # else:
    #     axs[0].plot(bpm,'b')
    #     axs[1].plot(rr_int,'b')
    #     axs[2].plot(ecgmean,'b')
    #     axs[3].plot(ecgstd,'b')
    #     axs[4].plot(ecgskew,'b')
    #     axs[5].plot(ecgkur,'b')
    #     axs[6].plot(ecg_quality,'b')
    #     axs[7].plot(bp_estimat,'b')
    #     axs[8].plot(per_mean,'b')
    #     axs[9].plot(per_std,'b')
    #     axs[10].plot(per_skew,'b')
    #     axs[11].plot(per_kurt,'b')
    #     axs[12].plot(per_grad,'b')
    #     axs[13].plot(qual_per,'b')
    #     axs[14].plot(per_ampl,'b')
    #     axs[15].plot(magic,'b')
    #     axs[0].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[1].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[2].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[3].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[4].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[5].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[6].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[7].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[8].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[9].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[10].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[11].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[12].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[13].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[14].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[15].set_xlabel('Instance Number (Single Heartbeat)')
    #     axs[0].set_ylabel('BPM')
    #     axs[1].set_ylabel('R-R Interval RV')
    #     axs[2].set_ylabel('EGM Mean RV')
    #     axs[3].set_ylabel('EGM STD RV')
    #     axs[4].set_ylabel('EGM Skewness RV')
    #     axs[5].set_ylabel('EGM Kurtosis RV')
    #     axs[6].set_ylabel('EGM Quality')
    #     axs[7].set_ylabel('BP Estimat')
    #     axs[8].set_ylabel('Per Mean')
    #     axs[9].set_ylabel('Per STD')
    #     axs[10].set_ylabel('Per Skewness')
    #     axs[11].set_ylabel('Per Kurtosis')
    #     axs[12].set_ylabel('Current Perfusion Grad')
    #     axs[13].set_ylabel('Quality of Perfusion')
    #     axs[14].set_ylabel('Perfusion Amplitude')
    #     axs[15].set_ylabel('Magic Laser')
# plt.subplot_tool()
# plt.subplots_adjust(top=0.1, bottom=0.05)
# fig.tight_layout()

# plt.show()
plt.savefig("paok.jpg",dpi=320)

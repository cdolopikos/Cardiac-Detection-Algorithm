import os,glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.fft import fft, ifft,fftfreq
import scipy.signal

# path = "/Users/cmdgr/Dropbox/Imperial_ILR/2020-06-28 Experiments/raw_data"
path = "/Users/cmdgr/Dropbox/Imperial_ILR/LDF_17_11_20"
os.chdir(path)

for f in glob.glob("*"):
    flname = f.lower().split(".txt")[0]
    path_file=path+"/"+f
    df = pd.read_csv(path_file, header=None, index_col=False)
    for i in range(1,(len(df.columns)-1)):
        df = df[i][1:]
        df = np.array(df).flatten()
        # print(df.head())
        # f, t, sx = scipy.signal.spectrogram(df, nfft=4096, fs=25000)

        print("hello")





        N = len(df)
        T = 1/25000
        x = np.linspace(0.0, N * T, N, endpoint=False)
        y = df
        print(flname,y)
        yf = fft(y)
        print(flname)
        print("yf",yf)
        # yf = [x for x in yf if np.isnan(x) == False]
        # print("yf",yf)
        xf = fftfreq(N, T)[:N // 2]
        plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
        plt.title(flname+str(i))
        plt.grid()
        plt.show()

        # fft_data = 2.0 / N * np.abs(yf[0:N // 2])
        # fft_data=pd.DataFrame(fft_data)
        # fft_data.to_csv("/Users/cmdgr/Dropbox/Imperial_ILR/2020-06-28 Experiments/fft_data/"+flname+"_fft_data.csv")

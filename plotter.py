import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas import Series

PERFUSION_PATH = "/Traces_unzipped_examples/A Tach/Atach_CRTD_21_10_2020_164349_/qfin.txt"
data = pd.read_csv(
    "/Traces_unzipped_examples/Normal/normalbostoncrtd_11_11_2020_144740_/qfin.txt")
data = pd.read_csv(PERFUSION_PATH)
data = data[0:1200]
data= data.to_numpy()
print(data.shape)
data=data.flatten()
print(data.shape)
data=(data - min(data)) / (max(data) - min(data))
window_size=50
window = np.ones(window_size)/float(window_size)
out=np.sqrt(np.convolve(data, window, 'valid'))

plt.plot(data)
plt.plot(out)
# # Change series to data frame and add convolved data as a new column
# series = Series.to_frame(series)
# series['convolved_data'] = convolved_data
#
# # Plot both original and smooth data
# plt.ylabel('Temperature (C)')
# plt.xlabel('Time')
# plt.title('Daily Minimum Temperatures in Melbourne (1981-1990)')
# plt.plot(series)
# plt.legend(('Original', 'Smooth'))
plt.show()
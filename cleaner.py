import pandas as pd
import numpy as np

df = pd.read_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/dk_vf_db.csv")

df['Patient'].replace('', np.nan, inplace=True)
df.dropna(subset=['Patient'], inplace=True)

df.to_csv("/Users/cmdgr/Dropbox/AAD-Documents/Traces/vf_cleaned.csv")
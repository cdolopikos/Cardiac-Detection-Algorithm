# load model and scaler and make predictions on new data
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import load
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# prepare dataset
df=pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/cleaned_data.csv")
# data = df[["Max Actual BP", "Current Perfusion Grad"]]
x=np.array(df["Current Perfusion Grad"]).reshape(-1,1)[:5000]
y=df["Max Actual BP"][:5000]
# load the model
model = load(open('model.pkl', 'rb'))
# load the scaler
count=0
tmp=[]
tmp1=[]
for i in range(len(x)):
    y_pred = model.predict([x[i]])
    y_true=y[i]
    print(y_pred,y[i],x[i])
    tmp.append(y_pred)
    tmp1.append(y_true)

plt.scatter(x, y)
plt.plot(x, tmp,"r")
plt.show()
print("acc",count/len(x))
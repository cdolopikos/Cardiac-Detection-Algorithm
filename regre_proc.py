import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
#
# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/combined_csv.csv")
# print(df.head(5))
#
# data = df[["Max Actual BP", "Current Perfusion Grad"]]
# print(data.head(5))
# print(len(data))
# print(data.shape)
# for i in range(len(data)):
#     print(i)
#     iter_data = data.iloc[i]
#     bp = iter_data[0]
#     theta = iter_data[1]
#     #  ann to theta einai mikro kai to bp megalo --> eimai ok
#     # bp>100 & theeta <45
#     # if 100<bp<200 and theta<45:
#     #     print("")
#     if theta<25:
#         df=df.drop(i)
#         continue
#     if bp<100:
#         print("bp<100")
#         if theta < 45:
#             print("theta < 45")
#             df=df.drop(i)
#     elif bp> 90:
#         print("bp > 90")
#         if theta > 60:
#             print("theta < 60")
#             df=df.drop(i)
#
#     # elif 40<bp<100 and theta>45:
#     #     print("")
# df.to_csv("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/cleaned_data.csv")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from pickle import dump

df=pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/!Project/AAD_1/idk/cleaned_data.csv",index_col=False)
# data = df[["Max Actual BP", "Current Perfusion Grad"]]
x=np.array(df["Current Perfusion Grad"]).reshape(-1,1)
y=df["Max Actual BP"]

poly = PolynomialFeatures(2, include_bias=False)
poly.fit_transform(x)
# model = LinearRegression(fit_intercept=True)
poly_model = make_pipeline(PolynomialFeatures(6),LinearRegression())
poly_model.fit(x, y)
# print(poly_model.decision_function)
# print(poly_model.get_params())

# xfit = np.linspace(0, 10, 1000)
yfit = poly_model.predict(x)
dump(poly_model, open('model.pkl', 'wb'))
plt.scatter(x, y)
plt.plot(x, yfit,"r")
plt.show()



# print("Model slope:    ", poly_model.coef_[0])
# print("Model intercept:", poly_model.intercept_)



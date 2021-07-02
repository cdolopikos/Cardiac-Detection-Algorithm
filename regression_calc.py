import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from pickle import dump
import matplotlib.pyplot as plt

df=pd.read_csv("/Preprocessed_data/cleaned_data.csv")
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



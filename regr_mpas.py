import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv("/Volumes/mc1/test_weka.csv", index_col=False)
print(df.head(10))
y = np.array(df["BP"])
x = np.array(df["LS"]).reshape(-1,1)

transformer = PolynomialFeatures(degree=2, include_bias=False)
transformer.fit(x)
x_ = transformer.transform(x)
x_ = PolynomialFeatures(degree=2, include_bias=False).fit_transform(x)
print(x_)
model = LinearRegression().fit(x_, y)
r_sq = model.score(x_, y)

print('intercept:', model.intercept_)
print('coefficients:', model.coef_)
y_pred = model.predict(x_)
print(len(x))
print(len(y_pred))
print('predicted response:', y_pred)
for i in range(len(y_pred)):
    print("y_pred",y_pred[i]*100*(abs(model.coef_ + 1)))
    print("BP",x[i]*100)
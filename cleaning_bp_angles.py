import pandas as pd
import numpy as np

# import concat
# import automation_labelling_algo
# import regression_calc
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

#
df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/training_combined_csv.csv")
print(df.head(5))

data = df[["Max Actual BP", "Current Perfusion Grad"]]
print(data.head(5))
print(len(data))
print(data.shape)
for i in range(len(data)):
    print(i)
    iter_data = data.iloc[i]
    bp = iter_data[0]
    theta = iter_data[1]
    #  ann to theta einai mikro kai to bp megalo --> eimai ok
    # bp>100 & theeta <45
    # if 100<bp<200 and theta<45:
    #     print("")
    if theta < 25:
        df = df.drop(i)
        continue
    if bp < 20:
        df = df.drop(i)
    elif bp < 90:
        print("bp<100")

        if theta < 50:
            print("theta < 45")
            df = df.drop(i)
    elif bp > 90:
        print("bp > 90")
        if theta > 60:
            print("theta < 60")
            df = df.drop(i)


    # elif 40<bp<100 and theta>45:
    #     print("")
df.to_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/cleaned_data.csv")

# regression_calc.regre_proc()
# automation_labelling_algo.reader()
# concat.concat()

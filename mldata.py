import os

import torch
import pandas as pd
from sklearn.model_selection import train_test_split


path = os.path.dirname(os.getcwd()) + '/AAD_1/Preprocessed_data/combined_csv.csv'


# gets preprocessed data from csv and returns a dataframe
def setData(pth):
    df = pd.read_csv(pth, dtype='float32')
    df = df.replace('#', '')
    return df


df = setData(path)
# print(df.shape)

# x gets all attributes
x = df.iloc[:, 0:(len(df.columns) - 1)]
# y gets all labels
y = df.iloc[:, -1]
print(y)
# split x and y into training and test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=66)

# for i in range(len(y)):
#     print(y[i])

# get functions
def getDataframe():
    return df


def getAttribute_train():
    return x_train


def getAttribute_test():
    return x_test


def getLabel_train():
    return y_train


def getLabel_test():
    return y_test


def getAll_attributes():
    return x


def getAll_labels():
    return y


def pdtoTensor(data):
    train = pd.read_csv(data)
    data = torch.tensor(train.values)
    return data


def getAnnDataX_train():
    x_data = torch.tensor(x_train.values)
    return x_data


def getAnnDataX_test():
    x_data = torch.tensor(x_test.values)
    return x_data


def getAnnDataY_train():
    y_data = torch.tensor(y_train.values)
    return y_data


def getAnnDataY_test():
    y_data = torch.tensor(y_test.values)
    return y_data

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
import mldata
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from pickle import dump
import pandas as pd
import numpy

# X = mldata.getAll_attributes()
# y = mldata.getAll_labels()

# Multilayer Perceptron initially used for configuration and testing
def mlpDev():
    X = mldata.getAll_attributes()
    y = mldata.getAll_labels()
    # print(y,"gamw tin poutana mou")
    data = mldata.df

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    # sc = StandardScaler()
    # X_train_ar = sc.fit_transform(X_train.values)
    X_train = pd.DataFrame(X_train, index=X_train.index, columns=X_train.columns)
    # X_test_array = sc.transform(X_test.values)
    X_test = pd.DataFrame(X_test, index=X_test.index, columns=X_test.columns)
    # test_data = pd.DataFrame([X_test, y_test])
    # test_data.to_csv("test_data.csv")
    # Initializing the multilayer perceptron
    # mlp = MLPClassifier(hidden_layer_sizes=(50,25,11,7,4), solver='lbfgs', learning_rate_init=0.65, max_iter=1000000)
    # mlp = MLPClassifier(hidden_layer_sizes=(56,28,14,7), solver='lbfgs', learning_rate_init=0.65, max_iter=100000)
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128,64,32,8), solver='lbfgs', learning_rate_init=0.5, max_iter=1000)
    mlp.fit(X_train, y_train)
    # for i in y_train:
        # print(i)
    # print(mlp.get_params())
    # calibrator = CalibratedClassifierCV(mlp, cv=10)
    # calibrator.fit(X_train, y_train)
    # Outputs:
    pred = mlp.predict(X_test)
    # count = 0
    # for i, z in zip(y_test, pred):
    #     if i == z:
    #         count += 1
    # score = count / len(pred)
    # print(score)
    print("floki",mlp.score(X_test, y_test))
    # for i in y_test:
        # print(i)
    print("accuracy",accuracy_score(y_test,pred))
    cm = confusion_matrix(y_test,pred)
    precision = precision_score(y_test, pred, average="macro")
    print('Precision: %f' % precision)
    recall = recall_score(y_test, pred, average="macro")
    print("recall",recall)
    return cm

xi=[]
for i in range(5):
    print(i)
    x= mlpDev()
    if i == 0:
        xi=x
    else:
        xi=xi+x
    print(xi)


# Multilayer Perceptron setup
def mlp(att, lbl):
    print("Multilayer Perceptron is setting up")
    # Standarization to have a faster classifier
    sc = StandardScaler()
    attribute = sc.fit_transform(att)
    att_array = sc.fit_transform(attribute)
    # X_train = pd.DataFrame(att_array, index=att.index, columns=att.columns)

    # Initializing the multilayer perceptron
    mlp = MLPClassifier(hidden_layer_sizes=(512,256,128,64,32,8), solver='lbfgs', learning_rate_init=0.5, max_iter=1000)
    mlp.fit(att_array, lbl)
    dump(mlp, open('mlp.pkl', 'wb'))
    print("Multilayer Perceptron setting up is finished")
    # return mlp

# mlp(X,y)

from pickle import dump, load
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt


# the whole signal is logged and takes min to max




#SVM for development purposes such as parameter tuning
def svmDev(i):


    model = svm.SVC(kernel='rbf', C=10, gamma=0.1)
    print("Training model.")
    # train model
    model.fit(features_matrix, labels)
    predicted_labels = model.predict(test_feature_matrix)
    print("FINISHED classifying. accuracy score : ")
    print(accuracy_score(test_labels, predicted_labels))
    print(len(test_labels))
    if i ==1:

        cm=confusion_matrix(test_labels, predicted_labels, labels=("No shock", "Shock"))
    elif i ==2:
        cm=confusion_matrix(test_labels, predicted_labels, labels=(0,1,2,3,4))
    print("cm", cm)
    # dump(model, open('svm.pkl', 'wb'))



# md = load(open('svm.pkl', 'rb'))
# for i in range(len(X)):
#     x=np.array(X.iloc[i]).reshape((1,-1))
#     pred=md.predict(x)
#     if pred==Y.iloc[i]:
#         count=count+1
# print(count/len(Y))
#Actual svm setup
def supvm(att, lbl,i):
    print("Support Vector Machine is setting up")
    # sc = StandardScaler()
    # X = sc.fit_transform(att)
    features_matrix = X
    labels = lbl
    model = svm.SVC(kernel='rbf', C=10, gamma=0.1, probability=True)
    # train model
    model.fit(att, lbl)
    print("Support Vector Machine setting up is finished")
    dump(model, open('svm_laser_laser2'+str(i)+'.pkl', 'wb'))
    # return model





X = mldata.getAll_attributes()
# Y = mldata.y
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
for i in range(1, 3):
    if i == 1:
        # X = mldata.x1
        Y = mldata.y
    # Y = mldata.y1
    elif i == 2:
        print("Nai gamw tin poutana sou")
        Y = mldata.y1
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
    sc = StandardScaler()
    # X_train = sc.fit_transform(X_train)
    # X_test = sc.fit_transform(X_test)

    features_matrix = X_train
    # features_matrix = X
    # labels = Y
    labels = Y_train
    model = svm.SVC()
    test_feature_matrix = X_test
    test_labels = Y_test
    for j in range(0, 10):
        svmDev(i)
    supvm(features_matrix, labels,i)
#     print(i)

# rbf_feature = RBFSampler(gamma=0.1, random_state=1)
# X_features = rbf_feature.fit_transform(X)
# clf = SGDClassifier(max_iter=5)
# clf.fit(X_train, Y_train)
#
# print("poutana",clf.score(X_test, Y_test))
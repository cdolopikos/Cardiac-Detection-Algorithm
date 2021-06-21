import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import mldata

param = []
# Random Forest Classifier setup for parameter tuning based on existing data
def rdfDev():
    x_train = mldata.getAttribute_train()
    x_test = mldata.getAttribute_test()
    y_train = mldata.getLabel_train()
    y_test = mldata.getLabel_test()
    y = mldata.getAll_labels()
    x = mldata.getAll_attributes()
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_predict = rfc.predict(x_test)

    rfc_cv_score = cross_val_score(rfc, x, y, cv=10, scoring='accuracy')

    # number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # number of features at every split
    max_features = ['auto', 'sqrt']
    # max depth
    max_depth = [int(x) for x in np.linspace(100, 500, num=11)]
    max_depth.append(None)
    # create random grid
    random_grid = {'n_estimators': n_estimators, 'max_features': max_features, 'max_depth': max_depth}
    # Random search of parameters
    rfc_random = RandomizedSearchCV(estimator=rfc, param_distributions=random_grid, n_iter=1, cv=10, verbose=2,
                                    random_state=42, n_jobs=-1)
    # Fit the model
    rfc_random.fit(x_train, y_train)
    # print results

    rfc = RandomForestClassifier(n_estimators=rfc_random.best_params_['n_estimators'],
                                 max_depth=rfc_random.best_params_['max_depth'],
                                 max_features=rfc_random.best_params_['max_features'])
    param = [rfc_random.best_params_['n_estimators'],rfc_random.best_params_['max_depth'],rfc_random.best_params_['max_features']]
    rfc.fit(x_train, y_train)
    print("n_estimators", rfc_random.best_params_['n_estimators'])
    print("max_depth", rfc_random.best_params_['max_depth'])
    print("max_features", rfc_random.best_params_['max_features'])
    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, rfc_predict))
    print('\n')
    print("=== Classification Report ===")
    print(classification_report(y_test, rfc_predict))
    print('\n')
    print("=== All AUC Scores ===")
    print(rfc_cv_score)
    print('\n')
    print("=== Mean AUC Score ===")
    print("Mean AUC Score - Random Forest: ", rfc_cv_score.mean())
    # cm = confusion_matrix(rfc_predict, y_test, labels=(1, 2, 3, 4, 5))
    # print("cm", cm)
    return rfc


rdfDev()


# Random Forest Classifier setup later used
def randonf(att, lbl):
    print("Random Forest is setting up")
    sc = StandardScaler()
    attribute = sc.fit_transform(att)
    rfc = rdfDev()
    rfc.fit(attribute, lbl)
    print("Random Forest setting up is finished")
    return rfc

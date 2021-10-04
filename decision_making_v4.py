from pickle import load, dump
from collections import deque
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

import mldata

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/realistic test/outvtfi0014_vvi_160_01_09_02_2021_122951_.csv")
instances=mldata.x
diagnosis = mldata.y1
treatment = mldata.y
tmp=[]
tmp1=[]

# todo prwta apo ola thes na fereis mesa raw data kai meta na ta kaneis preprocess


# Possible outcomes for the decison is 0 --> Noise, 1--> Normal, 2--> VVI, 3 --> AAI, 4--> VT

def getML_treatment(instance):
    ml = load(open('svm_treatment_based.pkl', 'rb'))
    ml_based_treatment = ml.predict(instance)
    ml_based_prob = ml.predict_proba(instance)

    # print("Paok",ml_based_treatment,ml_based_prob)
    return (ml_based_treatment[0]),ml_based_prob


def getML_diagnosis(instance, label):
    print(instance)
    # ml = load(open('svm_laser_laser22.pkl', 'rb'))
    ml = load(open('svm_condition_based.pkl', 'rb'))
    ml_based_decision = ml.predict(instance)
    # ml.partial_fit(instance, label)
    # dump(ml, open('svm_condition_based.pkl', 'wb'))
    # print(ml_based_decision[0])
    return (ml_based_decision[0])

# Checks the heamodynamic Stability of the patient based on BP estimat biomarker


ct=0
sc = StandardScaler()
final_treat=""
ml_treat=0
ml_prob=0
for i in range(len(instances)):
    X=(np.array(instances.iloc[i]).reshape(1,-1))
    bpd = X[0][7]
    # print(X.shape)
    # print(X[0][7])
    # ml_dec=getML_decision(np.array(instances.iloc[i]).reshape(1,-1))
    ml_diag=getML_diagnosis(X, diagnosis[i])
    # print(ml_diag)
    # print(ml_treat, ml_prob)
    if ml_diag == 3 or ml_diag==4 or ml_diag==2 or ml_diag==5:
        # print("Hooray")
        ml_treat, ml_prob=getML_treatment(X)
        # final_treat=ml_treat
        if ml_treat == "Shock" and bpd>0:
            # print("nai")
            final_treat="Shock"
        elif ml_treat == "No Shock" and bpd<0:
            final_treat = "No Shock"
        elif ml_treat == "Shock" and bpd<0:
            print(ml_prob[0][1], "xyn")
            # bpd = 1.6 * bpd

            prob=ml_prob[0][1]*-1
            if abs(prob) > 0.8:
                prob=prob*2
            else:
                prob=prob*1.5
            tmp_treat = bpd + prob
            if tmp_treat<0:
                final_treat="Shock"
            else:
                final_treat="No Shock"
        elif ml_treat == "No Shock" and bpd>0:
            # bpd = 1.3 * bpd
            if abs(bpd) > 1.5:
                bpd = 2.5*bpd
            # an to bpd einai poli egalitero tou 1 tote akou to gamidid to bpd
            prob = ml_prob[0][0]*2
            tmp_treat = bpd + prob
            if tmp_treat<0:
                final_treat="Shock"
            else:
                final_treat= "No Shock"

            # print(ml_prob[0][0], "floki")
            # print(ml_prob[0], "floki")
        # print(treatment[i], final_treat)
    else:
        final_treat="No Shock"
    if treatment[i] == final_treat:
        ct +=1
        tmp.append([X,ml_prob,ml_treat, ml_diag, diagnosis[i],final_treat, bpd, treatment[i]])
    else:
        tmp.append([X,ml_prob,ml_treat, ml_diag, diagnosis[i],final_treat, bpd, treatment[i]])
        tmp1.append([X,ml_prob,ml_treat, ml_diag, diagnosis[i],final_treat, bpd, treatment[i]])
    print(ct / len(treatment))

dt=pd.DataFrame(tmp)
dt1=pd.DataFrame(tmp1)
dt.to_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/tmp.csv")
dt1.to_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/tmp1.csv")

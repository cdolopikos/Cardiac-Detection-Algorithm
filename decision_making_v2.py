from pickle import load
from collections import deque
import pandas as pd
import numpy as np
import mldata

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/realistic test/outvtfi0014_vvi_160_01_09_02_2021_122951_.csv")
instances=mldata.x
diagnosis = mldata.y

# todo prwta apo ola thes na fereis mesa raw data kai meta na ta kaneis preprocess


# Possible outcomes for the decison is 0 --> Noise, 1--> Normal, 2--> VVI, 3 --> AAI, 4--> VT

def getML_decision(instance):
    ml = load(open('svm.pkl', 'rb'))
    ml_based_decision = ml.predict(instance)
    return int(ml_based_decision[0])

# Checks the heamodynamic Stability of the patient based on BP estimat biomarker
def haemodynamicStability(instance):
    mark= instance["BP Estimat"]
    if mark >0:
        stability = 1
    else:
        stability = 0
    return stability


def makeDecision(ml_decision, heam_stab, ecg_based):
    decision="n/a"
    if ml_decision == 0 or ml_decision == 1:
        decision = 0
    elif ml_decision==2 or ml_decision == 3 or ml_decision==4:
        if heam_stab==1:
            decision=1
        else:
            if ecg_based == 1:
                decision = 1
            else:
                decision = 0
    return decision

ecg_history=deque(maxlen=36)
def ecgBased(instance):
    bpm=instance["BPM"]
    rr_interval = instance["R-R Interval RV"]
    # ecg_quality=instance["EGM Quality"]
    # current = [bpm, rr_interval, ecg_quality]
    # ecg_history.append(current)
    if bpm > rr_interval/10:
        situation = 1
    else:
        situation =0
    # situation=1
    ecg_history.append(situation)
    death_score = sum(1 for i in ecg_history if i >0)
    if death_score > 28:
        decision = 1
    else:
        decision = 0
    return decision

ct=0
for i in range(len(instances)):
    ml_dec=getML_decision(np.array(instances.iloc[i]).reshape(1,-1))
    heam_stab=haemodynamicStability(instances.iloc[i])
    ecg_based= ecgBased(instances.iloc[i])
    final_decision=makeDecision(ml_dec, heam_stab, ecg_based)
    if diagnosis.iloc[i]== 4 or diagnosis.iloc[i]==2 or diagnosis.iloc[i]==3:
        if ml_dec == 1 or ml_dec==0:
            ct=ct+1
            print("!!!!!!!!!!!!!!!!", "diagnosis", diagnosis.iloc[i], "ml dec", ml_dec, "stab", heam_stab,"Ecg Based", ecg_based,"shock", final_decision, ct,
                  len(diagnosis))

    print("diagnosis", diagnosis.iloc[i], "ml dec", ml_dec, "stab", heam_stab,"Ecg Based", ecg_based, "shock", final_decision, ct, len(diagnosis))

ml = load(open('svm.pkl', 'rb'))
count=0
for i in range(len(instances)):
    x=np.array(instances.iloc[i]).reshape((1,-1))

    pred=ml.predict(x)
    if pred==diagnosis.iloc[i]:
        count=count+1
print(count/len(diagnosis))
print(len(ecg_history))
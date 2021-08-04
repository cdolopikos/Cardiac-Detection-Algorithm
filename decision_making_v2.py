from pickle import load
import pandas as pd
import numpy as np
import mldata

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/realistic test/outvtfi0014_vvi_160_01_09_02_2021_122951_.csv")
instances=mldata.x
diagnosis = mldata.y

print(instances)
print(diagnosis)
# todo prwta apo ola thes na fereis mesa raw data kai meta na ta kaneis preprocess
# todo ta preprocessed pou erxontai pernan tin eksis aksiologisi

# Possible outcomes for the decison is 0 --> Noise, 1--> Normal, 2--> VVI, 3 --> AAI, 4--> VT

def getML_decision(instance):
    ml = load(open('svm.pkl', 'rb'))
    ml_based_decision = ml.predict(instance)
    return ml_based_decision

# Checks the heamodynamic Stability of the patient based on BP estimat biomarker
def haemodynamicStability(instance):
    mark= instance["BP Estimat"]
    if mark >0:
        stability = 1
    else:
        stability = 0
    return stability

def makeDecision(ml_decision, heam_stab):
    if ml_decision == 0 or ml_decision == 1:
        decision = 0
    elif ml_decision==2 or ml_decision == 3 or ml_decision==4:
        if heam_stab==1:
            decision=1
        else:
            print("need to further investigate")
    return decision

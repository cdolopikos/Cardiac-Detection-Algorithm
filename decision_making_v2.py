from pickle import load
import pandas as pd
import numpy as np
import mldata

# df = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/realistic test/outvtfi0014_vvi_160_01_09_02_2021_122951_.csv")
instances=mldata.x
diagnosis = mldata.y

# todo prwta apo ola thes na fereis mesa raw data kai meta na ta kaneis preprocess
# todo ta preprocessed pou erxontai pernan tin eksis aksiologisi

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

def makeDecision(ml_decision, heam_stab):
    decision="n/a"
    if ml_decision == 0 or ml_decision == 1:
        decision = 0
    elif ml_decision==2 or ml_decision == 3 or ml_decision==4:
        if heam_stab==1:
            decision=1
        else:
            print("need to further investigate")
    return decision

for i in range(len(instances)):
    ml_dec=getML_decision(np.array(instances.iloc[i]).reshape(1,-1))
    heam_stab=haemodynamicStability(instances.iloc[i])
    adoume=makeDecision(ml_dec, heam_stab)
    print(diagnosis.iloc[i], ml_dec, heam_stab, adoume)

ml = load(open('svm.pkl', 'rb'))
count=0
for i in range(len(instances)):
    x=np.array(instances.iloc[i]).reshape((1,-1))

    pred=ml.predict(x)
    if pred==diagnosis.iloc[i]:
        count=count+1
print(count/len(diagnosis))
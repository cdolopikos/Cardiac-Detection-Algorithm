# import mldata, multilayerperceptron
from pickle import load
import pandas as pd
import numpy as np

# mlp = load(open('mlp.pkl', 'rb'))
# mlp.predict()
from sklearn.preprocessing import StandardScaler

# data = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/combined_csv.csv")
data = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/testing_combined_csv.csv")
# data = pd.read_csv("/Users/cmdgr/OneDrive - Imperial College London/pr_data/Preprocessed_data/outvtfi0014_vvi_180_01_09_02_2021_122759_Baseline.csv")
print(data.head(5))
print(data.columns)
dt = data.iloc[:, 0:(len(data.columns) - 1)]
label = data['Decision']
print(label)

def getRuleBasedDecision(theta, theta_threshold,per_amplitude, per_amplitude_threshold,ecg_histroy,bpm_threshold, rr_threshold):
    tmp=[]
    for i in ecg_histroy:
        if i != 0:
            bpm = i[0]
            rr_interval = i[1]
            if bpm < bpm_threshold and rr_interval<rr_threshold:
                tmp.append(0)
    if len(tmp)>=18:
        ecg_based_decision = "shock"
    elif 15<len(tmp)<18:
        ecg_based_decision="wait"
    elif len(tmp)<15:
        ecg_based_decision = "no-shock"

    if theta<theta_threshold and per_amplitude<per_amplitude_threshold:
        per_based_decision = 2
    elif theta<theta_threshold and not per_amplitude<per_amplitude_threshold:
        per_based_decision =1
    elif not theta<theta_threshold and per_amplitude<per_amplitude_threshold:
        per_based_decision =1
    else:
        per_based_decision =0
    if ecg_based_decision == "shock" and per_based_decision == 2:
        decision = "shock"
    elif ecg_based_decision == "shock" and per_based_decision == 1:
        decision = "shock"
    elif ecg_based_decision == "shock" and per_based_decision == 0:
        decision = "no-shock"
    elif ecg_based_decision=="wait" and per_based_decision == 2:
        decision = "shock"
    elif ecg_based_decision=="wait" and per_based_decision == 1:
        decision = "no-shock"
    elif ecg_based_decision=="wait" and per_based_decision == 0:
        decision = "no-shock"
    elif ecg_based_decision=="no-shock":
        if per_based_decision == 2:
            decision = "no-shock"
        if per_based_decision == 1:
            decision = "no-shock"
        if per_based_decision == 0:
            decision = "no-shock"

    return decision
# returns string

def getMLP_decision(instance):
    mlp = load(open('svm.pkl', 'rb'))
    mlp_based_decision = mlp.predict(instance)
    return mlp_based_decision
# return int

def cumulative_decsion(mlp_based_decision,ecgbased):
    if ecgbased == "no-shock":
        ecg_based_decision = 0
    elif ecgbased == "shock":
        ecg_based_decision = 1
    decision = mlp_based_decision * ecg_based_decision
    if decision == 1:
        decision = "shock"
    else:
        decision = "no-shock"
    return decision


ecg_history=[0]*20
results=[]
for i in range(len(data)):
    print(dt.iloc[i]["Per Mean"])
    # print()
    theta=dt.iloc[i]["Current Perfusion Grad"]
    per_amplitude =dt.iloc[i]["Perfusion Amplitude"]
    X = np.array(dt.iloc[i]).reshape(1,-1)
    # sc = StandardScaler()
    # x = pd.DataFrame(x, index=x.index, columns=x.columns)
    # X = sc.fit_transform(x)
    # X = pd.DataFrame(X, index=X.index, columns=X.columns)
    print(X)

    mlp_based_decision = getMLP_decision(X)
    ecg_history.append([dt.iloc[i]["BPM"],dt.iloc[i]["R-R Interval RV"]])
    ecgbased = getRuleBasedDecision(theta=theta,theta_threshold=15,per_amplitude=per_amplitude,per_amplitude_threshold=15,bpm_threshold=150,rr_threshold=350,ecg_histroy=ecg_history)
    cum_decsion = cumulative_decsion(mlp_based_decision,ecgbased)
    print(mlp_based_decision)
    print(data.iloc[i])
    print(cumulative_decsion)
    print(label[i])
    if label[i] == 0:
        lbl="Lead Noise"
    elif label[i] == 1:
        lbl="Normal"
    elif label[i] == 2:
        lbl = "VVI"
    elif label[i] == 3:
        lbl = "AAI"
    elif label[i] == 4:
        lbl = "Slow"
    results.append([dt.iloc[i], cum_decsion,lbl,ecgbased,mlp_based_decision])
results = pd.DataFrame(results,columns=["Data","Cum dec", "Actual Label", "ECG decision", "mlp decision"])

results.to_csv("vamos.csv")

import mldata, multilayerperceptron
from pickle import load

# mlp = load(open('mlp.pkl', 'rb'))
# mlp.predict()

def getRuleBasedDecision(theta, theta_threshold,per_amplitude, per_amplitude_threshold,ecg_histroy,bpm_threshold, rr_threshold):
    tmp=[]
    for i in ecg_histroy:
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
    mlp = load(open('mlp.pkl', 'rb'))
    mlp_based_decision = mlp.predict(instance)
    return mlp_based_decision
# return int

def cumulative_decsion(mlp_based_decision,ecgbased):
    if ecgbased == "no-shock":
        ecg_based_decision = 0
    elif ecgbased == "shock":
        ecg_based_decision = 1
    decision = mlp_based_decision * ecg_based_decision
    return decision
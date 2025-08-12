import numpy as np
import joblib

# Load pretrained models (make sure these are in same folder)
csp = joblib.load("csp_model.pkl")
lda = joblib.load("lda_model.pkl")

def extract_stat_features(X_csp):
    features = []
    for trial in X_csp:
        trial_feats = []
        for component in trial:
            trial_feats.extend([
                np.mean(component), np.var(component), np.std(component),
                np.max(component), np.min(component), np.median(component),
                np.sum(np.abs(component)),
            ])
        features.append(trial_feats)
    return np.array(features)

def classify_trial(trial, true_label):
    trial = trial[np.newaxis, :]  # Shape (1, channels, timepoints)
    trial_csp = csp.transform(trial)
    features = extract_stat_features(trial_csp)
    prediction = lda.predict(features)[0]

    action = "Open Hand" if prediction == 2 else "Close Hand"
    print(f"Prediction: {action} | Ground Truth: {true_label} | {'✅' if prediction == true_label else '❌'}")

if __name__ == "__main__":
    # Replace with whichever trial file you want to demo
    trial_file = "trial_1.npy"
    label_file = "trial_1_label.npy"

    trial = np.load(trial_file)
    true_label = np.load(label_file).item()  # scalar

    classify_trial(trial, true_label)

import numpy as np
import mne
from mne.datasets import eegbci
import joblib

# Load pretrained models
csp = joblib.load("csp_model.pkl")
lda = joblib.load("lda_model.pkl")

def load_and_preprocess(subject, runs=[4, 8, 12], channels=['C3', 'Cz', 'C4']):
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(channels)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.resample(250, npad='auto')
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id={'T1':1,'T2':2})
    epochs = mne.Epochs(raw, events, event_id={'T1':1,'T2':2}, tmin=0, tmax=2, baseline=None, preload=True, verbose=False)
    return epochs

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

def get_correct_trial_indices(X, y):
    correct_indices = []
    for i in range(len(X)):
        trial = X[i][np.newaxis, :]
        features = extract_stat_features(csp.transform(trial))
        pred = lda.predict(features)[0]
        if pred == y[i]:
            correct_indices.append(i)
    return correct_indices

if __name__ == "__main__":
    subject_id = int(input("Enter subject ID to check: "))
    epochs = load_and_preprocess(subject_id)
    X, y = epochs.get_data(), epochs.events[:, -1]

    correct_indices = get_correct_trial_indices(X, y)

    left_correct = [i for i in correct_indices if y[i] == 1]
    right_correct = [i for i in correct_indices if y[i] == 2]

    print(f"Subject {subject_id} - Correct Left Trials: {len(left_correct)} indices: {left_correct}")
    print(f"Subject {subject_id} - Correct Right Trials: {len(right_correct)} indices: {right_correct}")

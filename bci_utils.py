# bci_utils.py
#this contains  all the loading,preprocessing and extraction of statistiacal features
import numpy as np
import mne
from mne.datasets import eegbci

CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}

def load_data(subject, runs, sfreq=250):
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(CHANNELS)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    raw.resample(sfreq, npad='auto')
    return raw

def preprocess(raw):
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS)
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=2, baseline=None, preload=True, verbose=False)
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

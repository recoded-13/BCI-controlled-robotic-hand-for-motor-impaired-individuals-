import numpy as np
import mne
from mne.datasets import eegbci

# Settings for loading Subject 85 data
SUBJECT = 85
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}

def load_data(subject, runs, sfreq=250):
    files = eegbci.load_data(subject, runs)
    raws = [mne.io.read_raw_edf(f, preload=True) for f in files]
    raw = mne.concatenate_raws(raws)
    mne.datasets.eegbci.standardize(raw)
    raw.pick_channels(CHANNELS)
    raw.set_montage(mne.channels.make_standard_montage('standard_1020'))
    raw.resample(sfreq, npad='auto')
    return raw

def preprocess(raw):
    raw.filter(8, 30, fir_design='firwin', verbose=False)
    events, _ = mne.events_from_annotations(raw, event_id=EVENT_IDS)
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=2, baseline=None, preload=True, verbose=False)
    return epochs

# Load and preprocess Subject 85 data
raw = load_data(SUBJECT, RUNS)
epochs = preprocess(raw)
X = epochs.get_data()
y = epochs.events[:, -1]

# Save first two trials and their labels
np.save("trial_0.npy", X[0])
np.save("trial_0_label.npy", y[0])
np.save("trial_1.npy", X[41])
np.save("trial_1_label.npy", y[41])


print("Saved trial_0.npy, trial_0_label.npy, trial_1.npy, trial_1_label.npy")

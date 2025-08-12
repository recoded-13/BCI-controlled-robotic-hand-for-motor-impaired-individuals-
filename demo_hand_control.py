import numpy as np
import mne
from mne.datasets import eegbci
import joblib
import threading
import time
import queue

# Settings
TEST_SUBJECT = 77
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}

# Load pretrained models
csp = joblib.load("csp_model.pkl")
lda = joblib.load("lda_model.pkl")

# Queues for real-time threading
raw_data_queue = queue.Queue()
prediction_queue = queue.Queue()

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

def data_acquisition(X, y, indices=[0, 1]):
    print("👾 Starting demo acquisition (2 trials)...")
    for i in indices:
        raw_data_queue.put((X[i], y[i]))
        time.sleep(1.5)  # simulate 1.5s between inputs
    raw_data_queue.put(None)

def classification_thread():
    while True:
        item = raw_data_queue.get()
        if item is None:
            prediction_queue.put(None)
            break
        trial, true_label = item
        trial = trial[np.newaxis, :]
        trial_csp = csp.transform(trial)
        features = extract_stat_features(trial_csp)
        prediction = lda.predict(features)[0]
        prediction_queue.put((prediction, true_label))

def output_thread():
    correct = 0
    total = 0
    while True:
        result = prediction_queue.get()
        if result is None:
            acc = 100 * correct / total if total > 0 else 0
            print(f"\n🎯 Demo Accuracy: {acc:.2f}%")
            break
        pred, true = result
        total += 1
        correct += int(pred == true)
        action = "Open Hand" if pred == 2 else "Close Hand"
        print(f"[{total}] Prediction: {action} | Ground Truth: {true} → {'✅' if pred == true else '❌'}")
        # 🔧 Here you can add GPIO or serial commands

# Load Subject 85 and extract 2 trials
raw = load_data(TEST_SUBJECT, RUNS)
epochs = preprocess(raw)
X, y = epochs.get_data(), epochs.events[:, -1]

# Launch threads
t1 = threading.Thread(target=data_acquisition, args=(X, y, [41, 0]))  # first two trials

t2 = threading.Thread(target=classification_thread)
t3 = threading.Thread(target=output_thread)

t1.start()
t2.start()
t3.start()

t1.join()
t2.join()
t3.join()

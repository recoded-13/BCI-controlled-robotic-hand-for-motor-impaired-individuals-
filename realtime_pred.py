#Real-Time Streaming + Prediction Smoothing + Servo Control.
import numpy as np
import mne
from mne.datasets import eegbci
import joblib
import threading
import time
import queue
import RPi.GPIO as GPIO
import signal
import sys
from collections import deque

# === Settings ===
TEST_SUBJECT = 77
RUNS = [4, 8, 12]
CHANNELS = ['C3', 'Cz', 'C4']
EVENT_IDS = {'T1': 1, 'T2': 2}
SMOOTHING_WINDOW = 5  # Number of recent predictions for majority vote

# === GPIO Setup ===
servoPIN1 = 17
servoPIN2 = 18

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN1, GPIO.OUT)
GPIO.setup(servoPIN2, GPIO.OUT)

p1 = GPIO.PWM(servoPIN1, 50)
p2 = GPIO.PWM(servoPIN2, 50)

p1.start(2.5)  # start at 0 degrees (open)
p2.start(2.5)

def move_servos(angle):
    duty_cycle = 2.5 + (angle / 18)
    p1.ChangeDutyCycle(duty_cycle)
    p2.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)
    p1.ChangeDutyCycle(0)
    p2.ChangeDutyCycle(0)

# === Load pretrained models ===
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

# === Queues for threading ===
raw_data_queue = queue.Queue()
prediction_queue = queue.Queue()

# === Real-time acquisition simulation ===
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
    epochs = mne.Epochs(raw, events, event_id=EVENT_IDS, tmin=0, tmax=2,
                        baseline=None, preload=True, verbose=False)
    return epochs

def data_acquisition(X, y):
    print("👾 Starting simulated real-time data acquisition...")
    for i in range(len(X)):
        raw_data_queue.put((X[i], y[i]))
        time.sleep(1.0)  # adjust speed here
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
    prediction_history = deque(maxlen=SMOOTHING_WINDOW)
    correct = 0
    total = 0
    last_action = None

    while True:
        result = prediction_queue.get()
        if result is None:
            acc = 100 * correct / total if total > 0 else 0
            print(f"\n🎯 Final Accuracy: {acc:.2f}%")
            cleanup()
            break

        pred, true = result
        prediction_history.append(pred)

        # Majority vote smoothing
        if len(prediction_history) == SMOOTHING_WINDOW:
            counts = {1:0, 2:0}
            for p in prediction_history:
                counts[p] += 1
            smoothed_pred = max(counts, key=counts.get)
        else:
            smoothed_pred = pred  # fallback if window not full

        total += 1
        if smoothed_pred == true:
            correct += 1

        action = "Open Hand" if smoothed_pred == 2 else "Close Hand"
        print(f"[{total}] Smoothed Prediction: {action} | Ground Truth: {true} | {'✅' if smoothed_pred == true else '❌'}")

        # Move servo only if action changes
        if action != last_action:
            if smoothed_pred == 2:
                move_servos(0)
            else:
                move_servos(180)
            last_action = action

def cleanup():
    print("\nCleaning up GPIO and exiting...")
    p1.stop()
    p2.stop()
    GPIO.cleanup()
    sys.exit(0)

def signal_handler(sig, frame):
    cleanup()

signal.signal(signal.SIGINT, signal_handler)

# === Main ===
if __name__ == "__main__":
    raw = load_data(TEST_SUBJECT, RUNS)
    epochs = preprocess(raw)
    X, y = epochs.get_data(), epochs.events[:, -1]

    t1 = threading.Thread(target=data_acquisition, args=(X, y))
    t2 = threading.Thread(target=classification_thread)
    t3 = threading.Thread(target=output_thread)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

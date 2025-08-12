# train_model.py
import joblib
from bci_utils import load_data, preprocess, extract_stat_features
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
import numpy as np

ALL_SUBJECTS = list(range(1, 110))
TEST_SUBJECT = 77
TRAIN_SUBJECTS = [s for s in ALL_SUBJECTS if s != TEST_SUBJECT]
RUNS = [4, 8, 12]

def train_and_save():
    X_all, y_all = [], []
    for subj in TRAIN_SUBJECTS:
        try:
            raw = load_data(subj, RUNS, sfreq=250)
            epochs = preprocess(raw)
            X = epochs.get_data()
            y = epochs.events[:, -1]
            X_all.append(X)
            y_all.append(y)
            print(f"Loaded Subject {subj}")
        except Exception as e:
            print(f"Failed Subject {subj}: {e}")

    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    csp = CSP(n_components=2)
    X_csp = csp.fit_transform(X_all, y_all)
    X_feat = extract_stat_features(X_csp)

    lda = LinearDiscriminantAnalysis()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(lda, X_feat, y_all, cv=cv)
    print(f"Cross-val accuracy: {np.mean(scores)*100:.2f}%")

    lda.fit(X_feat, y_all)

    joblib.dump(csp, "csp_model.pkl")
    joblib.dump(lda, "lda_model.pkl")
    print("✅ Models saved.")

train_and_save()

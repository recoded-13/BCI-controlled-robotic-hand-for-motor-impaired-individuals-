import numpy as np
import joblib
import RPi.GPIO as GPIO
import time

# === GPIO Setup ===
servoPIN1 = 17  # First servo (e.g., index finger)
servoPIN2 = 18  # Second servo (e.g., middle finger)

GPIO.setmode(GPIO.BCM)
GPIO.setup(servoPIN1, GPIO.OUT)
GPIO.setup(servoPIN2, GPIO.OUT)

p1 = GPIO.PWM(servoPIN1, 50)  # 50 Hz PWM
p2 = GPIO.PWM(servoPIN2, 50)

p1.start(2.5)  # Initialize to 0° (open hand)
p2.start(2.5)

def move_servos(angle):
    duty_cycle = 2.5 + (angle / 18)  # Convert angle to duty cycle
    p1.ChangeDutyCycle(duty_cycle)
    p2.ChangeDutyCycle(duty_cycle)
    time.sleep(0.5)  # Wait for servos to move
    p1.ChangeDutyCycle(0)  # Optional: stop PWM signal to prevent jitter
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

def classify_trial(trial, true_label):
    trial = trial[np.newaxis, :]  # Shape (1, channels, timepoints)
    trial_csp = csp.transform(trial)
    features = extract_stat_features(trial_csp)
    prediction = lda.predict(features)[0]

    action = "Open Hand" if prediction == 2 else "Close Hand"
    print(f"Prediction: {action} | Ground Truth: {true_label} | {'✅' if prediction == true_label else '❌'}")

    # Move servos according to prediction
    if prediction == 2:
        move_servos(0)    # Open hand position
    else:
        move_servos(180)  # Close hand position

if __name__ == "__main__":
    try:
        # Replace with your chosen trial files
        trial_file = "trial_1.npy"
        label_file = "trial_1_label.npy"

        trial = np.load(trial_file)
        true_label = np.load(label_file).item()  # scalar

        classify_trial(trial, true_label)

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        # Cleanup GPIO on exit
        p1.stop()
        p2.stop()
        GPIO.cleanup()

# BCI-controlled-robotic-hand-for-motor-impaired-individuals | Final Year Project 

This repository contains the code, data, and models for our undergraduate final year project: a Brain-Computer Interface (BCI)-controlled robotic hand using EEG motor imagery signals. The system performs EEG acquisition, preprocessing, feature extraction (CSP), classification (LDA), and real-time robotic actuation.
The project was implemented using Python for machine learning and real-time processing, and MATLAB for signal processing and simulation.

## Background
Motor imagery-based BCIs translate brain activity into commands without requiring physical movement. This makes them valuable in assistive robotics and rehabilitation.
In our case, EEG signals corresponding to left and right hand motor imagery were recorded, classified in real time, and used to control a low-cost robotic hand via a Raspberry Pi.

The CSP+LDA pipeline was chosen for its low latency and robust performance in two-class motor imagery tasks, making it suitable for real-time applications compared to computationally expensive deep learning models.

## Project Overview
### Phase 1 — Model Training with Public Dataset

We trained a binary classification model using the PhysioNet motor imagery EEG dataset.
- Subject 85's data was used for training the real-time model
- Achieved 80% real-time accuracy controlling a robotic hand via Raspberry Pi
- Achieved 62% overall accuracy across the dataset
- Controlled a 3D-printed robotic hand using classification outputs

### Phase 2 — Data Collection & ERP Analysis

We collected EEG data using gold cup electrodes and the OpenBCI Ganglion board.
- Conducted left/right hand motor imagery tasks
- Stored raw EEG in CSV format via Raspberry Pi
- Performed ERP analysis in MATLAB EEGLAB to identify positive and negative deflections for left/right tasks

This repository validates our machine learning model using the [PhysioNet EEG Motor Imagery dataset] (https://physionet.org/content/eegmmidb/1.0.0/) due to noise in real-time acquisition, but the control system is designed for real-world operation.
## Methods
### Signal Acquisition

- Phase 1: Public EEG dataset (PhysioNet, 64 channels, motor imagery tasks)
- Phase 2:
   - EEG: OpenBCI Ganglion board (gold cup electrodes)
   - Sampling rate: 250 Hz
   - Electrodes placed at C3, Cz, and C4
   - Tasks designed and presented using PsychoPy

### Preprocessing (MATLAB)

- Bandpass filtering (8–30 Hz) to extract mu and beta rhythms
- Segmentation based on stimulus triggers
- Artifact removal (manual/automatic)
- ERP analysis in EEGLAB to detect motor-related potentials
- Exported segmented trials as .npy for Python classification

### Feature Extraction (Python)

- Common Spatial Patterns (CSP) for spatial filtering
- Variance of CSP-filtered signals as feature vectors

### Classification

- Linear Discriminant Analysis (LDA) for binary classification
- Real-time inference tested with Raspberry Pi controlling robotic hand

### Results
- Real-time demo (Subject 85): 80% accuracy in robotic control
- Overall dataset accuracy: 62%
- ERP Analysis: Positive and negative peaks identifiable for left/right hand motor imagery tasks
- CSP+LDA pipeline provided low-latency classification, enabling smooth control in demonstration scenarios

### Limitations
- Real-time EEG acquisition produced noisy signals, limiting accuracy
- Final live demo used pre-trained model data.
- System currently supports only binary motor imagery classification
- 3D-printed robotic hand was low-cost but mechanically limited in grip strength

## How to Interact With This Repo
### Repository Structure 
- `bci_utils.py`	Common helper functions for EEG data loading, preprocessing, and feature extraction. Imported by other scripts.
- `train_model.py`	Training script: Trains CSP + LDA models on all subjects except test subject. Saves pretrained models (csp_model.pkl, lda_model.pkl). Run once before demo day.
- `demo_hand_control.py`	Demo script: Loads pretrained models and runs a real-time simulation on test subject data. Uses selected EEG trials to simulate robotic hand control. Run during open house/demo.

If you are interested in replicating or extending this project, you can follow these steps:

I. Preprocess EEG Data (Python)
   - Run `bci_utils.py`

II. Train Classification Model (Python)
   - `python train_model.py`

III. Run Real-time Prediction
   - `python real_time.py`

IV. Demonstrate Robotic Control
   - `python demo_hand_control.py`

## Technologies Used
- EEG: OpenBCI (Ganglion board)
- MATLAB: EEGLAB for ERP analysis
- Python: MNE, scikit-learn
- PsychoPy for motor imagery task design
- Raspberry Pi for robotic control
- 3D printing for robotic hand fabrication

## Impact
This system provides a cost-effective assistive technology that integrates real-time brain signal processing with robotic actuation, aimed at improving mobility and independence for motor-impaired individuals.

## Team
- Reehab Ahmed
- Zoobiya Aalam 
- Aliza Shabraiz

# MeterWhisperer

MeterWhisperer is a prototype machine-learning system that detects water leaks by
listening to the vibration signature of a water meter. It ingests accelerometer
readings, extracts time-domain and frequency-domain features, and classifies the
activity using a Random Forest model.

The project ships with a trained model and a small Flask API that exposes it for
real-time predictions.

---

## What it does

Given a stream of 3-axis accelerometer readings (as they would come from a sensor
mounted on a water meter), the system predicts whether the current activity is:

| Class | Meaning |
|-------|---------|
| `normal` | No leak detected, normal flow |
| `tap_water_leak` | Leak originating from a tap/faucet |
| `washing_machine_leak` | Leak originating from a washing machine |
| `toilet_leak` | Persistent toilet refill / tank leak pattern |
| `pipe_fitting` | Irregular pipe or fitting vibration anomaly |

For detected leaks it also estimates a **severity** (`LOW`, `MODERATE`, `HIGH`)
based on the average vibration RMS.

> **Note:** This is a prototype trained on synthetic data. Real-world deployment
> requires validation with labelled sensor recordings.

---

## Project structure

```
MeterWhisperer/
├── app.py                       # Flask API server (entry point)
├── requirements.txt             # Python dependencies
├── data/
│   └── raw/                     # Raw CSV sensor recordings, grouped by class
│       ├── normal/
│       ├── tap_water_leak/
│       ├── washing_machine_leak/
│       ├── toilet_leak/
│       └── pipe_fitting/
├── models/
│   └── meter_whisperer_model.joblib   # Trained model + feature metadata
├── reports/
│   ├── classification_report.txt      # Precision/recall/F1 per class
│   └── confusion_matrix.csv           # Confusion matrix
├── static/                      # Web UI served by Flask
│   ├── index.html               # Single-page interface
│   ├── css/style.css            # Styles
│   └── js/app.js                # Frontend logic (demo + API calls)
└── src/
    ├── features.py              # Feature extraction from signals
    ├── train_model.py           # Trains the Random Forest model
    └── predict.py               # (placeholder)
```

---

## Requirements

- Python 3.8+ (developed and tested on Python 3.13)

Python packages:

| Package | Purpose |
|---------|---------|
| `flask` | Web framework for the API |
| `flask-cors` | Cross-origin request support |
| `joblib` | Save/load the trained model |
| `numpy` | Numerical operations |
| `pandas` | Tabular data handling |
| `scipy` | Frequency-domain feature extraction |
| `scikit-learn` | Model training and evaluation |

Install them all with:

```bash
pip install -r requirements.txt
```

---

## Setup

```bash
cd /path/to/MeterWhisperer
pip install -r requirements.txt
```

Make sure the trained model exists at `models/meter_whisperer_model.joblib`
(it is included in the repository). The app will refuse to start if it is missing.

---

## Usage

### 1. Start the API server

```bash
python3 app.py
```

The server starts at:

```
http://127.0.0.1:5000
```

On startup it loads the model and prints the available classes:

```
Meter Whisperer model loaded successfully.
Available classes: ['normal', 'pipe_fitting', 'tap_water_leak', 'toilet_leak', 'washing_machine_leak']
```

### 2. Open the web UI

Open `http://127.0.0.1:5000/` in a browser to use the interactive interface:

- **Signal Lab** — switch between raw / processed / frequency / interpretation views of the signal.
- **Interactive AI demonstration** — pick an activity pattern (Normal, Tap, Washing machine, Toilet, Pipe / fitting) and see a real model prediction.
- **Real sensor recordings** — load any CSV from `data/raw/` and run a real prediction.

The UI is served by Flask from the `static/` folder (`static/index.html`,
`static/css/style.css`, `static/js/app.js`) and talks to the same backend on the
same origin, so no CORS configuration is needed.

### 3. Health check

```bash
curl http://127.0.0.1:5000/api/health
```

Response:

```json
{
  "classes": ["normal", "pipe_fitting", "tap_water_leak", "toilet_leak", "washing_machine_leak"],
  "model": "Meter Whisperer",
  "status": "online"
}
```

### 4. Predict a leak

Send a `POST` request to `/api/predict` with the raw sensor readings.

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_s":      [0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
    "acc_x_m_s2":  [0.01, 0.02, 0.01, 0.00, -0.01, 0.02],
    "acc_y_m_s2":  [0.00, 0.01, 0.02, 0.01, 0.00, 0.01],
    "acc_z_m_s2":  [0.00, 0.00, 0.01, 0.01, 0.00, 0.00]
  }'
```

#### Request format

The JSON body must contain four parallel one-dimensional arrays of equal length:

| Field | Description |
|-------|-------------|
| `time_s` | Timestamps in seconds (must be strictly increasing) |
| `acc_x_m_s2` | Accelerometer X-axis readings (m/s²) |
| `acc_y_m_s2` | Accelerometer Y-axis readings (m/s²) |
| `acc_z_m_s2` | Accelerometer Z-axis readings (m/s²) |

Validation rules applied by the server:

- At least 200 readings (~2 seconds) are required.
- Timestamps must be strictly increasing.
- The inferred sampling rate must be between 1 Hz and 1000 Hz.


#### Response format

```json
{
  "success": true,
  "predicted_class": "washing_machine_leak",
  "source": "Washing Machine Leak",
  "confidence_percent": 93.4,
  "severity": "MODERATE",
  "average_vibration_rms": 0.031234,
  "note": "Prototype prediction. Real-world performance requires validation with labelled sensor recordings."
}
```

| Field | Description |
|-------|-------------|
| `predicted_class` | Raw class label (`normal`, `tap_water_leak`, `washing_machine_leak`, `toilet_leak`, `pipe_fitting`) |
| `source` | Human-readable class name |
| `confidence_percent` | Model confidence (0–100) |
| `severity` | `NORMAL`, `LOW`, `MODERATE`, or `HIGH` |
| `average_vibration_rms` | Average RMS of the input signal |

#### Severity thresholds

For a leak class, severity is estimated from the average vibration RMS:

| RMS range | Severity |
|-----------|----------|
| `< 0.02` | `LOW` |
| `0.02 – 0.06` | `MODERATE` |
| `>= 0.06` | `HIGH` |

These are prototype thresholds and should be calibrated with real data.

---

## Retraining the model

The training script rebuilds the dataset from `data/raw/` and trains a
Random Forest classifier.

```bash
cd src
python3 train_model.py
```

What it does:

1. Reads every CSV under `data/raw/<class>/`.
2. Splits each recording into 2-second windows (1-second step).
3. Extracts time-domain and frequency-domain features per window.
4. Splits data at the **recording level** (no window from the same CSV leaks
   between train and test sets).
5. Trains a `RandomForestClassifier` (500 trees, balanced class weights).
6. Saves:
   - the model → `models/meter_whisperer_model.joblib`
   - the classification report → `reports/classification_report.txt`
   - the confusion matrix → `reports/confusion_matrix.csv`

> Run the training script from inside `src/` (it imports `features` directly and
> resolves paths relative to the project root).


---

## Input data format

Training CSVs are stored under `data/raw/<class_name>/`. Each file is a
comma-separated table with the following columns:

```csv
time_s,acc_x_m_s2,acc_y_m_s2,acc_z_m_s2,acc_abs_m_s2,acc_accuracy
0.0,0.0010958688089016174,0.016496127937235854,0.00567740272045111,0.017480162099310245,3
0.01,0.0037965364097519117,-0.009725087175092953,0.00534884044885872,0.01173034967173087,3
```

The feature extractor uses `time_s`, `acc_x_m_s2`, `acc_y_m_s2`, and
`acc_z_m_s2`. The accelerometer magnitude is computed as:

```
magnitude = sqrt(acc_x² + acc_y² + acc_z²)
```

The three axes are combined into a single magnitude signal before feature
extraction.

### Features used

Each 2-second window produces the following features:

- **Time-domain:** RMS, standard deviation, peak absolute value, peak-to-peak,
  mean absolute value, sampling rate.
- **Frequency-domain:** dominant frequency, spectral centroid, and energy in the
  0–5, 5–15, 15–30, 30–60, and 60–100 Hz bands.

---

## Current model performance

The included model was evaluated on a held-out, recording-level test split:

- **Test accuracy:** 99.83% (synthetic data)

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| normal | 1.00 | 1.00 | 1.00 | 145 |
| tap_water_leak | 1.00 | 1.00 | 1.00 | 87 |
| washing_machine_leak | 1.00 | 1.00 | 1.00 | 87 |
| toilet_leak | 0.99 | 1.00 | 1.00 | 116 |
| pipe_fitting | 1.00 | 0.99 | 1.00 | 145 |

These results reflect the synthetic training data and are **not** a substitute
for real-world validation.

---

## Limitations

- Trained on synthetic accelerometer data, not real sensor recordings.
- Severity thresholds are heuristic and uncalibrated.
- Synthetic toilet and pipe/fitting examples are included for prototyping; real-world
  validation with labelled recordings is still required.


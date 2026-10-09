# MeterWhisperer — Test Plan & Test Cases

This document lists the test cases for verifying the MeterWhisperer project
(Flask API, feature extraction, model, and training pipeline).

## Scope

- **Unit tests** — `src/features.py` helper functions.
- **API integration tests** — `app.py` endpoints (`/api/health`, `/api/predict`).
- **Model tests** — trained model artifact `models/meter_whisperer_model.joblib`.
- **Pipeline tests** — `src/train_model.py` end-to-end run.

## Test environment

- OS: macOS (darwin)
- Python: 3.13
- Working directory: `MeterWhisperer/` (project root)
- Dependencies installed via `pip install -r requirements.txt`

## How to run the tests

```bash
# Start the server (leave running in one terminal)
cd /path/to/MeterWhisperer
python3 app.py

# In another terminal, run each API test with curl (see each case)
```

Unit tests can be run interactively:

```bash
python3 -c "from src.features import extract_features_from_signal; print(extract_features_from_signal([1,-1,1,-1], 100))"
```

---

## Test Case Index

| ID | Area | Title |
|----|------|-------|
| T-ENV-01 | Environment | All dependencies are installed and importable |
| T-FEAT-01 | Unit | `calculate_sampling_rate` returns correct rate |
| T-FEAT-02 | Unit | `calculate_sampling_rate` rejects constant timestamps |
| T-FEAT-03 | Unit | `calculate_sampling_rate` rejects decreasing timestamps |
| T-FEAT-04 | Unit | `extract_features_from_signal` returns all 13 features |
| T-FEAT-05 | Unit | `extract_features_from_signal` computes correct RMS |
| T-FEAT-06 | Unit | `extract_features_from_signal` removes DC offset |
| T-FEAT-07 | Unit | `extract_features_from_csv` reads a valid CSV |
| T-FEAT-08 | Unit | `extract_features_from_csv` rejects missing columns |
| T-FEAT-09 | Unit | `extract_features_from_csv` rejects too-few rows |
| T-FEAT-10 | Unit | `build_dataset` builds a non-empty labeled dataset |
| T-FEAT-11 | Unit | `extract_features_from_csv` on a real CSV (with results) |
| T-FEAT-12 | Unit | `build_dataset` over all `data/raw` CSVs (with results) |
| T-FEAT-13 | Unit | End-to-end classification of every `data/raw` recording |
| T-API-01 | API | Health check returns 200 with expected fields |
| T-API-02 | API | Predict rejects a non-JSON-object body |
| T-API-03 | API | Predict rejects missing fields |
| T-API-04 | API | Predict rejects a 2D (non-1D) array |
| T-API-05 | API | Predict rejects unequal array lengths |
| T-API-06 | API | Predict rejects fewer than 200 samples |
| T-API-07 | API | Predict rejects non-finite numbers |
| T-API-08 | API | Predict rejects non-increasing timestamps |
| T-API-09 | API | Predict rejects invalid sampling rate |
| T-API-10 | API | Predict returns a valid prediction on good input |
| T-MODEL-01 | Model | Model artifact exists and loads |
| T-MODEL-02 | Model | Model bundle contains required keys |
| T-MODEL-03 | Model | Model exposes the expected classes |
| T-TRAIN-01 | Pipeline | Training script runs to completion |
| T-TRAIN-02 | Pipeline | Training produces report artifacts |

---

## Environment Tests

### T-ENV-01 — Dependencies are installed

- **Objective:** Verify every required package is importable in the current Python.
- **Preconditions:** Project root as working directory.
- **Steps:**
  ```bash
  python3 -c "import flask, flask_cors, joblib, numpy, pandas, scipy, sklearn; print('OK')"
  ```
- **Expected result:** Prints `OK` with no `ModuleNotFoundError`.

---

## Unit Tests — `src/features.py`

### T-FEAT-01 — `calculate_sampling_rate` returns the correct rate

- **Objective:** Confirm the sampling rate is the inverse of the median time step.
- **Steps:**
  ```bash
  python3 -c "from src.features import calculate_sampling_rate; print(calculate_sampling_rate([0.0, 0.01, 0.02, 0.03]))"
  ```
- **Expected result:** Prints `100.0` (median Δt = 0.01 s → 100 Hz).

### T-FEAT-02 — `calculate_sampling_rate` rejects constant timestamps

- **Objective:** Constant timestamps yield no positive time step.
- **Steps:**
  ```bash
  python3 -c "from src.features import calculate_sampling_rate; calculate_sampling_rate([1.0, 1.0, 1.0])"
  ```
- **Expected result:** Raises `ValueError` ("Could not calculate sampling rate.").

### T-FEAT-03 — `calculate_sampling_rate` rejects decreasing timestamps

- **Objective:** A decreasing sequence yields no positive time step.
- **Steps:**
  ```bash
  python3 -c "from src.features import calculate_sampling_rate; calculate_sampling_rate([3.0, 2.0, 1.0])"
  ```
- **Expected result:** Raises `ValueError` ("Could not calculate sampling rate.").

### T-FEAT-04 — `extract_features_from_signal` returns all 13 features

- **Objective:** Verify the feature dictionary has every column the model expects.
- **Steps:**
  ```bash
  python3 -c "from src.features import extract_features_from_signal, FEATURE_COLUMNS; f = extract_features_from_signal(list(range(256)), 100); print(sorted(f.keys()) == sorted(FEATURE_COLUMNS), sorted(f.keys()))"
  ```
- **Expected result:** Prints `True` followed by the 13 feature names.

### T-FEAT-05 — `extract_features_from_signal` computes correct RMS

- **Objective:** RMS of a zero-mean alternating signal is the signal amplitude.
- **Steps:**
  ```bash
  python3 -c "from src.features import extract_features_from_signal; print(extract_features_from_signal([1,-1,1,-1], 100)['rms'])"
  ```
- **Expected result:** Prints `1.0`.

### T-FEAT-06 — `extract_features_from_signal` removes DC offset

- **Objective:** A constant (DC-only) signal has zero energy after centering.
- **Steps:**
  ```bash
  python3 -c "from src.features import extract_features_from_signal; print(extract_features_from_signal([5.0]*100, 100)['rms'])"
  ```
- **Expected result:** Prints `0.0`.

### T-FEAT-07 — `extract_features_from_csv` reads a valid CSV

- **Objective:** A valid CSV under `data/raw/` produces a feature DataFrame.
- **Steps:**
  ```bash
  python3 -c "from src.features import extract_features_from_csv; import pandas as pd; df = extract_features_from_csv('data/raw/normal/normal_01_SYNTHETIC.csv'); print(type(df).__name__, len(df) > 0)"
  ```
- **Expected result:** Prints `DataFrame True`.

### T-FEAT-08 — `extract_features_from_csv` rejects missing columns

- **Objective:** A CSV without the required columns is rejected.
- **Steps:** Create a temp CSV with only `time_s` and call the function.
  ```bash
  printf 'time_s\n0.0\n0.01\n' > /tmp/bad.csv
  python3 -c "from src.features import extract_features_from_csv; extract_features_from_csv('/tmp/bad.csv')"
  ```
- **Expected result:** Raises `ValueError` naming the missing columns.

### T-FEAT-09 — `extract_features_from_csv` rejects too-few rows

- **Objective:** A CSV with fewer than 20 rows is rejected.
- **Steps:**
  ```bash
  python3 -c "from src.features import extract_features_from_csv; import pandas as pd; pd.DataFrame({'time_s':[0.0], 'acc_x_m_s2':[0.0], 'acc_y_m_s2':[0.0], 'acc_z_m_s2':[0.0]}).to_csv('/tmp/tiny.csv', index=False); extract_features_from_csv('/tmp/tiny.csv')"
  ```
- **Expected result:** Raises `ValueError` ("does not contain enough data.").

### T-FEAT-10 — `build_dataset` builds a non-empty labeled dataset

- **Objective:** The full dataset builds with a `label` column and 3 populated classes.
- **Steps:**
  ```bash
  python3 -c "from src.features import build_dataset; d = build_dataset('data/raw'); print(d['label'].value_counts().to_dict())"
  ```
- **Expected result:** Prints counts for `normal`, `tap_water_leak`, and
  `washing_machine_leak` (all > 0). If no CSVs exist, raises `RuntimeError`.


---

## API Integration Tests — `app.py`

> Precondition for all API tests: the server is running at
> `http://127.0.0.1:5000` (start with `python3 app.py`).

### T-API-01 — Health check returns 200

- **Steps:**
  ```bash
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/api/health
  curl -s http://127.0.0.1:5000/api/health
  ```
- **Expected result:** First command prints `200`. Second returns JSON with keys
  `status` (`"online"`), `model` (`"Meter Whisperer"`), and a `classes` array.

### T-API-02 — Predict rejects a non-object body

- **Steps:**
  ```bash
  curl -s -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '[]'
  ```
- **Expected result:** HTTP 400 with
  `{"error": "Send a JSON object containing sensor readings."}`.

### T-API-03 — Predict rejects missing fields

- **Steps:**
  ```bash
  curl -s -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"time_s": [0.0, 0.01]}'
  ```
- **Expected result:** HTTP 400 with an error listing the missing fields
  (`acc_x_m_s2`, `acc_y_m_s2`, `acc_z_m_s2`).

### T-API-04 — Predict rejects a 2D array

- **Steps:**
  ```bash
  curl -s -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"time_s": [[0.0],[0.01]], "acc_x_m_s2": [0.0,0.01], "acc_y_m_s2": [0.0,0.01], "acc_z_m_s2": [0.0,0.01]}'
  ```
- **Expected result:** HTTP 400 with
  `{"error": "Each reading field must be a one-dimensional array."}`.

### T-API-05 — Predict rejects unequal array lengths

- **Steps:**
  ```bash
  curl -s -X POST http://127.0.0.1:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"time_s": [0.0,0.01,0.02], "acc_x_m_s2": [0.0,0.01], "acc_y_m_s2": [0.0,0.01,0.02], "acc_z_m_s2": [0.0,0.01,0.02]}'
  ```
- **Expected result:** HTTP 400 with
  `{"error": "All reading arrays must have equal lengths."}`.

### T-API-06 — Predict rejects fewer than 200 samples

- **Steps:** Send 4 parallel arrays each with 100 values (all finite, increasing
  timestamps at 0.01 s).
- **Expected result:** HTTP 400 with
  `{"error": "Send at least 2 seconds of readings."}`.

### T-API-07 — Predict rejects non-finite numbers

- **Steps:** Send 200-sample arrays where one accelerometer value is `"NaN"`
  (or `Infinity`).
- **Expected result:** HTTP 400 with
  `{"error": "Readings must contain valid finite numbers."}`.

### T-API-08 — Predict rejects non-increasing timestamps

- **Steps:** Send 200-sample arrays where `time_s` contains a repeated or
  decreasing value (e.g. `[..., 1.0, 1.0, ...]`).
- **Expected result:** HTTP 400 with
  `{"error": "Timestamps must be strictly increasing."}`.

### T-API-09 — Predict rejects invalid sampling rate

- **Steps:** Send 200-sample arrays with timestamps spaced so the median Δt is
  outside 0.001–1.0 s (e.g. Δt = 2.0 s → 0.5 Hz).
- **Expected result:** HTTP 400 with `{"error": "Invalid sampling rate."}`.

### T-API-10 — Predict returns a valid prediction on good input

- **Objective:** Full happy-path prediction with realistic input.
- **Steps:** Generate 400 samples at 100 Hz with a known leak-like vibration
  (e.g. a sine wave on the axes) and POST:
  ```bash
  python3 - <<'PY'
  import json, numpy as np
  t = np.arange(400) * 0.01
  x = 0.05 * np.sin(2*np.pi*20*t)
  y = 0.05 * np.sin(2*np.pi*20*t + 1)
  z = 0.05 * np.sin(2*np.pi*20*t + 2)
  print(json.dumps({
      "time_s": t.tolist(),
      "acc_x_m_s2": x.tolist(),
      "acc_y_m_s2": y.tolist(),
      "acc_z_m_s2": z.tolist(),
  }))
  PY
  ```
  Then POST the output JSON to `/api/predict`.
- **Expected result:** HTTP 200 with `"success": true` and a body containing
  `predicted_class`, `source`, `confidence_percent` (0–100), `severity`
  (`NORMAL`/`LOW`/`MODERATE`/`HIGH`), `average_vibration_rms`, and `note`.


---

## Model Tests — `models/meter_whisperer_model.joblib`

### T-MODEL-01 — Model artifact exists and loads

- **Steps:**
  ```bash
  python3 -c "import joblib; b = joblib.load('models/meter_whisperer_model.joblib'); print('loaded', type(b))"
  ```
- **Expected result:** Prints `loaded <class 'dict'>` (or the bundle type) with
  no error. If the file is missing, `app.py` should have raised `FileNotFoundError`.

### T-MODEL-02 — Model bundle contains required keys

- **Steps:**
  ```bash
  python3 -c "import joblib; b = joblib.load('models/meter_whisperer_model.joblib'); print(sorted(b.keys()))"
  ```
- **Expected result:** Contains `classes`, `features`, `model`, and `version`.

### T-MODEL-03 — Model exposes the expected classes

- **Steps:**
  ```bash
  python3 -c "import joblib; b = joblib.load('models/meter_whisperer_model.joblib'); print(list(b['model'].classes_))"
  ```
- **Expected result:** Prints `['normal', 'tap_water_leak', 'washing_machine_leak']`.

---

## Pipeline Tests — `src/train_model.py`

### T-TRAIN-01 — Training script runs to completion

- **Objective:** Verify the full feature-extraction + training pipeline executes.
- **Preconditions:** `data/raw/` contains at least one CSV per class.
- **Steps:**
  ```bash
  cd src
  python3 train_model.py
  ```
- **Expected result:** Prints "Training complete." and exits with status 0. It
  reports a test accuracy and a classification report. (Note: the script must be
  run from inside `src/` because it imports `features` directly.)

### T-TRAIN-02 — Training produces report artifacts

- **Objective:** Verify output files are written.
- **Steps:** After running `train_model.py`, check:
  ```bash
  ls -la models/meter_whisperer_model.joblib reports/classification_report.txt reports/confusion_matrix.csv
  ```
- **Expected result:** All three files exist and are non-empty, and
  `models/meter_whisperer_model.joblib` has a recent modification time.

---

## Feature Extraction & Classification Tests — `features.py` + `data/raw/`

These tests exercise `src/features.py` against the real CSV recordings under
`data/raw/`. The commands and the actual captured results are shown below.

### T-FEAT-11 — Extract features from a single CSV

- **Objective:** Verify `extract_features_from_csv` produces the 13-feature table
  from a real recording.
- **Command:**
  ```bash
  python3 -c "from src.features import extract_features_from_csv; df = extract_features_from_csv('data/raw/normal/normal_01_SYNTHETIC.csv'); print('SHAPE:', df.shape); print('COLUMNS:', df.columns.tolist()); print(df.head(3).round(4).to_string())"
  ```
- **Result (PASS):**
  ```
  SHAPE: (29, 13)
  COLUMNS: ['rms', 'std', 'peak_abs', 'peak_to_peak', 'mean_abs', 'sampling_rate_hz', 'dominant_frequency_hz', 'spectral_centroid_hz', 'band_0_5', 'band_5_15', 'band_15_30', 'band_30_60', 'band_60_100']
        rms     std  peak_abs  peak_to_peak  mean_abs  sampling_rate_hz  dominant_frequency_hz  spectral_centroid_hz  band_0_5  band_5_15  band_15_30  band_30_60  band_60_100
  0  0.0088  0.0088    0.0514        0.0659    0.0066             100.0                    9.0               19.1477    0.1501     0.3690      0.1963      0.2846          0.0
  1  0.0098  0.0098    0.0510        0.0659    0.0075             100.0                   31.0               25.3613    0.1041     0.1741      0.2676      0.4542          0.0
  2  0.0095  0.0095    0.0323        0.0486    0.0075             100.0                   49.0               26.8064    0.0950     0.1566      0.3316      0.4168          0.0
  ```

### T-FEAT-12 — Build the full labeled dataset from `data/raw/`

- **Objective:** Verify `build_dataset` processes all 60 recordings and produces a
  balanced, labeled dataset.
- **Command:**
  ```bash
  python3 -c "from src.features import build_dataset; d = build_dataset('data/raw'); print('SHAPE:', d.shape); print(d['label'].value_counts())"
  ```
- **Result (PASS):** The function logs one line per CSV (20 files per class) and
  warns about the two classes with no data folder (`tap_water`,
  `washing_machine`), then prints:
  ```
  SHAPE: (1740, 15)
  label
  normal                  580
  tap_water_leak          580
  washing_machine_leak    580
  ```
  (15 columns = 13 features + `source_file` + `label`; 1740 = 60 files × 29 windows.)

### T-FEAT-13 — End-to-end classification of every recording

- **Objective:** Feed every CSV through `features.py` and the trained model and
  verify all recordings are correctly classified (majority vote per file).
- **Command:**
  ```bash
  python3 - <<'PY'
  import joblib
  from pathlib import Path
  from collections import Counter
  from src.features import extract_features_from_csv

  bundle = joblib.load('models/meter_whisperer_model.joblib')
  model = bundle['model']
  features = bundle['features']
  classes = list(model.classes_)

  for class_name in classes:
      files = sorted(Path('data/raw', class_name).glob('*.csv'))
      correct = sum(
          Counter(model.predict(extract_features_from_csv(f)[features])).most_common(1)[0][0] == class_name
          for f in files
      )
      print(f'{class_name}: {correct}/{len(files)} files correctly classified')
  PY
  ```
- **Result (PASS):**
  ```
  normal: 20/20 files correctly classified
  tap_water_leak: 20/20 files correctly classified
  washing_machine_leak: 20/20 files correctly classified
  ```

---

## Notes

- API tests T-API-06 through T-API-09 can be executed with short Python helper
  snippets that build the JSON payloads; the core check is the HTTP status and
  the `error` string returned by the server.
- The `src/predict.py` file is currently an empty placeholder and has no test cases.
- Severity thresholds (`LOW`/`MODERATE`/`HIGH`) are heuristic; assert only that
  the returned `severity` is one of the four allowed values.


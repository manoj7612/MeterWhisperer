
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.features import extract_features_from_signal


# =====================================================
# PATHS AND APP CONFIGURATION
# =====================================================

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "meter_whisperer_model.joblib"

app = Flask(__name__)
CORS(app)

DISPLAY_NAMES = {
    "normal": "Normal",
    "tap_water_leak": "Tap Water Leak",
    "washing_machine_leak": "Washing Machine Leak",
    "toilet_leak": "Toilet Leak",
    "pipe_fitting": "Pipe / Fitting Anomaly",
}

# Human-readable explanation / action text surfaced by the UI so that the
# "AI interpretation" card can be filled from real predictions rather than
# hardcoded strings.
CLASS_INFO = {
    "normal": {
        "label": "No abnormal activity",
        "explanation": "The pattern is quiet and consistent with a resting meter.",
        "action": "Continue monitoring",
        "tone": "quiet",
    },
    "tap_water_leak": {
        "label": "Possible faucet / tap",
        "explanation": "Short bursts of vibration look more consistent with intermittent use.",
        "action": "Check for a tap that was left slightly open.",
        "tone": "green",
    },
    "washing_machine_leak": {
        "label": "Possible washing machine",
        "explanation": "A changing signature suggests time-bounded, cyclical activity.",
        "action": "Compare the timing with a scheduled wash cycle.",
        "tone": "blue",
    },
    "toilet_leak": {
        "label": "Possible toilet leak",
        "explanation": "A persistent pulsing pattern suggests water movement similar to a toilet refill leak.",
        "action": "Check the toilet tank flapper, fill valve, and overflow tube.",
        "tone": "red",
    },
    "pipe_fitting": {
        "label": "Possible pipe / fitting issue",
        "explanation": "Irregular bursts can indicate vibration from a loose pipe, fitting, or intermittent pressure change.",
        "action": "Inspect nearby pipe joints and fittings for vibration or seepage.",
        "tone": "orange",
    },
}


# =====================================================
# LOAD TRAINED MODEL
# =====================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found at: {MODEL_PATH}"
    )

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["features"]
classes = list(model.classes_)

print("Meter Whisperer model loaded successfully.")
print("Available classes:", classes)


# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/api/health")
def health():
    return jsonify({
        "status": "online",
        "model": "Meter Whisperer",
        "classes": classes,
    })


# =====================================================
# FRONTEND / STATIC UI
# =====================================================

@app.get("/")
def index():
    return app.send_static_file("index.html")


# =====================================================
# META / CLASSES
# =====================================================

@app.get("/api/classes")
def api_classes():
    return jsonify({
        "classes": classes,
        "display_names": DISPLAY_NAMES,
        "info": {
            class_name: CLASS_INFO.get(class_name, {})
            for class_name in classes
        },
    })


# =====================================================
# RECORDINGS (real data from data/raw)
# =====================================================

@app.get("/api/recordings")
def api_recordings():
    data_root = ROOT / "data" / "raw"
    recordings = {}

    if data_root.exists():
        for class_dir in sorted(data_root.iterdir()):
            if not class_dir.is_dir():
                continue

            files = sorted(
                csv_file.name
                for csv_file in class_dir.glob("*.csv")
            )

            if files:
                recordings[class_dir.name] = files

    return jsonify({"recordings": recordings})


@app.get("/api/recording/<class_name>/<filename>")
def api_recording(class_name, filename):
    csv_path = ROOT / "data" / "raw" / class_name / filename

    if not csv_path.exists():
        return jsonify({"error": "Recording not found."}), 404

    try:
        df = pd.read_csv(csv_path)
    except Exception as error:  # pragma: no cover - defensive
        return jsonify({"error": str(error)}), 400

    required = ["time_s", "acc_x_m_s2", "acc_y_m_s2", "acc_z_m_s2"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        return jsonify({
            "error": f"Recording is missing columns: {missing}"
        }), 400

    return jsonify({
        "class": class_name,
        "filename": filename,
        "time_s": df["time_s"].tolist(),
        "acc_x_m_s2": df["acc_x_m_s2"].tolist(),
        "acc_y_m_s2": df["acc_y_m_s2"].tolist(),
        "acc_z_m_s2": df["acc_z_m_s2"].tolist(),
    })


# =====================================================
# SEVERITY ESTIMATION
# Prototype thresholds; calibrate with real data.
# =====================================================

def calculate_severity(predicted_class, average_rms):

    if predicted_class == "normal":
        return "NORMAL"

    if average_rms < 0.02:
        return "LOW"

    if average_rms < 0.06:
        return "MODERATE"

    return "HIGH"


# =====================================================
# PREDICTION API
# =====================================================

@app.post("/api/predict")
def predict():

    try:
        payload = request.get_json(silent=True)

        if not isinstance(payload, dict):
            return jsonify({
                "error": "Send a JSON object containing sensor readings."
            }), 400

        required_fields = [
            "time_s",
            "acc_x_m_s2",
            "acc_y_m_s2",
            "acc_z_m_s2",
        ]

        missing = [
            key for key in required_fields
            if key not in payload
        ]

        if missing:
            return jsonify({
                "error": f"Missing fields: {missing}"
            }), 400

        time_values = np.asarray(
            payload["time_s"], dtype=float
        )
        x = np.asarray(
            payload["acc_x_m_s2"], dtype=float
        )
        y = np.asarray(
            payload["acc_y_m_s2"], dtype=float
        )
        z = np.asarray(
            payload["acc_z_m_s2"], dtype=float
        )

        arrays = [time_values, x, y, z]

        if any(a.ndim != 1 for a in arrays):
            return jsonify({
                "error": "Each reading field must be a one-dimensional array."
            }), 400

        if len({len(a) for a in arrays}) != 1:
            return jsonify({
                "error": "All reading arrays must have equal lengths."
            }), 400

        if len(x) < 200:
            return jsonify({
                "error": "Send at least 2 seconds of readings."
            }), 400

        if not all(np.isfinite(a).all() for a in arrays):
            return jsonify({
                "error": "Readings must contain valid finite numbers."
            }), 400

        time_differences = np.diff(time_values)

        if np.any(time_differences <= 0):
            return jsonify({
                "error": "Timestamps must be strictly increasing."
            }), 400

        sampling_rate = 1.0 / float(
            np.median(time_differences)
        )

        if not 1 <= sampling_rate <= 1000:
            return jsonify({
                "error": "Invalid sampling rate."
            }), 400

        # Combine accelerometer axes into magnitude.
        magnitude = np.sqrt(x**2 + y**2 + z**2)

        window_size = int(2.0 * sampling_rate)
        step_size = max(1, int(1.0 * sampling_rate))

        if len(magnitude) < window_size:
            return jsonify({
                "error": "Not enough samples for a 2-second window."
            }), 400

        feature_rows = []

        for start in range(
            0,
            len(magnitude) - window_size + 1,
            step_size
        ):
            window = magnitude[start:start + window_size]

            features = extract_features_from_signal(
                window,
                sampling_rate
            )

            feature_rows.append(features)

        feature_df = pd.DataFrame(feature_rows)

        # Match the feature order used during training.
        feature_df = feature_df[feature_columns]

        probabilities = model.predict_proba(feature_df)

        average_probabilities = probabilities.mean(axis=0)

        best_index = int(np.argmax(average_probabilities))
        predicted_class = classes[best_index]
        confidence = float(average_probabilities[best_index])

        average_rms = float(feature_df["rms"].mean())

        severity = calculate_severity(
            predicted_class,
            average_rms
        )

        info = CLASS_INFO.get(predicted_class, {})

        return jsonify({
            "success": True,
            "predicted_class": predicted_class,
            "source": DISPLAY_NAMES.get(
                predicted_class,
                predicted_class
            ),
            "confidence_percent": round(
                confidence * 100, 2
            ),
            "severity": severity,
            "average_vibration_rms": round(
                average_rms, 6
            ),
            "explanation": info.get("explanation", ""),
            "action": info.get("action", ""),
            "tone": info.get("tone", "quiet"),
            "note": (
                "Prototype prediction. "
                "Real-world performance requires validation "
                "with labelled sensor recordings."
            ),
        })

    except (ValueError, TypeError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction request failed")
        return jsonify({
            "error": "Prediction failed. Check the backend terminal."
        }), 500


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
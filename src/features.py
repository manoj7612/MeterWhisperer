from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# SETTINGS
# ============================================================

WINDOW_SECONDS = 2.0
STEP_SECONDS = 1.0


# ============================================================
# FEATURES USED BY THE MODEL
# ============================================================

FEATURE_COLUMNS = [
    "rms",
    "std",
    "peak_abs",
    "peak_to_peak",
    "mean_abs",
    "sampling_rate_hz",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "band_0_5",
    "band_5_15",
    "band_15_30",
    "band_30_60",
    "band_60_100",
]


# ============================================================
# FIVE CLASSES
# ============================================================

CLASS_NAMES = [
    "normal",
    "tap_water_leak",
    "washing_machine_leak",
    "toilet_leak",
    "pipe_fitting",
]


# ============================================================
# CALCULATE SAMPLING RATE
# ============================================================

def calculate_sampling_rate(time_values):

    time_values = np.asarray(time_values, dtype=float)

    differences = np.diff(time_values)

    differences = differences[differences > 0]

    if len(differences) == 0:
        raise ValueError("Could not calculate sampling rate.")

    median_dt = np.median(differences)

    if median_dt <= 0:
        raise ValueError("Invalid time values.")

    return 1.0 / median_dt


# ============================================================
# FREQUENCY FEATURES
# ============================================================

def calculate_frequency_features(signal, sampling_rate):

    frequencies, power = welch(
        signal,
        fs=sampling_rate,
        nperseg=min(256, len(signal))
    )

    total_power = np.sum(power)

    if total_power <= 0:
        total_power = 1e-12

    # Frequency with maximum power
    dominant_frequency = frequencies[np.argmax(power)]

    # Spectral centroid
    spectral_centroid = (
        np.sum(frequencies * power) / total_power
    )

    # Energy inside frequency bands
    def band_energy(low, high):

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        if not np.any(mask):
            return 0.0

        return np.sum(power[mask]) / total_power

    return {
        "dominant_frequency_hz": dominant_frequency,
        "spectral_centroid_hz": spectral_centroid,
        "band_0_5": band_energy(0, 5),
        "band_5_15": band_energy(5, 15),
        "band_15_30": band_energy(15, 30),
        "band_30_60": band_energy(30, 60),
        "band_60_100": band_energy(60, 100),
    }


# ============================================================
# EXTRACT FEATURES FROM ONE SIGNAL WINDOW
# ============================================================

def extract_features_from_signal(signal, sampling_rate):

    signal = np.asarray(signal, dtype=float)

    # Remove DC component
    signal = signal - np.mean(signal)

    # Time-domain features
    rms = np.sqrt(np.mean(signal ** 2))

    std = np.std(signal)

    peak_abs = np.max(np.abs(signal))

    peak_to_peak = np.ptp(signal)

    mean_abs = np.mean(np.abs(signal))

    # Frequency-domain features
    frequency_features = calculate_frequency_features(
        signal,
        sampling_rate
    )

    features = {
        "rms": rms,
        "std": std,
        "peak_abs": peak_abs,
        "peak_to_peak": peak_to_peak,
        "mean_abs": mean_abs,
        "sampling_rate_hz": sampling_rate,
        **frequency_features,
    }

    return features


# ============================================================
# EXTRACT FEATURES FROM ONE CSV FILE
# ============================================================

def extract_features_from_csv(csv_path):

    csv_path = Path(csv_path)

    df = pd.read_csv(csv_path)

    # Required columns
    required_columns = [
        "time_s",
        "acc_x_m_s2",
        "acc_y_m_s2",
        "acc_z_m_s2",
    ]

    # Check missing columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            f"{csv_path.name} is missing columns: "
            f"{missing_columns}"
        )

    # Remove incomplete rows
    df = df.dropna(
        subset=required_columns
    )

    if len(df) < 20:

        raise ValueError(
            f"{csv_path.name} does not contain enough data."
        )

    # Calculate sampling rate
    sampling_rate = calculate_sampling_rate(
        df["time_s"].values
    )

    # Accelerometer axes
    x = df["acc_x_m_s2"].values
    y = df["acc_y_m_s2"].values
    z = df["acc_z_m_s2"].values

    # Calculate total acceleration magnitude
    magnitude = np.sqrt(
        x ** 2 +
        y ** 2 +
        z ** 2
    )

    # Convert seconds into samples
    window_size = int(
        WINDOW_SECONDS * sampling_rate
    )

    step_size = int(
        STEP_SECONDS * sampling_rate
    )

    if window_size <= 0 or step_size <= 0:

        raise ValueError(
            "Invalid window configuration."
        )

    results = []

    # Slide through recording
    for start in range(
        0,
        len(magnitude) - window_size + 1,
        step_size
    ):

        end = start + window_size

        window = magnitude[start:end]

        features = extract_features_from_signal(
            window,
            sampling_rate
        )

        results.append(features)

    if not results:

        raise ValueError(
            f"{csv_path.name} is shorter than "
            f"the required window."
        )

    return pd.DataFrame(results)


# ============================================================
# BUILD COMPLETE TRAINING DATASET
# ============================================================

def build_dataset(data_directory):

    data_directory = Path(data_directory)

    all_rows = []

    # Process each of the five classes
    for class_name in CLASS_NAMES:

        class_directory = (
            data_directory / class_name
        )

        if not class_directory.exists():

            print(
                f"WARNING: Missing folder: "
                f"{class_directory}"
            )

            continue

        csv_files = sorted(
            class_directory.glob("*.csv")
        )

        if not csv_files:

            print(
                f"WARNING: No CSV files found in "
                f"{class_directory}"
            )

            continue

        # Process every recording
        for csv_file in csv_files:

            try:

                features_df = extract_features_from_csv(
                    csv_file
                )

                # Add filename
                features_df["source_file"] = str(
                    csv_file
                )

                # Add class label
                features_df["label"] = class_name

                all_rows.append(
                    features_df
                )

                print(
                    f"Processed: "
                    f"{class_name}/{csv_file.name} "
                    f"({len(features_df)} windows)"
                )

            except Exception as error:

                print(
                    f"ERROR processing "
                    f"{csv_file}: {error}"
                )

    # Make sure something was found
    if not all_rows:

        raise RuntimeError(
            "No usable CSV files were found."
        )

    # Combine all recordings
    dataset = pd.concat(
        all_rows,
        ignore_index=True
    )

    return dataset
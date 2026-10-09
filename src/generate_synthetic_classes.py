"""Generate synthetic MeterWhisperer recordings for additional classes.

This script creates 20 CSV files each for:

- toilet_leak
- pipe_fitting

The output format matches the existing recordings in data/raw/*:
time_s, acc_x_m_s2, acc_y_m_s2, acc_z_m_s2, acc_abs_m_s2, acc_accuracy

Each recording is 30 seconds at 100 Hz (3000 samples).
"""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "raw"
SAMPLE_RATE_HZ = 100
DURATION_SECONDS = 30
SAMPLE_COUNT = SAMPLE_RATE_HZ * DURATION_SECONDS
HEADER = "time_s,acc_x_m_s2,acc_y_m_s2,acc_z_m_s2,acc_abs_m_s2,acc_accuracy"


def make_rng(class_name, file_index):
    seeds = {
        "toilet_leak": 4400,
        "pipe_fitting": 5500,
    }
    return np.random.default_rng(seeds[class_name] + file_index)


def toilet_envelope(time_values, rng, file_index):
    """Persistent refill-like leak with slow pulsing and light variability."""
    base = rng.uniform(0.039, 0.050)
    modulation_frequency = rng.uniform(0.35, 0.75)
    modulation = rng.uniform(0.015, 0.024) * np.sin(
        2 * np.pi * modulation_frequency * time_values + rng.uniform(0, 2 * np.pi)
    )
    slow_breathing = rng.uniform(0.004, 0.010) * np.sin(
        2 * np.pi * rng.uniform(0.05, 0.12) * time_values
    )
    jitter = rng.normal(0, rng.uniform(0.002, 0.004), len(time_values))

    # Small deterministic variation across files.
    file_bias = (file_index - 10.5) * 0.00035
    envelope = base + file_bias + modulation + slow_breathing + jitter
    return np.clip(envelope, 0.012, None)


def pipe_envelope(time_values, rng, file_index):
    """Irregular pipe/fitting chatter with intermittent mechanical bursts."""
    baseline = rng.uniform(0.008, 0.016)
    envelope = baseline + rng.normal(0, 0.002, len(time_values))

    # Add short bursts at irregular intervals.
    burst_count = rng.integers(18, 34)
    for _ in range(burst_count):
        center = rng.uniform(0.3, DURATION_SECONDS - 0.3)
        width = rng.uniform(0.025, 0.11)
        amplitude = rng.uniform(0.040, 0.085)
        envelope += amplitude * np.exp(-0.5 * ((time_values - center) / width) ** 2)

    # A faint repetitive mechanical component.
    envelope += rng.uniform(0.004, 0.012) * (np.sin(2 * np.pi * rng.uniform(1.4, 2.8) * time_values) > 0.65)
    file_bias = (file_index - 10.5) * 0.0002
    return np.clip(envelope + file_bias, 0.002, None)


def generate_recording(class_name, file_index):
    rng = make_rng(class_name, file_index)
    time_values = np.arange(SAMPLE_COUNT) / SAMPLE_RATE_HZ

    if class_name == "toilet_leak":
        envelope = toilet_envelope(time_values, rng, file_index)
        frequency = rng.uniform(13.0, 17.5)
        harmonic = rng.uniform(2.0, 4.5)
    elif class_name == "pipe_fitting":
        envelope = pipe_envelope(time_values, rng, file_index)
        frequency = rng.uniform(20.0, 27.5)
        harmonic = rng.uniform(5.0, 9.0)
    else:
        raise ValueError(f"Unsupported class: {class_name}")

    phase_y = rng.uniform(1.8, 2.4)
    phase_z = rng.uniform(3.8, 4.6)
    sensor_noise = rng.normal(0, (envelope * 0.18)[:, None], (SAMPLE_COUNT, 3))

    carrier = np.sin(2 * np.pi * frequency * time_values)
    harmonic_wave = 0.22 * np.sin(2 * np.pi * harmonic * time_values + rng.uniform(0, 2 * np.pi))

    x = envelope * (carrier + harmonic_wave) + sensor_noise[:, 0]
    y = envelope * (np.sin(2 * np.pi * frequency * time_values + phase_y) + 0.18 * harmonic_wave) + sensor_noise[:, 1]
    z = envelope * (np.sin(2 * np.pi * frequency * time_values + phase_z) + 0.20 * harmonic_wave) + sensor_noise[:, 2]
    acc_abs = np.sqrt(x**2 + y**2 + z**2)
    accuracy = np.full(SAMPLE_COUNT, 3)

    return np.column_stack([time_values, x, y, z, acc_abs, accuracy])


def write_csv(path, data):
    np.savetxt(
        path,
        data,
        delimiter=",",
        header=HEADER,
        comments="",
        fmt=["%.6f", "%.12f", "%.12f", "%.12f", "%.12f", "%d"],
    )


def main():
    for class_name in ["toilet_leak", "pipe_fitting"]:
        class_dir = DATA_ROOT / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for old_file in class_dir.glob("*.csv"):
            old_file.unlink()

        for file_index in range(1, 21):
            data = generate_recording(class_name, file_index)
            output_path = class_dir / f"{class_name}_{file_index:02d}_SYNTHETIC.csv"
            write_csv(output_path, data)
            print(f"Created {output_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
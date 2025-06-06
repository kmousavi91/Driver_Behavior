import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
import os

# === Config ===
RAW_FILE = "data/sensor_raw.csv"
OUTPUT_CLEANED = "data/cleaned_sensor_data.csv"
WINDOW_FEATURES = "data/windowed_features.csv"
WINDOW_SIZE = 20  # e.g., 2 seconds of data at 10Hz

# === Load raw data ===
df = pd.read_csv(RAW_FILE)
print(f"Raw data shape: {df.shape}")
print(df.info())

# === Basic Cleaning ===
df = df.dropna().reset_index(drop=True)
print(f"After dropping NaNs: {df.shape}")

# === Summary statistics ===
print(df.describe())

# === Plot example accelerometer data ===
if set(['AccX', 'AccY', 'AccZ']).issubset(df.columns):
    plt.figure(figsize=(12, 6))
    plt.plot(df['AccX'], label='AccX')
    plt.plot(df['AccY'], label='AccY')
    plt.plot(df['AccZ'], label='AccZ')
    plt.title('Accelerometer Over Time')
    plt.legend()
    plt.xlabel("Sample Index")
    plt.ylabel("Acceleration")
    plt.tight_layout()
    plt.savefig("plots/acc_plot.png")
    plt.show()
    plt.close()
else:
    print("Accelerometer columns not found.")

# === Feature Engineering: Magnitudes ===
df['AccMagnitude'] = np.sqrt(df['AccX']**2 + df['AccY']**2 + df['AccZ']**2)
df['GyroMagnitude'] = np.sqrt(df['GyroX']**2 + df['GyroY']**2 + df['GyroZ']**2)

# === Rolling Features (example) ===
df['AccMagnitude_mean_5'] = df['AccMagnitude'].rolling(window=5).mean()
df['GyroMagnitude_mean_5'] = df['GyroMagnitude'].rolling(window=5).mean()

# Drop NaNs caused by rolling
df = df.dropna().reset_index(drop=True)

# === Save cleaned data ===
df.to_csv(OUTPUT_CLEANED, index=False)
print(f"Cleaned data saved to: {OUTPUT_CLEANED}")

# === Windowing for model input ===
features = []
for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
    window = df.iloc[start:start + WINDOW_SIZE]
    try:
        label = mode(window['Target(Class)'], keepdims=True).mode[0]
    except Exception:
        label = mode(window['Target(Class)'])[0][0]

    feat = {
        'AccX_mean': window['AccX'].mean(),
        'AccY_mean': window['AccY'].mean(),
        'AccZ_mean': window['AccZ'].mean(),
        'GyroX_mean': window['GyroX'].mean(),
        'GyroY_mean': window['GyroY'].mean(),
        'GyroZ_mean': window['GyroZ'].mean(),
        'AccMagnitude_mean': window['AccMagnitude'].mean(),
        'GyroMagnitude_mean': window['GyroMagnitude'].mean(),
        'AccX_std': window['AccX'].std(),
        'AccY_std': window['AccY'].std(),
        'AccZ_std': window['AccZ'].std(),
        'GyroX_std': window['GyroX'].std(),
        'GyroY_std': window['GyroY'].std(),
        'GyroZ_std': window['GyroZ'].std(),
        'AccMagnitude_std': window['AccMagnitude'].std(),
        'GyroMagnitude_std': window['GyroMagnitude'].std(),
        'Target': label
    }

    features.append(feat)

features_df = pd.DataFrame(features)
features_df.to_csv(WINDOW_FEATURES, index=False)
print(f"Windowed features saved to: {WINDOW_FEATURES}")
print(features_df.head())


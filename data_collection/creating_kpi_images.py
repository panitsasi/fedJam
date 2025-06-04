import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from config import CSV_FILE_APP, CSV_FILE_PHY, TIME_INFO_FILE, KPI_OUTPUT_FOLDER, SPECTROGRAM_FOLDER



# === PARAMETERS ===
WINDOW_DURATION = 20  
TARGET_SIZE = 224  
# OUTPUT_BASE = 'KPI_in_Images/single_tone_gain_30_fft_1024'  

# # === PATHS ===
# app_kpis_path = 'APP_KPIs/single_tone_gain_30_fft_1024.csv'
# phy_kpis_path = 'PHY_KPIs/single_tone_gain_30_fft_1024.csv'
# time_info_path = 'spectrograms/single_tone_gain_30_fft_1024/time_info.csv'

OUTPUT_BASE = KPI_OUTPUT_FOLDER 

# === PATHS ===
# app_kpis_path = 'APP_KPIs/single_tone_gain_30_fft_1024.csv'
# phy_kpis_path = 'PHY_KPIs/single_tone_gain_30_fft_1024.csv'
# time_info_path = 'spectrograms/single_tone_gain_30_fft_1024/time_info.csv'

app_kpis_path = CSV_FILE_APP
phy_kpis_path = CSV_FILE_PHY
time_info_path = TIME_INFO_FILE


# === LOAD DATA ===
app_df = pd.read_csv(app_kpis_path)
phy_df = pd.read_csv(phy_kpis_path)
time_info_df = pd.read_csv(time_info_path)

# === Parse Time columns ===
app_df['Time'] = pd.to_datetime(app_df['Time'], format='%H:%M:%S.%f')
phy_df['Time'] = pd.to_datetime(phy_df['Time'], format='%H:%M:%S.%f')
time_info_df['Time'] = pd.to_datetime(time_info_df['Time'], format='%H:%M:%S.%f')

# === Extract timestamps ===
image_times = time_info_df['Time'].tolist()
num_images = len(image_times)

print(f"Found {num_images} spectrogram images.")

# === Define KPIs and prepare output folders ===
kpi_mapping = {
    'Latency': app_df,
    'Jitter': app_df,
    'Packet_Loss_Count': app_df,
    'Noise': phy_df,
    'SNR': phy_df,
}

# Create output directories
for kpi_name in kpi_mapping.keys():
    output_folder = os.path.join(OUTPUT_BASE, kpi_name)
    os.makedirs(output_folder, exist_ok=True)

# === Helper function to interpolate KPI window ===
def extract_interpolated_window(df, kpi_name, start_time, end_time):
    # Filter KPI data in the 20-second window
    mask = (df['Time'] >= start_time) & (df['Time'] <= end_time)
    kpi_values = df.loc[mask, kpi_name].values

    if len(kpi_values) == 0:
        return np.zeros((TARGET_SIZE, TARGET_SIZE))
    
    # Interpolate to exactly TARGET_SIZE * TARGET_SIZE points
    original_idx = np.linspace(0, 1, len(kpi_values))
    target_idx = np.linspace(0, 1, TARGET_SIZE * TARGET_SIZE)

    interpolator = interp1d(original_idx, kpi_values, kind='linear', fill_value="extrapolate")
    interpolated_values = interpolator(target_idx)

    interpolated_values = np.clip(interpolated_values, 0, 1)

    # Reshape into (224, 224)
    image = interpolated_values.reshape((TARGET_SIZE, TARGET_SIZE))
    return image

# === Main loop to create images ===
for idx, img_time in enumerate(image_times):
    print(f"[INFO] Processing Image {idx+1}/{num_images} at {img_time}...")

    start_time = img_time - timedelta(seconds=WINDOW_DURATION)
    end_time = img_time

    for kpi_name, df_source in kpi_mapping.items():
        output_folder = os.path.join(OUTPUT_BASE, kpi_name)
        kpi_image = extract_interpolated_window(df_source, kpi_name.replace('_', ' '), start_time, end_time)

        # Save the image
        output_path = os.path.join(output_folder, f"{idx+1}.png")
        plt.imsave(output_path, kpi_image, cmap='gray', vmin=0, vmax=1)

print("All KPI images successfully created!")

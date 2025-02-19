import numpy as np
import pandas as pd
import mne

def load_eeg(file_path):
    """Carga datos EEG desde un archivo CSV o MAT."""
    if file_path.endswith(".csv"):
        eeg_data = pd.read_csv(file_path)
    elif file_path.endswith(".mat"):
        from scipy.io import loadmat
        eeg_data = loadmat(file_path)
    else:
        raise ValueError("Formato no soportado. Usa .csv o .mat")

    print(f"EEG Data cargada: {eeg_data.shape}")
    return eeg_data

def preprocess_eeg(eeg_data, sfreq=250):
    """Filtra y extrae bandas Alpha (8-12Hz) y Theta (4-8Hz)."""
    eeg_filtered = mne.filter.filter_data(eeg_data, sfreq=sfreq, l_freq=1, h_freq=40)
    alpha_power = eeg_filtered[:, 8:12].mean(axis=1)
    theta_power = eeg_filtered[:, 4:8].mean(axis=1)

    eeg_processed = pd.DataFrame({"theta_power": theta_power, "alpha_power": alpha_power})
    return eeg_processed

if __name__ == "__main__":
    eeg_file = "data/raw/eeg_data.csv"
    eeg_data = load_eeg(eeg_file)
    eeg_cleaned = preprocess_eeg(eeg_data)
    print("EEG procesado:\n", eeg_cleaned.head())


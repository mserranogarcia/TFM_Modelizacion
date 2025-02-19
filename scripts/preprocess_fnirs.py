import numpy as np
import pandas as pd
import mne

def load_fnirs(file_path):
    """Carga datos fNIRS desde un archivo CSV o SNIRF."""
    if file_path.endswith(".csv"):
        fnirs_data = pd.read_csv(file_path)
    elif file_path.endswith(".snirf"):
        fnirs_data = mne.io.read_raw_snirf(file_path)
    else:
        raise ValueError("Formato no soportado. Usa .csv o .snirf")

    print(f"fNIRS Data cargada: {fnirs_data.shape}")
    return fnirs_data

def preprocess_fnirs(fnirs_data):
    """Corrige artefactos y calcula cambios en HbO."""
    fnirs_data = fnirs_data - fnirs_data.mean()  # Normalización
    hbo_change = fnirs_data["HbO"].diff()  # Cambios en HbO
    fnirs_processed = pd.DataFrame({"hbo_change": hbo_change})
    return fnirs_processed

if __name__ == "__main__":
    fnirs_file = "data/raw/fnirs_data.csv"
    fnirs_data = load_fnirs(fnirs_file)
    fnirs_cleaned = preprocess_fnirs(fnirs_data)
    print("fNIRS procesado:\n", fnirs_cleaned.head())

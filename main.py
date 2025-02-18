import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nilearn import image, datasets, plotting
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Cargar los datos fMRI y palabras
def load_data(fmri_path, words_path):
    """Carga los datos de fMRI y la lista de palabras."""
    fmri_data = np.load(fmri_path)  # Asegúrate de que el archivo es .npy o ajusta la lectura
    words_df = pd.read_csv(words_path)
    return fmri_data, words_df

# Preprocesamiento de datos
def preprocess_data(fmri_data, words_df):
    """Normaliza los datos y los ajusta para el modelo."""
    fmri_data = (fmri_data - np.mean(fmri_data, axis=0)) / np.std(fmri_data, axis=0)
    words_vector = words_df.iloc[:, 1:].values  # Suponiendo que la primera columna es la palabra
    return fmri_data, words_vector

# Entrenar modelo de predicción (ejemplo: Ridge Regression)
def train_model(fmri_data, words_vector):
    """Entrena un modelo simple para predecir fMRI a partir de palabras."""
    X_train, X_test, y_train, y_test = train_test_split(words_vector, fmri_data, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Error cuadrático medio (MSE): {mse:.4f}")

    return model

# Visualizar resultados
def plot_results(model, X_test, y_test):
    """Visualiza la correlación entre valores reales y predichos."""
    y_pred = model.predict(X_test)
    plt.figure(figsize=(8,6))
    plt.scatter(y_test[:, 0], y_pred[:, 0], alpha=0.5)
    plt.xlabel("fMRI Real")
    plt.ylabel("fMRI Predicho")
    plt.title("Resultados del Modelo")
    plt.show()

# Script Principal
if __name__ == "__main__":
    fmri_data, words_df = load_data("data/fmri_data.npy", "data/words_data.csv")
    fmri_data, words_vector = preprocess_data(fmri_data, words_df)
    
    model = train_model(fmri_data, words_vector)
    plot_results(model, words_vector, fmri_data)

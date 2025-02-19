import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Importar funciones de preprocesamiento
from scripts.preprocess_eeg import load_eeg, preprocess_eeg
from scripts.preprocess_fnirs import load_fnirs, preprocess_fnirs
from scripts.preprocess_words import load_word_data

# Cargar los Datos de Palabras, EEG y fNIRS
def load_all_data():
    """Carga los datos de palabras, EEG y fNIRS."""
    words_file = "data/raw/word_lists.csv"
    corpus_file = "data/processed/corpus_normativo.csv"
    eeg_file = "data/raw/eeg_data.csv"
    fnirs_file = "data/raw/fnirs_data.csv"

    words_data = load_word_data(words_file, corpus_file)
    eeg_data = load_eeg(eeg_file)
    fnirs_data = load_fnirs(fnirs_file)

    return words_data, eeg_data, fnirs_data

# Preprocesar los Datos
def preprocess_all(words_data, eeg_data, fnirs_data):
    """Aplica preprocesamiento a palabras, EEG y fNIRS."""
    eeg_cleaned = preprocess_eeg(eeg_data)
    fnirs_cleaned = preprocess_fnirs(fnirs_data)

    return words_data, eeg_cleaned, fnirs_cleaned

# Entrenar Modelo de Predicción
def train_model(X, y):
    """Entrena un modelo de regresión Ridge para predecir memoria."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"📊 Error cuadrático medio (MSE): {mse:.4f}")

    return model, X_test, y_test

# Visualizar Resultados
def plot_results(model, X_test, y_test):
    """Visualiza la relación entre valores reales y predichos."""
    y_pred = model.predict(X_test)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.xlabel("Valores Reales")
    plt.ylabel("Valores Predichos")
    plt.title("Resultados del Modelo")
    plt.show()

# Ejecutar el Pipeline Completo
if __name__ == "__main__":
    print("Cargando datos...")
    words_data, eeg_data, fnirs_data = load_all_data()

    print("Preprocesando datos...")
    words_data, eeg_cleaned, fnirs_cleaned = preprocess_all(words_data, eeg_data, fnirs_data)

    print("Entrenando modelo...")
    model, X_test, y_test = train_model(words_data[["frecuencia", "imagenabilidad", "longitud"]], eeg_cleaned["theta_power"])

    print("Visualizando resultados...")
    plot_results(model, X_test, y_test)

    print("Proceso completado")

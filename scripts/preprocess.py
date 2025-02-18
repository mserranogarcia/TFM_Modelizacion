import os
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, masking
import matplotlib.pyplot as plt

def load_fmri_data(file_path):
    """ Carga un archivo NIfTI (.nii o .nii.gz) """
    fmri_img = nib.load(file_path)
    fmri_data = fmri_img.get_fdata()
    return fmri_data

def plot_brain_slice(fmri_data, slice_idx=50):
    """ Muestra una sección del cerebro en fMRI """
    plt.imshow(fmri_data[:, :, slice_idx, 0], cmap='gray')
    plt.title(f'Slice {slice_idx} de fMRI')
    plt.colorbar()
    plt.show()

def preprocess_fmri(file_path):
    """ Aplica suavizado y normalización a los datos fMRI """
    fmri_img = image.smooth_img(file_path, fwhm=6)  # Suavizado con kernel de 6mm
    fmri_data = fmri_img.get_fdata()
    fmri_data = (fmri_data - np.mean(fmri_data)) / np.std(fmri_data)  # Normalización Z-score
    return fmri_data

if __name__ == "__main__":
    # Prueba con un archivo de fMRI (cambia el path)
    sample_file = "data/raw/sample_fmri.nii.gz"
    
    if os.path.exists(sample_file):
        fmri_data = load_fmri_data(sample_file)
        plot_brain_slice(fmri_data)
        
        processed_data = preprocess_fmri(sample_file)
        plot_brain_slice(processed_data)
    else:
        print("Archivo fMRI no encontrado")

## **Objetivos (probablemente modificados en un futuro)**

Este proyecto tiene como objetivo analizar la relación entre las características de las palabras, la actividad cerebral y la memoria utilizando datos de EEG, fNIRS y respuestas conductuales obtenidas en un experimento de reconocimiento y recuerdo.

### **Objetivo general**
Modelar la memoria y reconocimiento de palabras en función de sus propiedades lingüísticas y los patrones de actividad cerebral medidos con EEG y fNIRS.

### **Objetivos específicos**
**Explorar la influencia de las características de las palabras**  
   - Analizar cómo factores como frecuencia, emoción y asociación semántica afectan el recuerdo y reconocimiento de palabras.  
   - Utilizar un corpus normativo para cuantificar estas propiedades.  

**Preprocesar y analizar datos EEG y fNIRS**  
   - Extraer características relevantes de EEG (potencia en bandas Alpha y Theta), asociadas con el proceso de memoria.  
   - Analizar cambios en HbO en fNIRS, reflejando activación en corteza prefrontal.  
   - Sincronizar estas señales con la presentación de palabras en el experimento.  

**Desarrollar modelos predictivos**  
   - Entrenar modelos de Machine Learning para predecir:
     - Memoria (palabras recordadas/no recordadas).  
     - Patrones cerebrales a partir de características de palabras.  
   - Evaluar el desempeño de los modelos con métricas como MSE y correlaciones.  

**Visualizar resultados y validar hipótesis**  
   - Comparar palabras recordadas vs. no recordadas en función de sus características.  
   - Explorar la relación entre señales EEG/fNIRS y la memoria.  
   - Interpretar los hallazgos para entender cómo el cerebro procesa y recuerda palabras.  


## Estructura 
- **data/raw/** → Datos originales
- **data/processed/** → Datos preprocesados
- **notebooks/** → Exploraciones iniciales en Jupyter
- **scripts/** → Código para preprocesamiento y modelado
- **results/** → Resultados y gráficos

## Scripts
- `preprocess_eeg.py` → Preprocesamiento de datos EEG
- `preprocess_fnirs.py` → Preprocesamiento de fNIRS
- `preprocess_words.py` → Unión de palabras con corpus normativo
- `main.py` → Entrenamiento y evaluación del modelo


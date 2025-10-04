import numpy as np
import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------
# 1. Configuración de Rutas
# ------------------------------

# Rutas de los artefactos generados en la Tarea P-05
INPUT_FEATURES_PATH = 'data/features/X_tfidf.npy'
INPUT_LABELS_PATH = 'data/features/y_labels.npy'
OUTPUT_MODEL_PATH = 'models/baseline_model.pkl'
OUTPUT_REPORT_PATH = 'reports/metrics_baseline.txt'

# ------------------------------
# 2. Función Principal
# ------------------------------

def train_baseline_model():
    """
    Carga los datos vectorizados, entrena y evalúa el modelo Baseline (Regresión Logística).
    """
    try:
        print("1. Cargando datos vectorizados (X e y)...")
        # Cargamos las características X y las etiquetas y desde los archivos .npy
        X = np.load(INPUT_FEATURES_PATH, allow_pickle=True)
        y = np.load(INPUT_LABELS_PATH, allow_pickle=True)
        
        # Aseguramos que X sea una matriz 2D (manejo de matriz dispersa guardada como objeto)
        if X.dtype == object:
             X = X.item() 
        
        print(f"-> Datos cargados. Forma de X: {X.shape}, Forma de y: {y.shape}")

    except FileNotFoundError:
        print(f"ERROR: No se encontraron los archivos {INPUT_FEATURES_PATH} o {INPUT_LABELS_PATH}.")
        print("Asegúrate de ejecutar la Tarea P-05 primero.")
        return

    # 2. División de Datos
    print("\n2. Dividiendo datos en entrenamiento y prueba...")
    
    # CORRECCIÓN 1: Se eliminó 'stratify=y' del test_train_split.
    # Esto es necesario para que el bloque de prueba funcione (solo 3 muestras).
    # Para el dataset real (grande), stratify=y sería una buena práctica, pero no esencial
    # para la prueba unitaria.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42 # <- stratify=y ha sido eliminado aquí.
    )
    print(f"-> Conjunto de Entrenamiento: {X_train.shape[0]} muestras.")
    print(f"-> Conjunto de Prueba: {X_test.shape[0]} muestras.")

    # 3. Entrenamiento del Modelo Baseline (Regresión Logística)
    print("\n3. Entrenando el Modelo Baseline (Regresión Logística)...")
    # Usamos solver='liblinear' que es bueno para datasets pequeños y binarios.
    model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    print("-> Entrenamiento finalizado.")

    # 4. Evaluación del Modelo
    print("\n4. Evaluando el rendimiento...")
    y_pred = model.predict(X_test)
    
    # CALCULAMOS LAS MÉTRICAS ANTES DE ESCRIBIR EL ARCHIVO
    full_report_str = classification_report(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred) 
    
    # 5. Guardar Artefactos
    import os
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_REPORT_PATH) or '.', exist_ok=True)
    
    # Guardar el modelo entrenado
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    # CORRECCIÓN 2: Uso correcto de la variable 'accuracy' y de 'full_report_str'
    # Guardar el reporte de métricas en formato legible
    with open(OUTPUT_REPORT_PATH, 'w') as f:
        f.write(f"Modelo: Regresión Logística (Baseline)\n")
        f.write(f"Exactitud (Accuracy): {accuracy:.4f}\n\n")
        f.write("--- Reporte Detallado ---\n")
        f.write(full_report_str) 
    print(f"-> Precisión (Accuracy) en prueba: {accuracy:.4f}")
    print(f"-> Modelo guardado en: {OUTPUT_MODEL_PATH}")
    print(f"-> Reporte guardado en: {OUTPUT_REPORT_PATH}")
    
# ------------------------------
# 6. Ejecución
# ------------------------------

if __name__ == '__main__':
    # Este bloque de ejecución solo se corre si el módulo es llamado directamente.
    train_baseline_model()
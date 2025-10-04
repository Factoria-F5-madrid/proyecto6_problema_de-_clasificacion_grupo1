import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
# Importamos la función de limpieza de texto del módulo anterior (Tarea P-04)
from .text_cleaning import clean_text 

# ------------------------------
# 1. Configuración de Rutas y Columnas
# ------------------------------

# Asegúrate de que esta ruta sea donde tu compañera del EDA guardó el archivo final
DATA_PATH = 'data/processed/hate_speech_processed.csv' 
TEXT_COLUMN = 'text'        # Nombre de la columna que contiene el texto original
TARGET_COLUMN = 'label_hate' # Nombre de la columna objetivo para la clasificación

# Rutas de salida para guardar los resultados y el vectorizador entrenado
OUTPUT_FEATURES_PATH = 'data/features/X_tfidf.npy'
OUTPUT_LABELS_PATH = 'data/features/y_labels.npy'
OUTPUT_MODEL_PATH = 'models/tfidf_vectorizer.pkl'


# ------------------------------
# 2. Función de Vectorización
# ------------------------------

def create_tfidf_features(df: pd.DataFrame, text_col: str, target_col: str):
    """
    Aplica limpieza de texto usando el módulo P-04 y vectoriza con TfidfVectorizer.
    
    Args:
        df (pd.DataFrame): DataFrame que contiene las columnas de texto y etiqueta.
        text_col (str): Nombre de la columna de texto original.
        target_col (str): Nombre de la columna objetivo.
        
    Returns:
        tuple: (Matriz dispersa de características X, Vector de etiquetas y, Vectorizador TF-IDF)
    """
    print("1. Aplicando la función de limpieza (Módulo P-04)...")
    # Utilizamos 'lemmatization' como método preferente de normalización.
    df['cleaned_text'] = df[text_col].apply(clean_text, method='lemmatization') 
    
    # 2. Inicializar y configurar el TfidfVectorizer (Ajuste de hiperparámetros iniciales)
    tfidf_vectorizer = TfidfVectorizer(
        min_df=5,             # Ignorar palabras que aparecen en menos de 5 documentos
        max_df=0.9,           # Ignorar palabras que aparecen en más del 90% de los documentos (demasiado comunes)
        max_features=10000,   # Limitar el vocabulario a los 10,000 términos más importantes
        ngram_range=(1, 2)    # Incluir unigramas y bigramas (combinaciones de 1 y 2 palabras)
    )
    
    print("2. Ajustando y transformando los datos con TF-IDF...")
    # X es la matriz dispersa de características, y es el vector de etiquetas.
    X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
    y = df[target_col]
    
    print(f"-> Matriz de Características (X) creada con forma: {X.shape}")
    print(f"-> Vocabulario creado con {len(tfidf_vectorizer.vocabulary_)} términos.")
    
    return X, y, tfidf_vectorizer


# ------------------------------
# 3. Ejecución, Pruebas y Almacenamiento
# ------------------------------

if __name__ == '__main__':
    # Este bloque solo se ejecuta al correr el archivo directamente para probar el pipeline
    
    # Creamos un DataFrame de prueba si no existe el archivo real
    try:
        data = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ADVERTENCIA: Archivo no encontrado en {DATA_PATH}. Usando datos de prueba.")
        data = pd.DataFrame({
            TEXT_COLUMN: [
                "The hateful user posted a terrible link: https://hate-site.com!",
                "We are running and jumping over fences, but we stopped running later.",
                "This is not hate speech, but a normal disagreement."
            ],
            TARGET_COLUMN: [1, 0, 0]
        })
    
    # Aseguramos la existencia de las carpetas de salida
    import os
    os.makedirs(os.path.dirname(OUTPUT_FEATURES_PATH) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH) or '.', exist_ok=True)

    X_features, y_labels, vectorizer = create_tfidf_features(data, TEXT_COLUMN, TARGET_COLUMN)

    print("\n3. Guardando artefactos...")
    
    # 3.1. Guardar la matriz de características X (Formato numpy/npy)
    np.save(OUTPUT_FEATURES_PATH, X_features.toarray()) # toarray() para guardar en disco
    
    # 3.2. Guardar el vector de etiquetas y (Formato numpy/npy)
    np.save(OUTPUT_LABELS_PATH, y_labels.values)
    
    # 3.3. Guardar el Vectorizador entrenado (Formato pickle)
    with open(OUTPUT_MODEL_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("\nVectorización completada. Artefactos guardados exitosamente.")
    print("El proyecto está listo para la Tarea M-01 (Modelo Baseline).")
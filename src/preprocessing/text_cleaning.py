import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer 

# ------------------------------
# 1. Configuración de recursos
# ------------------------------

# 1.1. Configuración de Stopwords y Stemmer/Lemmatizer.
# Suponemos que el dataset de 'measuring-hate-speech' es predominantemente en inglés
# ajustamos la lista de stopwords y la documentación
# Si se decide que es en español, la variable debe cambiarse a 'spanish'
STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer() 

# ------------------------------
# 2. Función Principal: clean_text
# ------------------------------

def clean_text(text: str, method: str = 'lemmatization') -> str:
    """
    Implementa el pipeline completo de limpieza y normalización de texto.
    
    Args:
        text (str): La cadena de texto a limpiar.
        method (str): Técnica de normalización a aplicar ('lemmatization' o 'stemming').
                      'lemmatization' es la preferida por su precisión[cite: 60].
        
    Returns:
        str: El texto limpio y normalizado.
    """
    # 1. Convertir a minúsculas para uniformidad[cite: 56].
    text = text.lower()
    
    # 2. Detectar y eliminar URLs presentes en el texto[cite: 58].
    # Utilizamos una expresión regular para patrones comunes de URL.
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # 3. Eliminar signos de puntuación y caracteres especiales[cite: 57].
    # Creamos una tabla de traducción: cada carácter de puntuación se mapea a None (eliminación).
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # 4. Tokenización (dividir en palabras) y filtrado de Stopwords.
    words = text.split()
    
    # Filtrar Stopwords. Eliminamos palabras comunes sin valor semántico.
    words = [word for word in words if word not in STOPWORDS]
    
    # 5. Aplicar técnicas de normalización[cite: 59].
    if method == 'lemmatization':
        # Lematización (preferente): Reduce a la forma base (lema).
        words = [LEMMATIZER.lemmatize(word) for word in words]
    elif method == 'stemming':
        # Stemming (alternativa/rápida): Recorta a la raíz (stem)[cite: 61].
        words = [STEMMER.stem(word) for word in words]
        
    # 6. Unir las palabras de nuevo en una cadena de texto.
    return ' '.join(words)

# ------------------------------
# 3. Documentación y Prueba
# ------------------------------

if __name__ == '__main__':
    # Guardar el script para que sea reutilizable y documentado[cite: 62, 63, 65].
    
    sample_text = (
        "The hateful user posted a terrible link: https://hate-site.com! "
        "We are running and jumping over fences, but we stopped running later."
    )
    
    print(f"--- Texto Original ---\n'{sample_text}'\n")
    
    # Prueba con Lematización
    lem_result = clean_text(sample_text, method='lemmatization')
    print(f"--- Resultado con Lematización (Preferente) ---\n'{lem_result}'\n")
    
    # Prueba con Stemming (Alternativa)
    stem_result = clean_text(sample_text, method='stemming')
    print(f"--- Resultado con Stemming (Alternativa) ---\n'{stem_result}'\n")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Modulo para recomendaciones basadas en contenido usando TF-IDF y similitud de coseno.


"""
    Construye el vectorizador TF-IDF y la matriz dispersa de características.
    Parámetros:
    - books: DataFrame con columnas 'Title', 'Author', 'Publisher', 'Year'.
    - max_features: número máximo de términos en el vocabulario TF-IDF.
    - min_df: frecuencia mínima de documentos para incluir un término.

    Retorna:
    - vec: objeto TfidfVectorizer entrenado.
    - M: matriz TF-IDF (sparse) de tipo float32 (paara eficiencia de memoria).
"""
def build_tfidf(books, max_features, min_df):
    # Columnas que realmente tenemos en el data loader
    columnas_texto = ['title', 'description', 'categories', 'published_date']

    # Normalizar columnas y asegurar que existan
    for col in columnas_texto:
        if col not in books.columns:
            books[col] = ''
        books[col] = books[col].fillna('').astype(str).str.lower()

    # Concatenar columnas para generar el texto
    content = (
        books['title'] + ' ' +
        books['description'] + ' ' +
        books['categories'] + ' ' +
        books['published_date']
    )

    # Crear TF-IDF
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        dtype=np.float32
    )
    matrix_tfidf = vec.fit_transform(content)

    return vec, matrix_tfidf


"""
    Generar recomendaciones basadas en similitud de contenido.

    Parámetros:
    - book_title: título del libro de referencia (string).
    - books: DataFrame original usado para construir M.
    - vec: vectorizador TF-IDF entrenado
    - M: matriz TF-IDF resultante
    - top_n: número de recomendaciones a devolver
    - thresh: umbral mínimo de similitud de coseno (0 a 1)

    Retorna:
    - DataFrame con columnas 'Title' y 'Score' ordenado descendente.
    """

def recommend_content(book_title, books, matrix_tfidf, num_recommendations, similarity_threshold):
    # localizar por POSICIÓN (no etiquetas) el índice del libro de referencia
    q = (book_title or "").strip().casefold()
    title_series = books['title'].astype(str).str.casefold()
    pos = np.flatnonzero(title_series.values == q)
    if pos.size == 0:
        raise KeyError(f"Libro '{book_title}' no encontrado")
    ref_pos = int(pos[0])

    # similitudes
    sim = cosine_similarity(matrix_tfidf[ref_pos], matrix_tfidf).ravel()
    sim[ref_pos] = -1.0  # excluir el mismo libro

    # filtrar por umbral y tomar top-N
    valid = np.where(sim >= float(similarity_threshold))[0]
    if valid.size == 0:
        return pd.DataFrame(columns=['title', 'similarity'])

    k = min(int(num_recommendations), valid.size)
    top_idx = valid[np.argsort(sim[valid])[::-1][:k]]

    return pd.DataFrame({
        'title': books.iloc[top_idx]['title'].values,
        'similarity': sim[top_idx]
    })

#verified
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import normalize_title 

# ------------------------------------------------------------------------------------
# Construcción de TF-IDF -Term Frequency-Inverse Document Frequency (Frecuencia de Término - Frecuencia Inversa de Documento)
""""
prepara los datos de texto de los libros y construye la matriz de importancia TF-IDF.
    parámetros:
        - books: dataframe de pandas con la información de los libros.
        - max_features: número máximo de palabras únicas a considerar.
        - min_df: frecuencia minima de documento para incluir una palabra en el vocabulario. (filtro de frecuencia)

    devuelve:
        - vectorizador TF-IDF ajustado.
        - matriz TF-IDF (sparse matrix).
"""
def build_tfidf(books, max_features, min_df):

    # las columnas de texto a usar para construir el perfil de contenido de cada libro
    columnas_texto = ['title', 'description', 'categories', 'published_date']

    for col in columnas_texto:
        # asegurarnos de que las columnas existan y limpiamos valores vacíos / NaN
        if col not in books.columns:
            books[col] = ''
        books[col] = books[col].fillna('').astype(str).str.lower()

    # unificamos todo el texto de un libro en una sola cadena para el analisis TF-IDF
    texto_unificado = (
        books['title'] + ' ' +
        books['description'] + ' ' +
        books['categories'] + ' ' +
        books['published_date']
    )

    # si todos los textos están vacíos, no podemos construir la matriz TF-IDF ni nada
    if texto_unificado.str.strip().eq("").all():
        print(" no hay documentos válidos para TF-IDF.")
        return None, None

    # contruimos la matriz TF-IDF con los limies de palabras definidos, calcula cuantas veces aparece una palabra TF y la importancia de esa palabra en el conjunto de documentos IDF
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        dtype=np.float32
    )
    # ajustamos la herramienta y creamos la matriz de importancia donde cada fila es un libro y cada columna una palabra  
    matrix_tfidf = vec.fit_transform(texto_unificado)

    return vec, matrix_tfidf


# ------------------------------------------------------------------------------------
# Recomendaciones basadas en contenido
"""
recomienda contenido similar basado en el contenido textual utilizando la similitud de coseno en la matriz TF-IDF.
    parámetros:
        - book_title: título del libro de referencia.
        - books: dataframe de pandas con la información de los libros.
        - matrix_tfidf: matriz de importancia previamente calculada TF-IDF (sparse matrix).
        - num_recommendations: número de recomendaciones a devolver.
        - similarity_threshold: umbral mínimo de similitud para considerar una recomendación válida. (0.0 a 1.0)

    devuelve:
        - dataframe con los títulos recomendados y sus puntuaciones de similitud.
"""
def recommend_content(book_title, books, matrix_tfidf, num_recommendations, similarity_threshold):
    
    # normalizar el título de entrada y del dataframe para encontrar coincidencias
    q = normalize_title(book_title)
    title_series = books['title'].apply(normalize_title)

    # encontrar la posición del libro de referencia en el dataframe
    position = np.flatnonzero(title_series.values == q)

    #sino la encontramos, lanzamos error
    if position.size == 0:
        raise KeyError(f"Libro '{book_title}' no encontrado después de normalización")
    ref_pos = int(position[0])

    # similitudes de coseno: aqui comparamos el libro de referencia con TODOS los demás libros
    sim = cosine_similarity(matrix_tfidf[ref_pos], matrix_tfidf).ravel()
    sim[ref_pos] = -1.0  # excluir el mismo libro

    # filtrar por threshold o nivel de exigencia, nos quedamos con los que superan el umbral
    valid = np.where(sim >= float(similarity_threshold))[0]
    # sino se supera el umbral, devolvemos vacio
    if valid.size == 0:
        return pd.DataFrame(columns=['title', 'similarity'])

    # determinamos cuántas recomendaciones devolver sin superar el número de libros válidos
    k = min(int(num_recommendations), valid.size)

    #ordeamos lo q si son validos de mayor a menor similitud y nos quedamos con los k mejores
    top_idx = valid[np.argsort(sim[valid])[::-1][:k]]

    #devolvemos el resultado con los títulos y sus puntuaciones de similitud
    return pd.DataFrame({
        'title': books.iloc[top_idx]['title'].values,
        'similarity': sim[top_idx]
    })

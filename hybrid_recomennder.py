import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nmf_recommender import crear_matriz_dispersa, entrenar_nmf
from content_recommender import build_tfidf


"""
    normalizar arreglo numpy a un rango de 0 a 1.

    parámetro arr: arreglo numpy a normalizar.
    devuelve: arreglo numpy normalizado.
"""
def normalizar (arr: np.ndarray) -> np.ndarray:
    # asegurarse de que el arreglo no esté vacío
    if arr.size == 0:
        return np.zeros_like(arr) # si el arreglo está vacío, devolver un arreglo de ceros del mismo tamaño
    
    min_v, max_v = arr.min(), arr.max() # se obtiene el valor mínimo y máximo del arreglo
    
    if max_v - min_v < 1e-8: # si la diferencia entre el valor máximo y mínimo es muy pequeña, devolver un arreglo de ceros
        # esto evita la división por cero
        return np.zeros_like(arr)
    else :
        return(arr - min_v) / (max_v - min_v)



#-----------------------------------------------------------------------------------------------------
"""
    calcular los pesos para el filtrado colaborativo y de contenido basado en la cantidad de valoraciones del usuario.
    parámetros:
    user_ratings: número de valoraciones que el usuario ha hecho.
    confidence_threshold: número mínimo de valoraciones para considerar que el usuario tiene información suficiente (por ejemplo, 10).
    weight_collab: peso máximo que se quiere dar al filtrado colaborativo
    devuelve: una tupla con los pesos ajustados para el colaborativo y peso para el de contenido según la cantidad de valoraciones del usuario y el umbral de confianza.
"""
def calcular_pesos (n_ratings, confidence_thereshold, weight_collab):
    
    fraccion = min(n_ratings / confidence_thereshold, 1.0)
    weight_collab_after_confidence_thereshold = weight_collab * (fraccion ** 0.5)

    return weight_collab_after_confidence_thereshold, 1 - weight_collab_after_confidence_thereshold



#-----------------------------------------------------------------------------------------------------
"""
    obtiene el puntaje colaborativo para un usuario específico utilizando NMF.
    parámetros:
    user_id: id del usuario para el cual se calculan las recomendaciones.
    ratings_df: DataFrame con las valoraciones de los usuarios.
    books_df: DataFrame con los libros.
    n_components: número de componentes para NMF.
    max_iterations: número máximo de iteraciones para el entrenamiento de NMF.

    devuelve: un arreglo numpy con los puntajes colaborativos normalizados para todos los libros.
    lanza KeyError si el usuario no se encuentra en el DataFrame de valoraciones.
"""
def obtener_puntaje_colaborativo(user_id, ratings_df, books_df, n_components, max_iterations):
    rating_matrix = crear_matriz_dispersa(ratings_df)
    if user_id not in rating_matrix.user_to_idx:
        raise KeyError(f"Usuario {user_id} no encontrado")

    max_factors = min(rating_matrix.shape) - 1
    if n_components > max_factors:
        n_components = max_factors

    _, W, H = entrenar_nmf(rating_matrix, n_components=n_components, max_iter=max_iterations)
    user_index = rating_matrix.user_to_idx[user_id]

    collaborative_scores = W.dot(H[:, user_index])
    print("collaborative_scores min/max:", collaborative_scores.min(), collaborative_scores.max())

    collab_norm = normalizar(collaborative_scores)
    print("collab_norm min/max:", collab_norm.min(), collab_norm.max())

    # mapeo por título normalizado
    books_titles_norm = books_df['title'].astype(str).str.strip().str.casefold().values
    title_to_idx = {t: i for i, t in enumerate(books_titles_norm)}

    collaborative_full = np.zeros(len(books_df), dtype=float)
    matches = 0
    for idx_train, score in enumerate(collab_norm):
        t_train = str(rating_matrix.idx_to_book.get(idx_train, ""))
        t_train_norm = t_train.strip().casefold()
        idx_full = title_to_idx.get(t_train_norm)
        if idx_full is not None:
            collaborative_full[idx_full] = score
            matches += 1
    print(f"Matches colaborativo (NMF->books): {matches}/{len(collab_norm)}")

    return collaborative_full  # ⬅️ IMPRESCINDIBLE




#-----------------------------------------------------------------------------------------------------
"""
obtenemos l puntaje basado en contenido para un usuario específico utilizando TF-IDF y similitud del coseno.
parámetros:
user_id: id del usuario para el cual se calculan las recomendaciones.
ratings_df: DataFrame con las valoraciones de los usuarios.
books_df: DataFrame con los libros.
tfidf_max_features: número máximo de características para TF-IDF.
tfidf_min_df: Número mínimo de documentos en los que debe aparecer una palabra para ser considerada.
"""
def obtener_puntaje_contenido(user_id, ratings_df, books_df, tfidf_max_features, tfidf_min_df):
    tfidf_vec, tfidf_matrix = build_tfidf(books_df, max_features=tfidf_max_features, min_df=tfidf_min_df)

    # --- NORMALIZACIÓN DE CLAVES PARA EL MAPE0 ---
    books_titles_norm = books_df['title'].astype(str).str.strip().str.casefold().values
    title_to_idx = {t: i for i, t in enumerate(books_titles_norm)}

    user_titles = ratings_df[ratings_df['user_id'] == user_id]['title'].astype(str).values
    user_titles_norm = [t.strip().casefold() for t in user_titles]
    user_indices = [title_to_idx[t] for t in user_titles_norm if t in title_to_idx]

    if not user_indices:
        return np.zeros(len(books_df), dtype=float), len(user_titles)

    sims = cosine_similarity(tfidf_matrix[user_indices], tfidf_matrix)
    cont_scores = sims.mean(axis=0)

    print("User:", user_id)
    print("Libros valorados (norm):", user_titles_norm)
    print("Índices encontrados:", user_indices)
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("Máximo puntaje contenido antes de normalizar:", float(np.max(cont_scores)))
    print("Mínimo puntaje contenido antes de normalizar:", float(np.min(cont_scores)))

    return normalizar(cont_scores), len(user_titles)



#-----------------------------------------------------------------------------------------------------
"""
obtiene un puntaje de popularidad para los libros basado en la cantidad de valoraciones.
parámetros:
ratings_df: DataFrame con las valoraciones de los usuarios.
books_df: DataFrame con la información d los libros.

devuelve: un arreglo numpy con los puntajes de popularidad normalizados para todos los libros.
"""
def obtener_valor_aware(ratings_df, books_df):
    # normalizar claves en ambos dataframes
    rtitles = ratings_df['title'].astype(str).str.strip().str.casefold()
    btitles = books_df['title'].astype(str).str.strip().str.casefold().values

    pop_counts = rtitles.value_counts()
    popularity_arr = np.array([pop_counts.get(t, 0) for t in btitles], dtype=float)
    print("unique popularity counts:", np.unique(popularity_arr))

    # inverso de popularidad
    inverse = 1.0 / (popularity_arr + 1.0)

    # si todos los valores son iguales, NO normalices (evita vector de ceros)
    if np.allclose(inverse.max() - inverse.min(), 0.0):
        return inverse

    # normalización min-max
    return (inverse - inverse.min()) / (inverse.max() - inverse.min())





#-----------------------------------------------------------------------------------------------------
"""
generar recomendaciones híbridas para un usuario específico combinando filtrado colaborativo, basado en contenido y popularidad.
parámetros:
user_id: id del usuario para el cual se calculan las recomendaciones.
books_df: DataFrame con la información de los libros.
ratings_df: DataFrame con las valoraciones de los usuarios.
n_components: número de componentes para NMF.
max_iterations: número máximo de iteraciones para el entrenamiento de NMF.
tfidf_max_features: número máximo de características para TF-IDF.
tfidf_min_df: número mínimo de documentos en los que debe aparecer una palabra para ser considerada.
weight_collab: peso máximo que se quiere dar al filtrado colaborativo.
weight_value: peso máximo que se quiere dar al valor basado en popularidad.
top_k: número de recomendaciones a devolver.
confidence_threshold: número mínimo de valoraciones para considerar que el usuario tiene información suficiente

devuelve: un DataFrame con los títulos de los libros recomendados y sus puntajes finales, ordenados de mayor a menor.
"""
def recomendacion_hibrida(user_id, books_df, ratings_df,
                         n_components, max_iterations,
                         tfidf_max_features, tfidf_min_df,
                         weight_collab, inverse_popularity,
                         top_k, confidence_threshold,
                         gamma):

    collaborative_full = obtener_puntaje_colaborativo(user_id, ratings_df, books_df, n_components, max_iterations)
    content_norm, n_ratings = obtener_puntaje_contenido(user_id, ratings_df, books_df, tfidf_max_features, tfidf_min_df)

    collab_weight_adjusted, content_weight_eff = calcular_pesos(n_ratings, confidence_threshold, weight_collab)

    print(f"collab_weight_adjusted: {collab_weight_adjusted}, content_weight_eff: {content_weight_eff}")
    print(f"collaborative_full min/max: {collaborative_full.min()}/{collaborative_full.max()}")
    print(f"content_norm min/max: {content_norm.min()}/{content_norm.max()}")


    raw_score = collab_weight_adjusted * collaborative_full + content_weight_eff * content_norm
    #print(f"Raw score: min={raw_score.min()}, max={raw_score.max()}")

    # Mezcla lineal ponderada con gamma para controlar peso de popularidad
    final_score = (1 - gamma) * raw_score + gamma * inverse_popularity

    seen_titles = set(ratings_df[ratings_df['user_id'] == user_id]['title'])
    df_scores = pd.DataFrame({'title': books_df['title'], 'final_score': final_score})

    df_scores = df_scores[~df_scores['title'].isin(seen_titles)]

    print(f"gamma: {gamma}")
    print(f"Min/Max raw_score: {raw_score.min()}/{raw_score.max()}")
    print(f"Min/Max inverse_popularity: {inverse_popularity.min()}/{inverse_popularity.max()}")
    print(f"Min/Max final_score: {final_score.min()}/{final_score.max()}")


    return df_scores.sort_values('final_score', ascending=False).head(top_k)


#verified
#debugging info added
#CAMBIOSSSSSSSSSSSSSSSSSSSSS
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nmf_recommender import NMF_recommender
from content_recommender import build_tfidf

# ------------------------------------------------------------------------------------
"""
NORMALIZA un array uando la min-max normalization para escalar los valores entre 0 y 1

    parametros:
        arr (np.ndarray): array de entrada a normalizar

    devuelve:
        np.ndarray: array normalizado entre 0 y 1
"""

def normalizar(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return np.zeros_like(arr)
    min_v, max_v = arr.min(), arr.max()

    # evitar division por cero si todos los valores son iguales
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr)
    
    return (arr - min_v) / (max_v - min_v)


# ------------------------------------------------------------------------------------
"""
AJUSTA el peso del modelo colaborativo basandose en el num de calficaciones del usuario 
si el num de ratings (n_ratings) es bajo, el peso colavorativo se reduce, menor confianza,
y se le da mas peso al modelo de contenido.

    parametros:
        n_ratings (int): numero de calificaciones del usuario
        confidence_threshold (int): mínimo de ratings para dar el peso completo a lo colaborativo
        weight_collab (float): peso base del modelo colaborativo (0 a 1)

    devuelve:
        tuple: (peso ajustado del modelo colaborativo, peso del modelo de contenido)
"""
# Ajuste de pesos colaborativo/contenido
def calcular_pesos(n_ratings: int, confidence_threshold: int, weight_collab: float) -> tuple:

    # calculo del peso ajustado
    fraccion = min(n_ratings / confidence_threshold, 1.0)

    #raiz cuadrada para ajuste
    weight_collab_after_conf = weight_collab * (fraccion ** 0.5)

    # el peso restante pa que sume 1
    return weight_collab_after_conf, 1 - weight_collab_after_conf


# ------------------------------------------------------------------------------------
"""
OBTIENE el puntaje de predicción del modelo nmf ya entrenado para evitar reentrenar cada vezz


    parametros:
        user_id (str): id del usuario para el cual se generan las recomendaciones
        books_df (pd.DataFrame): dataframe con la información de los libros
        nmf_model_entrenado (Any): modelo NMF ya entrenado

    devuelve:
        np.ndarray: array con las puntuaciones predichas para todos los libros
"""
def obtener_puntaje_colaborativo(user_id: str, books_df: pd.DataFrame, nmf_model_entrenado: Any) -> np.ndarray:

    
    # ahora llamamos al método recommendacion del modelo pre-entrenado.
    try:
        df_collab = nmf_model_entrenado.recomendaciones(user_id, top_n=len(nmf_model_entrenado.idx_to_book), normalize=True)
    
    except KeyError:

        # si el usuario no tiene perfil en el modelo entrenado se devuelve un array de ceros
        return np.zeros(len(books_df))

    # se convierten los titulos predichos en indices en la matriz de libros
    title_to_idx = {t: i for i, t in enumerate(books_df['title'].unique())}
    scores_collab_full = np.zeros(len(title_to_idx))

    for _, row in df_collab.iterrows():
        idx = title_to_idx.get(row['title'])
        if idx is not None:

            #el resultado final es un vector donde cada libro tiene putuacion nmf
            scores_collab_full[idx] = row['score']

    return scores_collab_full



# ------------------------------------------------------------------------------------
""" 
CALCULA la puntuación de similitud de contenido utilizando la matriz tfidf pre-calculada
en esta función se contruye el perfil promedio de gustos del usuario basandose
en los libros que calificó en el set de entrenamiento y compara ese perfil con todo el catálogo

    parametros:
        user_id (str): id del usuario para el cual se generan las recomendaciones
        ratings_df (pd.DataFrame): dataframe con las calificaciones de los usuarios
        books_df (pd.DataFrame): dataframe con la información de los libros
        tfidf_matrix (np.ndarray): la matriz que contiene los perfiles de contenido los libros
        title_to_idx (Dict[str, int]): diccionario donde se mapea los titulos del libre a sus indices en la matriz tfidf
        

    devuelve:
        tuple: array con las puntuaciones de similitud normalizadas, numero de libros calificados por el usuario      

"""
def obtener_puntaje_contenido(user_id: str, ratings_df: pd.DataFrame, books_df: pd.DataFrame, 
                              tfidf_matrix: np.ndarray, title_to_idx: Dict[str, int]) -> Tuple[np.ndarray, int]:


    # perfil del usuario: índices de los libros que el usuario ha calificado en el set de entrenamiento.
    user_titles = ratings_df[ratings_df['user_id'] == user_id]['title'].unique()
    user_indices = [title_to_idx[t] for t in user_titles if t in title_to_idx]

    if not user_indices:
        # si el usuario no tiene ratings en el set de entrenamiento, no se puede calcular el perfil.
        return np.zeros(len(books_df)), len(user_titles)

    # similitud de coseno: compara el perfil promedio de los libros vistos con todos los de la matriz tfidf
    sims = cosine_similarity(tfidf_matrix[user_indices], tfidf_matrix)
    
    # perfil promedio: promediamos la similitud a traves de los libros vistos (axis = 0)
    cont_scores = sims.mean(axis=0)
    
    return normalizar(cont_scores), len(user_titles)


# ------------------------------------------------------------------------------------
"""
funcion principal que combia las puntuaciones colaborativas, de contenido y la popularidad inversa
para generar recomendaciones híbridas para un usuario dado.

    parametros:
        user_id (str): id del usuario para el cual se generan las recomendaciones
        books_df (pd.DataFrame): dataframe con la información de los libros
        ratings_df (pd.DataFrame): dataframe con las calificaciones de los usuarios

        nmf_model_entrenado: la instancia de nmfrecommender ya entrenada.
        tfidf_matrix: la matriz tfidf de contenido pre-calculada
        title_to_idx (Dict[str, int]): diccionario donde se mapea los titulos del libre a sus indices en la matriz tfidf

        
        weight_collab (float): peso base del modelo colaborativo (0 a 1)
        top_k (int): número de recomendaciones a devolver
        confidence_threshold (int): mínimo de ratings para dar el peso completo a lo colaborativo
        gamma (float): peso para la popularidad inversa (0 a 1)
        inverse_popularity (np.ndarray): array con las puntuaciones de popularidad inversa para todos los libros        

    devuelve:
        pd.DataFrame: dataframe con las recomendaciones generadas (títulos y puntuaciones)
"""
def recomendacion_hibrida(user_id: str, books_df: pd.DataFrame, ratings_df: pd.DataFrame,
                         nmf_model_entrenado: Any, tfidf_matrix: np.ndarray, title_to_idx: Dict[str, int],
                         weight_collab: float,top_k: int, confidence_threshold: int, gamma: float,
                         inverse_popularity: np.ndarray) -> pd.DataFrame:
   

    # 1 puntuación colaborativa (nmf)
    scores_collab_full = obtener_puntaje_colaborativo(
        user_id, books_df, nmf_model_entrenado
    )

    # 2 puntuación de contenido (tfidf)
    scores_content_norm, n_ratings = obtener_puntaje_contenido(
        user_id, ratings_df, books_df, tfidf_matrix, title_to_idx
    )

    # 3 pesos ajustados por umbral d confianza
    collab_weight_adjusted, content_weight_eff = calcular_pesos(
        n_ratings, confidence_threshold, weight_collab
    )

    # 4 se crea el raw score del hibrido, mezclando colaborativo + contenido
    raw_score = collab_weight_adjusted * scores_collab_full + content_weight_eff * scores_content_norm

    # 5 final con popularidad inversa
    final_score = (1 - gamma) * raw_score + gamma * inverse_popularity

    # ahora se excluyen los libros ya vistos
    seen_titles = set(ratings_df[ratings_df['user_id'] == user_id]['title'])
    df_scores = pd.DataFrame({'title': books_df['title'], 'final_score': final_score})
    
    df_scores = df_scores[~df_scores['title'].isin(seen_titles)]

    # devolver el top k
    return df_scores.sort_values('final_score', ascending=False).head(top_k).reset_index(drop=True)
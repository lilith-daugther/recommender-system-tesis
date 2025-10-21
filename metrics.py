# verified 
#posibles cambios por paper, checar
#SPLIT RATIONS PER USER aregado, checar otras formas de cross validation
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# -------------------------------------------------------------------
"""
offline evaluation

DIVIDIR  las calidicaciones en conjuntos de entrenamiento y prueba asegurando que el modelo se pruebe con datos que nunca ha visto antes.
esta división se hace por usuario para simular un escenario del mundo real donde queremos predecir las calificaciones de los usuarios en libros que no han calificado previamente.

agrupar las calificaciones por usuario (user_id).
para cada usuario, toma una facción (test_size) de sus calificaciones para el conjunto de prueba (test_sample) y el resto para el conjunto de entrenamiento (train_sample).

esto garantiza que el modelo se entrena cona lgunos libros de un susuario y se prueba con otros libros que ese mismo usuario calificó, 
lo que refleja mejor cómo se utilizará el modelo en la práctica.

"""

def split_ratings_per_user(ratings_df, test_size, random_state=42):
    train_list, test_list = [], []

    for user_id, group in ratings_df.groupby('user_id'):
        if len(group) < 2:
            continue
        test_sample = group.sample(frac=test_size, random_state=random_state)
        train_sample = group.drop(test_sample.index)
        train_list.append(train_sample)
        test_list.append(test_sample)

    if not train_list or not test_list:
        print("DAMN no hay suficientes usuarios/libros después del filtrado.")
        return pd.DataFrame(), pd.DataFrame()

    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    return train, test


# -------------------------------------------------------------------
"""
cuenta los aciertos (hits) en los primeros k ítems recomendados y los divide por k
"""
# Precision@k
def precision_at_k(recommended, relevant, k):
    if len(recommended) == 0:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / k


# -------------------------------------------------------------------
"""
cuenta los aciertos (hits) en los primeros k y los divide por el número total de ítems relevantes que el usuario calificó en la prueba (len(relevant)).
"""
# Recall@k
def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    recommended_at_k = recommended[:k]
    hits = sum(1 for item in recommended_at_k if item in relevant)
    return hits / len(relevant)


# -------------------------------------------------------------------
"""
medir qué tan populares son en promedio, los libros que el modelo recomienda. un valor alto indica un sesgo de popularidad

toma la lista de titulos recomendados 
busca el num de ratings de cada titulo en popularity_counts
calcula el promedio de estos conteos de popularidad y lo devuelve

"""
# Popularidad promedio
def average_popularity(recommended_titles, popularity_counts):
    if not recommended_titles:
        return 0.0
    pops = [popularity_counts.get(t, 0) for t in recommended_titles]
    return np.mean(pops) if pops else 0.0


# -------------------------------------------------------------------
"""
mide que porcentaje de items disponibles (n_items_total) puede ser recomendado al menos una vez en todas las recomendaciones realizadas (all_recommendations).

une las recomednaciones generadas para todos los usuarios en un solo conjunto (unique_recs)
divide el numero de items únicos recomendados por el número total de items disponibles para obtener la cobertura
un coverage bajo significa que tu sistema solo recomienda una pequeña fracción de tu catálogo.

"""
# Cobertura
def coverage(all_recommendations, n_items_total):
    unique_recs = set()
    for rec_list in all_recommendations:
        unique_recs.update(rec_list)
    return len(unique_recs) / n_items_total if n_items_total > 0 else 0.0


# -------------------------------------------------------------------
"""
mide que tan diferentes son tematicamente los items recomendados en una misma lista

toma la lista de titulos recomendadoa (perfiles de contenido)
calcula la similitud coseno entre los pares de esos libros (cosine_similarity)
calcula la similitud promedio (avg_sim) entre todos los pares
la diversidad se define como 1 menos la similitud promedio (1 - avg_sim), si la similitud promedio es alta, la diversidad será baja y viceversa.
"""
# Diversidad
def diversity(recommended_titles, tfidf_matrix, title_to_idx):
    if len(recommended_titles) < 2:
        return 0.0

    indices = [title_to_idx[t] for t in recommended_titles if t in title_to_idx]
    if len(indices) < 2:
        return 0.0

    submatrix = tfidf_matrix[indices]
    sims = cosine_similarity(submatrix)
    upper_tri = sims[np.triu_indices_from(sims, k=1)]
    avg_sim = np.mean(upper_tri) if len(upper_tri) > 0 else 0.0

    return 1 - avg_sim

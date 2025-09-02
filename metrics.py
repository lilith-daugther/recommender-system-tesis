# metrics_documented.py
"""
Módulo de evaluación de sistemas de recomendación.
Define y documenta funciones para las siguientes métricas:

- Precision@K
- Recall@K
- Average Popularity
- Intra-list Diversity
- Item-space Coverage

También incluye una función utilitaria para dividir ratings en train/test.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def split_ratings_per_user(ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Divide el conjunto de ratings en train y test por usuario (leave-one-out style).

    Para cada usuario:
      - Si tiene < 2 ratings, todos van a train.
      - Si tiene >= 2, se separa una fracción `test_size` para test y el resto a train.

    Parámetros:
    - ratings: DataFrame con columnas ['User-ID', 'Title', 'Rating', ...]
    - test_size: fracción de ratings de cada usuario que van a test.
    - random_state: semilla para reproducibilidad.

    Retorna:
    - train: DataFrame de ratings para entrenamiento.
    - test: DataFrame de ratings para evaluación.
    """
    train_list, test_list = [], []
    # Agrupar por usuario
    for user, df_user in ratings.groupby('user_id'):
        if len(df_user) < 2:
            train_list.append(df_user)
        else:
            tn, te = train_test_split(df_user, test_size=test_size, random_state=random_state)
            train_list.append(tn)
            test_list.append(te)
    train = pd.concat(train_list).reset_index(drop=True)
    test = pd.concat(test_list).reset_index(drop=True)
    return train, test


def precision_at_k(recommended: list, ground_truth: set, k: int) -> float:
    """
    Calcula Precision@K:
    Proporción de ítems recomendados en las primeras K posiciones que están en el conjunto real.

    Parámetros:
    - recommended: lista ordenada de ítems recomendados.
    - ground_truth: conjunto de ítems realmente relevantes (test).
    - k: número de ítems a considerar del top-K.

    Retorna:
    - precision: float en [0,1].
    """
    
    topk = recommended[:k]
    hits = len([item for item in topk if item in ground_truth])
    #print('recommended', recommended, '\n ','hits', '\n', 'topk',topk)
    return hits / k


def recall_at_k(recommended: list, ground_truth: set, k: int) -> float:
    """
    Calcula Recall@K:
    Proporción de ítems relevantes que aparecen en las primeras K recomendaciones.

    Parámetros:
    - recommended: lista ordenada de ítems recomendados.
    - ground_truth: conjunto de ítems realmente relevantes (test).
    - k: número de ítems a considerar del top-K.

    Retorna:
    - recall: float en [0,1].
    """
    if not ground_truth:
        return 0.0
    topk = recommended[:k]
    hits = len([item for item in topk if item in ground_truth])
   
    recall = hits / len(ground_truth)
    return recall


def average_popularity(recommended: list, popularity_counts: dict) -> float:
    """
    Mide la popularidad media de una lista de recomendaciones.

    Parámetros usados:
    recommended: lista de ítems recomendados.
    popularity_counts: dict mapping ítem -> número total de ratings.

    Retornamos:
    avg_pop: promedio de counts de los ítems recomendados.

    Valores altos indican tendencia a recomendar ítems muy populares.
    Valores bajos indican recomendaciones menos conocidos.
    """
    if not recommended:
        return 0.0
    pops = [popularity_counts.get(item, 0) for item in recommended]
    return float(np.mean(pops))


def diversity(recommended: list, M: np.ndarray, title_to_idx: dict) -> float:
    """
    Calcula la diversidad intra-lista:
    promedio de similitud coseno entre todos los pares de ítems en la lista recomendada.

    Parámetros usados:
    recommended: lista de ítems recomendados.
    M: matriz TF-IDF de características de todos los libros.
    title_to_idx: dict mapping ítem (Title) -> índice en M.

    Retorna:
    diversity: float en [0,1], donde 1 es máxima diversidad (poca similitud).
    """
    # Obtener índices válidos
    idxs = [title_to_idx[item] for item in recommended if item in title_to_idx]
    K = len(idxs)
    if K < 2:
        return 0.0
    # Submatriz de similitudes
    sims = cosine_similarity(M[idxs], M[idxs])
    
    tssd = sims[np.triu_indices(K, k=1)]
    sim_mean = np.mean(tssd)
    return 1.0 - sim_mean


def coverage(recommended_sets: list, total_items: int) -> float:
    """
    Calcula el coverage:
    proporción de ítems únicos recomendados al menos una vez sobre el total disponible.

    Parámetros:
    recommended_sets: lista de listas de recomendaciones (por usuario).
    total_items: tamaño del catálogo filtrado.

    Retorna:
    coverage_ratio: float en [0,1].
    """
    if total_items <= 0:
        return 0.0
    unique_recs = set(item for recs in recommended_sets for item in recs)
    return len(unique_recs) / total_items


def evaluate_recommender(
    recommender_fn,
    books: pd.DataFrame,
    ratings: pd.DataFrame,
    M: np.ndarray,
    title_to_idx: dict,
    k: int = 5
) -> dict:
    """
    Ejecuta un ciclo de evaluación para un conjunto de usuarios y computa métricas agregadas:
    - Precision@K
    - Recall@K
    - Average Popularity
    - Intra-list Diversity
    - Coverage

    Parámetros:
    - recommender_fn: función(user_id, books, ratings, top_n=k) -> list de Titles.
    - books: DataFrame de libros filtrados.
    - ratings: DataFrame completo de ratings filtrados.
    - M, title_to_idx: para calcular diversidad.
    - k: tamaño de la lista de recomendación.

    Retorna:
    - metrics: dict con valores medios de cada métrica.
    """
    # Dividir train/test
    train, test = split_ratings_per_user(ratings)

    pop_counts = ratings['Title'].value_counts().to_dict()
    precisions, recalls = [], []
    pop_avgs, diversities = [], []
    all_recs = []

    for user, df_test in test.groupby('user_id'):
        ground_truth = set(df_test['title'])
        recs = recommender_fn(user_id=user, books=books, ratings=train, top_n=k)
        all_recs.append(recs)

        precisions.append(precision_at_k(recs, ground_truth, k))
        recalls.append(recall_at_k(recs, ground_truth, k))
        pop_avgs.append(average_popularity(recs, pop_counts))
        diversities.append(diversity(recs, M, title_to_idx))

    metrics = {
        'precision@k': np.mean(precisions),
        'recall@k': np.mean(recalls),
        'avg_popularity': np.mean(pop_avgs),
        'diversity': np.mean(diversities),
        'coverage': coverage(all_recs, len(books))
    }
    return metrics


# Ejemplo de uso:
if __name__ == '__main__':
    from hybrid_recomennder import hybrid_recommendation
    from data_loader import load_books, load_ratings, prepare_data

    # Carga y preparación
    books = load_books()
    ratings = load_ratings(sample_size=5000)
    books_f, ratings_f = prepare_data(books, ratings)

    # Precomputar TF-IDF para diversidad
    from content_recommender import build_tfidf
    vec, M = build_tfidf(books_f)
    title_to_idx = {t:i for i, t in enumerate(books_f['Title'])}

    metrics = evaluate_recommender(
        recommender_fn=lambda user_id, books, ratings, top_n: hybrid_recommendation(
            user_id=user_id,
            books=books_f,
            ratings=ratings_f,
            top_n=top_n
        ),
        books=books_f,
        ratings=ratings_f,
        M=M,
        title_to_idx=title_to_idx,
        k=5
    )
    #print("Métricas de evaluación:", metrics)

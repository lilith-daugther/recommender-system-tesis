import numpy as np
import pandas as pd
from data_loader import load_books, load_ratings, prepare_data, path_ratings, ruta_books
from metrics import (
    split_ratings_per_user,
    precision_at_k, recall_at_k,
    average_popularity, diversity, coverage
)
from hybrid_recomennder import recomendacion_hibrida, obtener_valor_aware
from content_recommender import build_tfidf

def main():
    # 1) Cargar datos
    books = load_books(ruta_books)
    ratings = load_ratings(path_ratings, sample_size=3000)

    # Filtrar por mínimos
    books_f, ratings_f = prepare_data(
        books, ratings,
        min_ratings_per_book=3,
        min_ratings_per_user=2
    )

    # TF-IDF para diversidad (ok construirlo sobre books_f)
    tfidf_vectorizer, tfidf_matrix = build_tfidf(
        books_f, max_features=2000, min_df=2
    )
    title_to_idx = {t: i for i, t in enumerate(books_f['title'])}

    # 2) Split por usuario
    train, test = split_ratings_per_user(ratings_f)

    # Elegir usuario del TEST
    test_user = test['user_id'].sample(1).iloc[0]
    print(f"Usuario de prueba: {test_user}")

    # Conjunto de verdad (items en TEST del usuario)
    real_recommendations = set(
        test.loc[test['user_id'] == test_user, 'title']
    )

    # --- IMPORTANTE ---
    # Calcular popularidad e INVERSA sobre TRAIN (no usar test para entrenar ni para "vistos")
    inverse_popularity = obtener_valor_aware(ratings_f, books_f)
    print(f"Popularidad inversa normalizada: min={inverse_popularity.min()}, max={inverse_popularity.max()}")

    k = 10

    # ENTRENAR Y RECOMENDAR usando SOLO TRAIN
    recomendador_hibrido = recomendacion_hibrida(
        user_id=test_user,
        books_df=books_f,
        ratings_df=train,               # <-- AQUÍ EL CAMBIO CLAVE
        n_components=10,
        max_iterations=500,
        tfidf_max_features=2000,
        tfidf_min_df=2,
        weight_collab=0.6,
        inverse_popularity=inverse_popularity,
        top_k=k,
        confidence_threshold=5,
        gamma=0.3
    )

    # --- Normalizador uniforme para títulos ---
    def norm_title(s):
        return str(s).strip().casefold()

    # Títulos recomendados (normalizados)
    recommended_titles = [norm_title(t) for t in recomendador_hibrido['title'].tolist()]

    # Ground truth del usuario (normalizado)
    real_recommendations_norm = set(norm_title(t) for t in real_recommendations)

    # Popularidad: construir counts con claves normalizadas
    popularity_counts = (
        ratings_f['title'].astype(str).str.strip().str.casefold().value_counts().to_dict()
    )

    # Evaluar métricas
    avg_popularity = average_popularity(recommended_titles, popularity_counts)
    print(f"Popularidad: {avg_popularity:.4f}")

    coverage_test = coverage([recommended_titles], len(books_f))
    print(f"Coverage: {coverage_test:.4f}")

    precision_test = precision_at_k(recommended_titles, real_recommendations_norm, k)
    recall_test = recall_at_k(recommended_titles, real_recommendations_norm, k)
    print(f"Precision@{k}: {precision_test:.4f}")
    print(f"Recall@{k}: {recall_test:.4f}")

    diversity_test = diversity(recommended_titles, tfidf_matrix, title_to_idx)
    print(f"Diversidad: {diversity_test:.4f}")


if __name__ == "__main__":
    main()

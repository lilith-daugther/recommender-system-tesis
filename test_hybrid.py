# test_hybrid_recommender_documented.py
"""
Script de prueba para el sistema de recomendación híbrida (NMF + contenido).

Pasos:
1. Cargar un subconjunto de ratings para no saturar memoria.
2. Filtrar libros y usuarios poco activos.
3. Ejecutar la función `hybrid_recommendation` y mostrar resultados.
"""

from data_loader import load_books, load_ratings, prepare_data
from hybrid_recomennder import hybrid_recommendation


def main():
    # 1) Cargar datos
    books = load_books()
    sample_size = 25000
    ratings = load_ratings(sample_size=sample_size)
    print(f"Ratings cargados: {len(ratings)} filas")

    # 2) Filtrar libros y usuarios poco activos
    books_f, ratings_f = prepare_data(
        books, ratings,
        min_ratings_per_book=5,
        min_ratings_per_user=1
    )
    print(f"Datos filtrados: {len(books_f)} libros, {len(ratings_f)} ratings.")

    # 3) Seleccionar un usuario de prueba al azar
    test_user = ratings_f['User-ID'].sample(n=1, random_state=42).iloc[0]
    user_name = ratings_f.loc[
    ratings_f['User-ID'] == test_user, 
        'UserName'
    ].iloc[0]

    n_components = 5       # Factores latentes en NMF
    max_iter = 200         # Iteraciones de NMF
    max_features = 2000    # Vocabulario TF-IDF
    min_df = 2             # Frecuencia mínima TF-IDF
    alpha = 0.6            # Peso de componente colaborativa, # 0.6 significa 60% NMF, 40% contenido
    top_n = 10              # Número de recomendaciones

    recs = hybrid_recommendation(
        user_id=test_user,
        books=books_f,
        ratings=ratings_f,
        n_components=n_components,
        max_iter=max_iter,
        max_features=max_features,
        min_df=min_df,
        alpha=alpha,
        top_n=top_n
    )

    # 5) Mostrar resultados
    print(f"top  {top_n} recomendaciones híbridas para usuario {user_name}:")
    print(recs)

if __name__ == '__main__':
    main()

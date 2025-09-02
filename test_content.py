# test de basado en contenido
from data_loader import load_books, load_ratings, prepare_data
from content_recommender import build_tfidf, recommend_content

def main():
    # 1) Cargar datos
    # Cargamos todos los libros y un subconjunto de N ratings para pruebas.
    books = load_books()
    sample_size = 20000  # Número de ratings a cargar para no saturar memoria
    ratings = load_ratings(sample_size=sample_size)

    # 2) Filtrar datos
    # Eliminamos usuarios y libros con pocas interacciones (< thresholds).
    books_f, ratings_f = prepare_data(books, ratings)
    print(f"{len(books_f)} libros y {len(ratings_f)} ratings después del filtrado")

    # 3) Vectorizar texto
    # Creamos el TF-IDF usando solo los libros filtrados,
    # limitando el vocabulario y documentos mínimos para cada término.
    max_features = 2000  # Máximo número de términos
    min_df = 2           # Mínimo documentos por término
    vec, M = build_tfidf(books_f, max_features=max_features, min_df=min_df)

    # 4) Probar recomendación
    # Seleccionamos un título de ejemplo y pedimos las top_n recomendaciones.
    ejemplo = books_f['Title'].sample(n=1, random_state=42).iloc[0]
    print("Libro de prueba:", ejemplo)

    top_n = 5      # Cantidad de recomendaciones a mostrar
    thresh = 0.2   # Umbral de similitud
    recomendaciones = recommend_content(
        ejemplo, 
        books_f, 
        vec, M, 
        top_n=top_n, 
        thresh=thresh
    )
     # El 'Score' indica la similitud de coseno: un valor entre 0 y 1. 
     # Valores más altos significan mayor similitud textual.
    print("Recomendaciones basadas en contenido:")
    print(recomendaciones)

if __name__ == "__main__":
    main()
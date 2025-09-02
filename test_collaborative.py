# test_collaborative.py

from data_loader import load_books, load_ratings, prepare_data
from nmf_recommender import create_sparse_matrix, train_nmf, recommend_nmf

def main():
    # 1) Cargar datos con números limitados de ratings
    books = load_books()
    ratings = load_ratings(sample_size=10000)
    
    # 2) Filtrar libros y usuarios poco activos OJO
    books_f, ratings_f = prepare_data(books, ratings)
    print(f"Datos filtrados: {len(books_f)} libros, {len(ratings_f)} ratings.")
    
    # 3) Construir matriz dispersa
    mat = create_sparse_matrix(ratings_f)
    
    # 4) Entrenar NMF
    n_components = 10
    nmf_model, W, H = train_nmf(mat, n_components=n_components, max_iter=500)
    #print(f" se esta entrenado con {n_components} componentes latentes.")
    
    # 5) Seleccionar un usuario de prueba al azar
    test_user = ratings_f['User-ID'].sample(n=1, random_state=42).iloc[0]
    user_name = ratings_f.loc[
    ratings_f['User-ID'] == test_user, 
        'UserName'
    ].iloc[0]
    
    # 6) Generar recomendaciones
    recs = recommend_nmf(
        user_id=test_user,
        W=W,
        H=H,
        idx_to_book=mat.idx_to_book,
        user_to_idx=mat.user_to_idx,    
        ratings_df=ratings_f,
        top_n=5
    )
    
    print(f"top  de recomendaciones para {user_name}:")
    print(recs)

if __name__ == "__main__":
    main()

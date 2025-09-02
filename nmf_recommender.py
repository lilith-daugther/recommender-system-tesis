import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from data_loader import load_books, load_ratings, prepare_data

"""
Módulo de recomendación colaborativa usando Non-negative Matrix Factorization (NMF).

Funciones:
- create_sparse_matrix: convierte ratings en matriz dispersa libros×usuarios con mapeos.
- train_nmf: entrena el modelo NMF y devuelve factorización.
- recommend_nmf: genera recomendaciones para un usuario dado.
https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9
"""


#-----------------------------------------------------------------------------------------------------
"""
    construir una matriz dispersa CSR de tamaño n (libros x #usuarios).
    parámetros:
    ratings: DataFrame con columnas 'User-ID', 'Title', 'Rating'.
    
    devuelve: csr_matrix libros por usuarios con mapeos de títulos y usuarios.
    """
def crear_matriz_dispersa(ratings):
    
    # identificar valores únicos
    books = ratings['title'].unique()
    users = ratings['user_id'].unique()

    # crear mapeos directos e inversos
    """
    se contruyen cuatro diccionarios.
    recorre la lista books con enumerate, que a cada elemento b le asigna un índice entero i
    """
    book_to_idx = {b: i for i, b in enumerate(books)}
    user_to_idx = {u: i for i, u in enumerate(users)}

    #se invierte el diccionario para obtener los índices a los títulos y usuarios
    idx_to_book = {i: b for b, i in book_to_idx.items()}
    idx_to_user = {i: u for u, i in user_to_idx.items()}

    # mapear filas (libros), columnas (usuarios) y valores
    rows = ratings['title'].map(book_to_idx)
    cols = ratings['user_id'].map(user_to_idx)
    vals = ratings['rating'].values.astype(np.float32)

    # crear matriz dispersa
    matriz_dispersa = csr_matrix((vals, (rows, cols)), shape=(len(books), len(users)))

    matriz_dispersa.book_to_idx = book_to_idx # asignar mapeo de libros
    matriz_dispersa.user_to_idx = user_to_idx # asignar mapeo de usuarios
    matriz_dispersa.idx_to_book = idx_to_book # asignar mapeo inverso de libros
    matriz_dispersa.idx_to_user = idx_to_user # asignar mapeo inverso de usuarios

    return matriz_dispersa #retorna la matriz dispersa con los mapeos


#-----------------------------------------------------------------------------------------------------
"""
    entrenar el modelo NMF sobre la matriz dispersa.
    parámetros:
    matriz_dispersa: matriz dispersa de calificaciones usuario-libro
    n_components: número de factores latentes.
    max_iter: iteraciones máximas para convergencia.
   devolvemos: modelo NMF entrenado, matrices W (libros por factores) y H (factores por usuarios).
    """
def entrenar_nmf(matriz_dispersa, n_components, max_iter):
    
    nmf_model = NMF(
        n_components=n_components, 
        init='nndsvda',
        max_iter=max_iter,
        random_state=42
    )
    matriz_w = nmf_model.fit_transform(matriz_dispersa) #se entrena el modelo NMF y se obtiene la matriz W (libros por factores)
    matriz_h = nmf_model.components_ #se obtiene la matriz H (factores por usuarios)
    return nmf_model, matriz_w, matriz_h 


#-----------------------------------------------------------------------------------------------------
    """
    genera recomendaciones para un usuario basado en la factorization NMF.

    parámetros:
    user_id: id del usuario para el cual se generan recomendaciones.
    matriz_w: matriz W (libros por factores).
    matriz_h: matriz H (factores por usuarios).
    idx_to_book: mapeo de índices a títulos de libros.
    user_to_idx: mapeo de usuarios a índices.
    ratings_df: DataFrame con las valoraciones de los usuarios.
    top_n: número de recomendaciones a devolver.
    devuelve:DataFrame con columnas 'Title' y 'Score', ordenado por Score descendente
    """
def recommend_nmf(user_id, matriz_w, matriz_h, idx_to_book, user_to_idx, ratings_df, top_n=5):
   
    # Verificar usuario
    if user_id not in user_to_idx:
        raise KeyError(f"Usuario {user_id} no encontrado")
    user_idx = user_to_idx[user_id]

    # Calcular puntuaciones: W (libros×factores) · H[:, user_idx]
    scores = matriz_w.dot(matriz_h[:, user_idx])

    # Construir DataFrame con todas las puntuaciones
    df_scores = pd.DataFrame({
        'title': [idx_to_book[i] for i in range(len(scores))],
        'Score': scores
    })

    # Filtrar libros ya valorados por el usuario
    seen = set(ratings_df[ratings_df['user_id'] == user_id]['title'])
    df_scores = df_scores[~df_scores['title'].isin(seen)]

    # Ordenar y devolver top_n
    return df_scores.sort_values('Score', ascending=False).head(top_n)

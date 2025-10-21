# vderified
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix
from typing import Dict, Any, List, Tuple

"""
clase nfm descompone la matriz de califiaciones: usuario -item en dos matrices mas pequeñas que representan los perfiles o factores latentes de los 
usuarios (w) y de los items (h).
    Referencias:
    https://www.youtube.com/watch?v=sUo8hCxXo-k
    https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9

"""
class NMF_recommender:

    """
        INICIALIZA el recomendador con parámetros para el entrenamiento.

        n_components: número de características latentes (el 'ancho' del perfil).
        max_iter: número máximo de iteraciones para el algoritmo NMF.
        
        """
    
    def __init__(self, n_components: int = None, max_iter: int = None):

        
        #parámetros del modelo
        self.n_components = n_components
        self.max_iter = max_iter
    
        self.model: Any = None # motor de factorización NMF

        # matriz w: el 'perfil del usuario' (cuánto le gusta cada tema oculto a cada usuario).
        self.W: np.ndarray = None
        # matriz h: el 'perfil del libro' (cuánto de cada tema oculto tiene cada libro).
        self.H: np.ndarray = None
        
        # mapeos de id reales a índices numéricos para la matriz dispersa
        self.book_to_idx: Dict[str, int] = {}
        self.user_to_idx: Dict[str, int] = {}
        """
        self.book_to_idx - traducir el título real de un libro a la posición de su fila en la matriz
        sef.user_to_idx - traducir el ID real de un usuario a la posición de su columna en la matriz
        
        self.idx_to_book - traducir el índice de fila de la matriz de vuelta al título real del libro
        self.idx_to_user - traducir el índice de columna de la matriz de vuelta al ID
        """
        self.idx_to_book: Dict[int, str] = {}
        self.idx_to_user: Dict[int, str] = {}
        
        # datos de entrenamiento
        self.ratings_df: pd.DataFrame = None


    # ------------------------------------------------------------------------------------
    """
    ENTRENA el modelo nmf. este proceso construye la matriz de interacción dispersa  y la factoriza para obtener los perfiles w y h.

        parametros:
            df_ratings: DataFrame de calificaciones con columnas 'user_id', 'title', 'rating'.

    """
    def entrenar_modelo(self, df_ratings: pd.DataFrame):
        
        # guardar datos de entrenamiento
        self.ratings_df = df_ratings 

        # preparar mapeos
        # creamos listas de todos los ítems y usuarios ÚNICOS para establecer las dimensiones de la matriz, definimos el tamaño y orden de las filas y columnas.
        titulos_unicos = df_ratings['title'].unique()
        usuarios_unicos = df_ratings['user_id'].unique()

        # comprensiones de diccionario, creamos diccionarios para la traducción id - índice

        """
        genera dos puntuaciones por separado y las combina mediante un promedio ponderado

        self.book_to_idx (título del libro -> índice)

            enumerate(titulos_unicos) toma la lista de títulos únicos y crea pares 
            (índice, título), donde 'índice' es la posición del título en la lista y 'título' es el valor real del título.
            luego, la comprensión del diccionario itera sobre estos pares y asigna cada título (b) a su índice correspondiente (i) en el diccionario self.book_to_idx.
            
            de resultado da algo como:
            { 'libro_a': 0, 'libro_b': 1, 'libro_c': 2 }.

        self.user_to_idx (id del usuario -> índice)
        
            enumerate(usuarios_unicos) crea los pares (índice, id_usuario), y la comprensión del diccionario asigna cada id_usuario (u) a su índice (i) en self.user_to_idx.
            el resultado es algo como:
            { 'user_1': 0, 'user_2': 1, 'user_3': 2 }.

        """
        self.book_to_idx = {
            b: i for i, b in enumerate(titulos_unicos)}
        self.user_to_idx = {
            u: i for i, u in enumerate(usuarios_unicos)}
        
        """
        mapeos inversos (índice numérico -> nombre real)
            self.idx_to_book (índice -> título del libro)
            self.book_to_idx.items(): toma los pares clave-valor del diccionario directo,la comprensión del diccionario invierte estos pares, asignando cada índice (i) al título correspondiente (b).
            el resultado es algo como:
            { 0: 'libro_a', 1: 'libro_b', 2: 'libro_c' }.

            self.idx_to_user (índice -> id del usuario)
            invierte el diccionario self.user_to_idx para que puedas obtener el id real del usuario a partir del índice de la matriz.
        """

        self.idx_to_book = {
            i: b for b, i in self.book_to_idx.items()}
        self.idx_to_user = {
            i: u for u, i in self.user_to_idx.items()}

        # construir matriz dispersa (csr)
        # convertimos los IDs del DataFrame a los índices numéricos de la matriz.
        indices_fila = df_ratings['title'].map(self.book_to_idx)
        indices_columna = df_ratings['user_id'].map(self.user_to_idx)
        valores_rating = df_ratings['rating'].values.astype(np.float32)

        # la matriz de interacción: libros como filas, usuarios como columnas (item x usuario).
        matriz_dispersa = csr_matrix( (valores_rating, (indices_fila, indices_columna)), shape=(len(titulos_unicos), len(usuarios_unicos)))

        # ajustar NMF si el número de factores es mayor que las dimensiones de la matriz.
        max_factores = min(matriz_dispersa.shape) - 1
        n_components_ajustado = min(self.n_components, max_factores)

        if n_components_ajustado < self.n_components:
            self.n_components = n_components_ajustado

        # entrenar el modelo
        self.model = NMF(
            n_components=self.n_components,
            init='nndsvda',
            max_iter=self.max_iter,
            random_state=42
        )

        # fit_transform: encuentra el perfil del usuario (w) y lo devuelve.
        self.W = self.model.fit_transform(matriz_dispersa) 
        # components_: el modelo calcula el perfil del libro (h) y lo almacena aquí.
        self.H = self.model.components_

        #print(f"entrenamiento NMF completado. error de Reconstrucción: {self.model.reconstruction_err_:.4f}")
    

    # ------------------------------------------------------------------------------------
    """
    GENERA una lista de libros recomendados para un usuario específico utilizando la multiplicación de matrices W y H.

        parametros:

        user_id: ID del usuario para el cual generar recomendaciones.
        top_n: Número de recomendaciones a devolver.
        normalize: Si se debe normalizar la puntuación de 0 a 1 (opcional).

        devuelve:
            
            dataFrame con las columnas 'title' y 'score' (puntuación predicha).
        """
    def recomendaciones(self, user_id: str, top_n: int = 0, normalize: bool = True) -> pd.DataFrame:
        
        if user_id not in self.user_to_idx:
            raise KeyError(f"Error: Usuario '{user_id}' no encontrado en los datos de entrenamiento.")

        # indice del usuario en la matriz
        user_idx = self.user_to_idx[user_id]

        # 1. reconstruir la Puntuación
        # predicción: multiplicamos el perfil del libro (W) por el perfil latente del usuario (H[:, user_idx]).
        # el resultado son las calificaciones predichas para TODOS los libros por ESE usuario.
        scores = self.W.dot(self.H[:, user_idx])

        # 2. filtrar libros vistos
        # creamos una lista de los libros que el usuario YA calificó.
        libros_vistos = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['title'])
        
        recomendaciones_lista: List[Dict[str, float]] = []
        for book_idx, score in enumerate(scores):
            book_title = self.idx_to_book[book_idx]
            if book_title not in libros_vistos:
                recomendaciones_lista.append({'title': book_title, 'score': score})

        df_recs = pd.DataFrame(recomendaciones_lista)

        # 3. normalizacion de puntuaciones
        if normalize and not df_recs.empty:

            min_v, max_v = df_recs['score'].min(), df_recs['score'].max()

            if max_v > min_v:
                # escalamiento de Min-Max para que las puntuaciones sean fáciles de interpretar (0 a 1).
                df_recs['score'] = (df_recs['score'] - min_v) / (max_v - min_v)

        # 4. rrdenar y devolver el top_n
        return df_recs.sort_values('score', ascending=False).head(top_n).reset_index(drop=True)


"""
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from scipy.sparse import csr_matrix


Módulo de recomendación colaborativa usando Non-negative Matrix Factorization (NMF).

Funciones:
- create_sparse_matrix: convierte ratings en matriz dispersa libros×usuarios con mapeos.
- train_nmf: entrena el modelo NMF y devuelve factorización.
- recommend_nmf: genera recomendaciones para un usuario dado.
https://medium.com/@quindaly/step-by-step-nmf-example-in-python-9974e38dc9f9



class NMFRecommender:
    def __init__(self, n_components=20, max_iter=200):
        self.n_components = n_components
        self.max_iter = max_iter
        self.model = None
        self.W = None
        self.H = None
        # Mapeos
        self.book_to_idx, self.user_to_idx = {}, {}
        self.idx_to_book, self.idx_to_user = {}, {}
        self.ratings_df = None

    def fit(self, ratings_df):
        Entrena el modelo NMF con un DataFrame de ratings (user_id, title, rating).
        print("Entrenando el recomendador NMF...")
        self.ratings_df = ratings_df

        # 1. Crear matriz dispersa y mapeos
        unique_books = ratings_df['title'].unique()
        unique_users = ratings_df['user_id'].unique()

        self.book_to_idx = {b: i for i, b in enumerate(unique_books)}
        self.user_to_idx = {u: i for i, u in enumerate(unique_users)}
        self.idx_to_book = {i: b for b, i in self.book_to_idx.items()}
        self.idx_to_user = {i: u for u, i in self.user_to_idx.items()}

        rows = ratings_df['title'].map(self.book_to_idx)
        cols = ratings_df['user_id'].map(self.user_to_idx)
        vals = ratings_df['rating'].values.astype(np.float32)

        sparse_matrix = csr_matrix((vals, (rows, cols)), shape=(len(unique_books), len(unique_users)))

        # 2. Ajustar n_components si es demasiado grande
        max_factors = min(sparse_matrix.shape) - 1
        n_components = min(self.n_components, max_factors)
        if n_components < self.n_components:
            print(f"⚠️ Ajustando n_components de {self.n_components} → {n_components}")
            self.n_components = n_components

        # 3. Entrenar el modelo NMF
        self.model = NMF(
            n_components=self.n_components,
            init='nndsvda',
            max_iter=self.max_iter,
            random_state=42
        )
        self.W = self.model.fit_transform(sparse_matrix)
        self.H = self.model.components_

        print(f"Entrenamiento NMF completado. Reconstruction error: {self.model.reconstruction_err_:.4f}")

    def recommend(self, user_id, top_n=10, normalize=True):
        Genera recomendaciones para un usuario específico.
        if user_id not in self.user_to_idx:
            raise KeyError(f"Usuario {user_id} no encontrado.")

        user_idx = self.user_to_idx[user_id]

        # 1. Reconstruir puntuaciones para ese usuario
        scores = self.W.dot(self.H[:, user_idx])

        # 2. Filtrar libros ya vistos
        seen_books = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['title'])
        recommendations = []
        for book_idx, score in enumerate(scores):
            book_title = self.idx_to_book[book_idx]
            if book_title not in seen_books:
                recommendations.append({'title': book_title, 'score': score})

        df_recs = pd.DataFrame(recommendations)

        # 3. Normalizar si se requiere
        if normalize and not df_recs.empty:
            min_v, max_v = df_recs['score'].min(), df_recs['score'].max()
            if max_v > min_v:
                df_recs['score'] = (df_recs['score'] - min_v) / (max_v - min_v)

        # 4. Ordenar y devolver top_n
        return df_recs.sort_values('score', ascending=False).head(top_n)
"""

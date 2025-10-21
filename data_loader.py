# verified
from typing import Tuple
import pandas as pd
import re
from scipy.sparse import csr_matrix #eficiencia en matrices grandes

path_books_cvs = 'data/archive/books_data.csv'
path_ratings_cvs = 'data/archive/books_rating.csv'

# ------------------------------------------------------------------------------------
"""
LIMPIA Y ESTANDARIZA los títulos de los libros para asegurar coincidencias consistentes haciendo coincidir con minusculas, 
eliminando caracteres especiales y espacios innecesarios.
    parametros:
        title: El título del libro a limpiar (string).
        
    devuelve:
        title: título limpio y estandarizado.
"""
def normalize_title(title: str) -> str:

    if not isinstance(title, str):
        return ""
    title = title.lower()

    # conservams solo letras, números, espacios y signos básicos de puntuación.
    title = re.sub(r"[^a-záéíóúñ0-9\s:'-]", "", title)

    # reemplazamos múltiples espacios con uno solo y eliminamos espacios al inicio/final.
    title = re.sub(r"\s+", " ", title).strip()

    return title


# ------------------------------------------------------------------------------------
"""
CARGAR el csv de metadatos de LIBROS, realiza la limpiezza inicial y estandariza nombres.
    parametros:
        path_books: ruta al archivo csv de libros.

    devuelve:
        dataframe de pandas con la información de los libros normalizada.
"""

def load_books(path_books: str = path_books_cvs) -> pd.DataFrame:
    #cargamos solo las columnas necesarias para el tdf-idf
    try:
        df_libros_cargados = pd.read_csv(
            path_books,
            sep=',',
            usecols=['Title', 'description', 'publishedDate', 'categories'],
            low_memory=False
        )

        # renombrar columnas para consistencia y facilidad de uso
        df_libros_cargados = df_libros_cargados.rename(columns={
            'Title': 'title', 'description': 'description',
            'publishedDate': 'published_date', 'categories': 'categories'
        })
        
        # aplicar la limpieza y normalización de títulos 
        df_libros_cargados['title'] = df_libros_cargados['title'].apply(normalize_title)

        # eliminar libros sin título después de normalizar
        df_libros_cargados = df_libros_cargados.dropna(subset=['title']) 
        df_libros_cargados = df_libros_cargados[df_libros_cargados['title'] != ""]

        # extraer el año de publicación si es posible 
        if 'published_date' in df_libros_cargados.columns:
            # error coerce convierte fechas inválidas en NaT
            df_libros_cargados['Year'] = pd.to_datetime(df_libros_cargados['published_date'], errors='coerce').dt.year
        
        return df_libros_cargados
    # manejo de errores de lectura, debugging
    except Exception as e:
        print(f"error al cargar el archivo {path_books}: {e}")
        return pd.DataFrame()


#------------------------------------------------------------------------------------------
"""
CARGA el csv de RATINGS y devuelve un DataFrame con las columnas, limpia datos numericos y normaliza títulos.
    ********************************************************columnas del csv original***************************************************
    'Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness','review/score', 'review/time', 'review/summary', 'review/text'

    parametros:
        path_ratings: ruta al archivo csv de ratings.
        sample_size: si se proporciona, carga solo una muestra aleatoria de este tamaño.
        random_state: semilla para reproducibilidad al muestrear.
    devuelve:
        dataframe de pandas con la información de ratings limpia y normalizada.
"""

def load_ratings(path_ratings: str = path_ratings_cvs, sample_size: int = None, random_state: int = None) -> pd.DataFrame:
    
    try:
        df_ratings_cargados = pd.read_csv(path_ratings, usecols=['Id', 'Title', 'review/score', 'User_id'])

    except Exception as e:
        print(f"Error al cargar ratings: {e}")

        return pd.DataFrame()

    # estandarizamos los nombres de las columnas para facilitar su uso
    df_ratings_cargados = df_ratings_cargados.rename(columns={
        'Id': 'id', 'Title': 'title', 'review/score': 'rating', 'User_id': 'user_id'
    })
    
    # normalizar títulos
    df_ratings_cargados['title'] = df_ratings_cargados['title'].apply(normalize_title)
    
    # limpieza de columna de calificaciones 
    df_ratings_cargados['rating'] = pd.to_numeric(df_ratings_cargados['rating'], errors='coerce')

    # eliminar filas con datos faltantes o inválidos
    df_ratings_cargados.dropna(subset=['rating', 'user_id', 'title'], inplace=True)
    df_ratings_cargados = df_ratings_cargados[df_ratings_cargados['title'] != ""]

    # aplicar muestreo si se especifica
    if sample_size is not None and sample_size < len(df_ratings_cargados):
        df_ratings_cargados = df_ratings_cargados.sample(n=sample_size, random_state=random_state)
    
    return df_ratings_cargados.reset_index(drop=True)



#------------------------------------------------------------------------------------------
"""
PREPARAR los datos aplicando filtrado iterativo para asegurar que cada libro y usuario cumpla con los requisitos mínimos de actividad, 
limpiando los datos para el motor de recomendación, este proceso se repite hasta que ya no se eliminen más filas (convergencia).
    parametros:
        df_libros: DataFrame inicial de metadatos de libros. ('books_df')
        df_ratings: DataFrame inicial de calificaciones. ('ratings_df')
        min_ratings_per_book: mínimo de calificaciones que debe tener un libro.
        min_ratings_per_user: mínimo de calificaciones que debe tener un usuario.
    
    deveulve:
        una tupla con (DataFrame de libros filtrados, DataFrame de ratings filtrados).
"""
def prepare_data(df_libros: pd.DataFrame, df_ratings: pd.DataFrame, min_ratings_per_book: int, min_ratings_per_user: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    
    df_actual = df_ratings.copy()

    while True:
        # conteo de actividad:
        book_counts = df_actual['title'].value_counts()
        libros_a_mantener = book_counts[book_counts >= min_ratings_per_book].index
        
        user_counts = df_actual['user_id'].value_counts()
        usuarios_a_mantener = user_counts[user_counts >= min_ratings_per_user].index
        
        tamanho_antes = len(df_actual)

        # aplicación del filtro:
        df_filtrado = df_actual[df_actual['title'].isin(libros_a_mantener)]
        df_filtrado = df_filtrado[df_filtrado['user_id'].isin(usuarios_a_mantener)]

        tamanho_despues = len(df_filtrado)

        if tamanho_antes == tamanho_despues:
            # parada: si la limpieza converge (no hay cambios), terminamos.
           
            break
        else:
            # continuar: Los filtros se impactan mutuamente, repetimos la limpieza.
            print(f"iteración completada. filas eliminadas: {tamanho_antes - tamanho_despues}. repitiendo... debugging")
            df_actual = df_filtrado.copy()
            
    # ajuste Final: sincronizamos la tabla de libros con las calificaciones restantes.
    libros_activos_finales = df_filtrado['title'].unique()
    
    # nos quedamos con los libros que SÍ tienen ratings y eliminamos duplicados por título.
    df_libros_final = df_libros[df_libros['title'].isin(libros_activos_finales)].drop_duplicates(subset=['title']).copy()

    # debugging

    print(f"\ntamaño de muestra: {len(df_filtrado)}.")
    print(f"libros finales: {len(df_libros_final)}")
    print(f"usuarios finales: {df_filtrado['user_id'].nunique()}")

    return df_libros_final.reset_index(drop=True), df_filtrado.reset_index(drop=True)

# ------------------------------------------------------------------------------------------
"""
convierte el DataFrame de calificaciones filtrado en una Matriz Dispersa (CSR).
esto es esencial para los algoritmos de recomendación colaborativa, ya que  ahorra memoria al almacenar solo las calificaciones (ignorando los ceros).
    
    parametros:
        df_ratings_filtrados: DataFrame de calificaciones después de la limpieza y el filtrado.
    
    devuelve:
        una tupla con (Matriz CSR, mapeo_usuario, mapeo_libro).
    """
def create_sparse_matrix(df_ratings_filtrados: pd.DataFrame) -> Tuple[csr_matrix, dict, dict]:
    
    # crear mapeos (diccionarios) para convertir IDs a índices de matriz.
    mapeo_usuario = {id_: i for i, id_ in enumerate(df_ratings_filtrados['user_id'].unique())}
    mapeo_libro = {title: i for i, title in enumerate(df_ratings_filtrados['title'].unique())}

    # 2. aplicar los mapeos al DataFrame.
    idx_usuario = df_ratings_filtrados['user_id'].apply(lambda x: mapeo_usuario[x]).values
    idx_libro = df_ratings_filtrados['title'].apply(lambda x: mapeo_libro[x]).values
    ratings = df_ratings_filtrados['rating'].values

    # 3. construir la Matriz Dispersa (CSR).
    # 'data' son los valores (ratings), 'row' son los índices de usuario, 'col' son los índices de libro.
    matriz_interaccion = csr_matrix((ratings, (idx_usuario, idx_libro)), shape=(len(mapeo_usuario), len(mapeo_libro)))
                                    
    print(f"matriz de Interacción CSR creada con {matriz_interaccion.nnz} elementos no nulos... debugging")

    return matriz_interaccion, mapeo_usuario, mapeo_libro
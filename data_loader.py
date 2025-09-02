# data_loader.py
import pandas as pd
from scipy.sparse import csr_matrix

ruta_books = 'data/archive/books_data.csv'
path_ratings = 'data/archive/books_rating.csv'

"""
    carga el csv de libros y devuelve un DataFrame con las columnas: 
    'Title', 'description', 'authors', 'image', 'previewLink', 'publisher', 
    'publishedDate', 'infoLink', 'categories', 'ratingsCount'
    ajusta los tipos de datos y renombra las columnas.
"""
def load_books(path_books = ruta_books):
    try:
        df = pd.read_csv(
            path_books, 
            sep=',',
            usecols=[
                'Title', 
                'description', 
                'publishedDate', 
                'categories'
            ],
            low_memory=False
        )
        
        # Renombrar columnas
        df = df.rename(columns={
            'Title': 'title',
            'description': 'description',
            'publishedDate': 'published_date',
            'categories': 'categories'
        })
        
        # Debug: revisar columnas
        print("Columnas después de rename:", df.columns.tolist())
        
        # Crear columna Year si existe published_date
        if 'published_date' in df.columns:
            df['Year'] = df['published_date'].astype(str).str[:4]
        else:
            print("⚠️ La columna 'published_date' no se encontró en el DataFrame.")
        
        return df
    
    except Exception as e:
        print(f"Error al cargar el archivo {path_books}: {e}")
        return pd.DataFrame()


#------------------------------------------------------------------------------------------
"""
    cargar el csv de ratings y devuelve un DataFrame con las columnas:
    'Id', 'Title', 'Price', 'User_id', 'profileName', 'review/helpfulness', 
    'review/score', 'review/time', 'review/summary', 'review/text'

    y según el sample_size toma una muestra aleatoria de ese tamaño.
"""
def load_ratings(path_ratings, sample_size=None, random_state=42) -> pd.DataFrame:
    try: 
        df = pd.read_csv(
            path_ratings, sep=',',
            encoding='utf-8',
            usecols=[
                'Id', 'Title', 'review/score', 'profileName', 'User_id',
            ],
            low_memory=False
        )
    except UnicodeDecodeError:
        df = pd.read_csv(
            path_ratings, sep=',',
            encoding='latin1',
            usecols=[
                'Id', 'Title', 'review/score', 'profileName', 'User_id',
            ],
            low_memory=False
        )
    
    # renombrar columnas
    df = df.rename(
        columns={
            'Id': 'id',
            'Title': 'title',
            'review/score': 'rating',
            'profileName': 'profile_name',
            'User_id': 'user_id'
        }
    )

    # convertir a tipo adecuado, RATING NUMÉRICO 
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])  # eliminar filas con rating NaN

    # muestreo de tamaño sample_size que no sea none
    if sample_size is not None and sample_size > 0 and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
    return df

#------------------------------------------------------------------------------------------
def prepare_data(books_df, ratings_df, min_ratings_per_book, min_ratings_per_user):
    """"
    filtra los libros y los ratings según los parámetros de mínimo de ratings por libro y usuario.
    parametros:
    books_df: DataFrame de libros.
    ratings_df: DataFrame de ratings.
    min_ratings_per_book: mínimo de ratings requeridas por libro.
    min_ratings_per_user: mínimo de ratings requeridas por usuario.
    devuelve:
    un DataFrame de libros filtrados y un DataFrame de ratings filtrados.
    """
    #print("Columnas en ratings_df:", ratings_df.columns)

    book_counts = ratings_df['title'].value_counts()
    user_counts = ratings_df['user_id'].value_counts()
    # Filtrar libros y usuarios si se especifica un mínimo
    if min_ratings_per_book is not None:
        keep_books = book_counts[book_counts >= min_ratings_per_book].index
    else:
        keep_books = book_counts.index  # Mantener todos los libros

    if min_ratings_per_user is not None:
        keep_users = user_counts[user_counts >= min_ratings_per_user].index
    else:
        keep_users = user_counts.index  # Mantener todos los usuarios
    # Filtrar calificaciones
    r_f = ratings_df[
        ratings_df['title'].isin(keep_books) &
        ratings_df['user_id'].isin(keep_users)
    ].copy()
    # Filtrar libros que queden
    b_f = books_df[books_df['title'].isin(r_f['title'])].drop_duplicates(subset=['title']).copy()
    return b_f.reset_index(drop=True), r_f.reset_index(drop=True)
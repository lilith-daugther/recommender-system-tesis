# verified
import numpy as np
import pandas as pd

"""
calcula el vector de pesos inversos de popularidad usando una función logarítmica, 
para favorecer la diversidad (nichos).

la fórmula aplicada es: w_i = 1 / log(pop_i + 1).

    parametros:
        ratings_df: DataFrame con las valoraciones de usuarios (debe incluir 'title').
        books_df: DataFrame con los libros (debe incluir 'title').

    devuelve:
        np.ndarray con valores normalizados en [0,1] para cada libro en books_df
"""

def calcular_item_weights_log(ratings_df: pd.DataFrame, books_df: pd.DataFrame) -> np.ndarray:


    # conteo de popularidad (número total de ratings por libro)
    pop_counts = ratings_df['title'].value_counts()

    # vector de popularidad alineado con books_df, aseguramos que la popularidad de cada libro esté en el mismo orden
    pop_arr = np.array([pop_counts.get(t, 0) for t in books_df['title']], dtype=float)

    # aplicación de la fórmula del "castigo logarítmico"
    # epsik¡lson añade para evitar errores log(0) si un libro no tiene ratings
    epsilon = 1e-6

    # la division por el lofaritmo reduce el peso de los items muy populares
    w = 1.0 / np.log(pop_arr + 1 + epsilon)

    # normalización min-max
    # escalamos el vector resultante a [0, 1] para que sea combinable en el híbrido.
    wmin, wmax = w.min(), w.max()

    # si todos los pesos son iguales (caso raro), devolvemos un vector de ceros
    if wmax - wmin < 1e-8:
        return np.zeros_like(w)

    return (w - wmin) / (wmax - wmin)

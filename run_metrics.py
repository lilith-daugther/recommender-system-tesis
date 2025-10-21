#verified

import pandas as pd
import numpy as np
from collections import defaultdict # para agrupar métricas
from typing import Dict, Any, List

from data_loader import load_books, load_ratings, prepare_data, path_books_cvs, path_ratings_cvs
from content_recommender import build_tfidf
from inverse_popularity import calcular_item_weights_log

from nmf_recommender import NMF_recommender
from hybrid_recomennder import recomendacion_hibrida 

from metrics import (
    split_ratings_per_user,
    precision_at_k,
    recall_at_k,
    coverage,
    diversity,
    average_popularity,
)

"""
funcion para normalizar strings (titulos) para comparaciones consistentes 
"""
# ------------------------------------------------------------------------------------------
def norm_title(s):
    return str(s).strip().casefold()

# ------------------------------------------------------------------------------------------
"""
    realiza la evaluación de  folds (simulados por seed) entrenando el modelo una vez por fold.
"""
def evaluar_modelo_general(ratings_f: pd.DataFrame, books_f: pd.DataFrame, config: Dict[str, Any], num_folds: int) -> Dict[str, float]:
   
    num_folds = 5
    print("\n  evaluación general con ", num_folds, " folds ---")

    # preparación de componentes fijos (válidos para todos los folds)
    
    # tF-IDF (para diversidad y contenido), se entrena una sola vez, fuera del bucle de folds.
    tfidf_vec, tfidf_matrix = build_tfidf(books_f, max_features=config['tfidf_max_features'], min_df=config['tfidf_min_df'])
    title_to_idx = {t: i for i, t in enumerate(books_f['title'])}
    
    if tfidf_matrix is None:
        print("no se pudo construir la matriz tfidf")
        return {}
    
    resultados_por_fold = defaultdict(list)

    for i in range(num_folds):
        random_seed = 42 + i # simular fold camiando la seed de aleatoriedad
        print(f"...ejecutando fold {i+1})")

        # split (train/test) con seed diferente
        train, test = split_ratings_per_user(ratings_f, test_size=config['test_size'], random_state=random_seed)
        
        if train.empty or test.empty:
            print(f"no hay suficientes datos después de la división.")
            continue
            
        # entrenamiento nmf una sola vez por fold pa no gastar tanta memoria
        # se entrena un nuevo modelo NMF para el set de entrenamiento de este fold
        nmf_model = NMF_recommender(n_components=config['n_components'], max_iter=config['max_iterations'])
        nmf_model.entrenar_modelo(train)

        # popularidad inversa una sola vez por fold, basada en el set de entrenamiento
        inverse_popularity = calcular_item_weights_log(train, books_f)
        popularity_counts = train['title'].astype(str).str.strip().str.casefold().value_counts().to_dict()
        
        # iteración y recopilación de métricas
        usuarios_a_probar = test['user_id'].unique()
        metricas_fold = []
        all_recommended_titles = [] 
        
        print(f"... evaluando {len(usuarios_a_probar)} usuarios de prueba.")

        for user_id in usuarios_a_probar:
            real_recs = set(test.loc[test['user_id'] == user_id, 'title'])
            if not real_recs: continue
                
            # generar recomendaciones (usando el nmf entrenado y tfidf)
            recommended_df = recomendacion_hibrida(
                user_id=user_id,
                books_df=books_f,
                ratings_df=train, # datos de entrenamiento
                nmf_model_entrenado=nmf_model, #modelo nmf pre-entrenado
                tfidf_matrix=tfidf_matrix,
                title_to_idx=title_to_idx,
                weight_collab=config['weight_collab'],
                top_k=config['top_k'],
                confidence_threshold=config['confidence_threshold'],
                gamma=config['gamma'],
                inverse_popularity=inverse_popularity
            )

            if recommended_df.empty: continue

            # cálculo de métricas por usuario
            recommended_titles = [norm_title(t) for t in recommended_df['title'].tolist()]
            real_recs_norm = set(norm_title(t) for t in real_recs)
            
            all_recommended_titles.append(recommended_titles)
            
            # recopilar resultados (precision y recall son métricas por usuario)
            metricas_fold.append({
                'precision': precision_at_k(recommended_titles, real_recs_norm, k=config['top_k']),
                'recall': recall_at_k(recommended_titles, real_recs_norm, k=config['top_k']),
                'diversity': diversity(recommended_titles, tfidf_matrix, title_to_idx),
                'popularity': average_popularity(recommended_titles, popularity_counts)
            })

        # promedio de las métricas de usuario en este fold
        df_metricas = pd.DataFrame(metricas_fold)

        if not df_metricas.empty:
            resultados_por_fold['precision'].append(df_metricas['precision'].mean())
            resultados_por_fold['recall'].append(df_metricas['recall'].mean())
            resultados_por_fold['diversity'].append(df_metricas['diversity'].mean())
            resultados_por_fold['popularity'].append(df_metricas['popularity'].mean())
            resultados_por_fold['coverage'].append(coverage(all_recommended_titles, len(books_f)))
        
    # promedio final
    resultados_finales = {
        f'precision_media_@{config["top_k"]}': np.mean(resultados_por_fold['precision']),
        f'recall_medio_@{config["top_k"]}': np.mean(resultados_por_fold['recall']),
        'diversidad_media': np.mean(resultados_por_fold['diversity']),
        'popularidad_media': np.mean(resultados_por_fold['popularity']),
        'coverage_media': np.mean(resultados_por_fold['coverage'])
    }
    
    return resultados_finales

# ------------------------------------------------------------------------------------------
def main():

    sample_size = 10000 #trngo miedo 
    min_ratings_per_book = 1
    min_ratings_per_user = 1
    test_size = 0.2
    num_folds = 5 

    n_components = 80
    max_iterations = 800
    tfidf_max_features = 2000
    tfidf_min_df = 2
    weight_collab = 0.6
    confidence_threshold = 5
    top_k = 10
    gamma = 0.4


    books = load_books(path_books_cvs)
    ratings = load_ratings(path_ratings_cvs, sample_size=sample_size)

    books_f, ratings_f = prepare_data(
        books, ratings,
        min_ratings_per_book=min_ratings_per_book,
        min_ratings_per_user=min_ratings_per_user
    )
    if books_f.empty or ratings_f.empty:
        print("no hay datos después del filtrado.")
        return

    config = {
        'sample_size': sample_size, 'min_ratings_per_book': min_ratings_per_book,
        'min_ratings_per_user': min_ratings_per_user, 'test_size': test_size,
        'n_components': n_components, 'max_iterations': max_iterations,
        'tfidf_max_features': tfidf_max_features, 'tfidf_min_df': tfidf_min_df,
        'weight_collab': weight_collab, 'confidence_threshold': confidence_threshold,
        'top_k': top_k, 'gamma': gamma
    }

    
    # EVALUACIÓN (CROSS-VALIDATION SIMULADA)
   
    resultados = evaluar_modelo_general(ratings_f, books_f, config, num_folds=num_folds)

   
    if resultados:

        print(f" reultados finales en promedio en({num_folds} folds)")
        print("-----------------------------------------------------------------")

        print(f"precision{config['top_k']}: {resultados[f'precision_media_@{config["top_k"]}']:.4f}")
        print(f"recall {config['top_k']}: {resultados[f'recall_medio_@{config["top_k"]}']:.4f}")
        print(f"diversidad: {resultados['diversidad_media']:.4f}")
        print(f"popularidad : {resultados['popularidad_media']:.2f}")
        print(f"coverage: {resultados['coverage_media']:.4f}")

if __name__ == "__main__":
    main()
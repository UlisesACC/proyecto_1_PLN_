from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
import numpy as np
import pandas as pd

# Separacion en entrenamiento y prueba
categories = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.misc']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                      remove=('headers', 'footers', 'quotes'),
                                      random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                     remove=('headers', 'footers', 'quotes'),
                                     random_state=42)
X_train = newsgroups_train.data[:1200]
y_train = newsgroups_train.target[:1200]
X_test = newsgroups_test.data
y_test = newsgroups_test.target
#stop_words='english'
# Pipeline: las etiquetas y_train NO se utilizan para aprender los clusters.
pipe = Pipeline([('text_representation', TfidfVectorizer(stop_words='english')),
                 ('dimensionality_reduction', TruncatedSVD(100, random_state=42)),
                 ('normalization', Normalizer()),
                 ('clustering', KMeans(n_clusters=4, n_init=10, random_state=42))])
print(pipe)

# Entrenamiento no supervisado
pipe.fit(X_train)
y_pred = pipe.predict(X_train)

# Evaluacion de los clusters de entrenamiento
X_vectors = pipe[:-1].transform(X_train)
print('\nK-Means: entrenamiento')
print('Documentos por cluster:', np.bincount(y_pred, minlength=4))
print('Silhouette (coseno):', round(silhouette_score(X_vectors, y_pred, metric='cosine',
                                                    sample_size=min(500, len(y_pred)),
                                                    random_state=42), 3))
print('ARI (solo evaluacion externa):', round(adjusted_rand_score(y_train, y_pred), 3))
print('NMI (solo evaluacion externa):', round(normalized_mutual_info_score(y_train, y_pred), 3))
print(pd.crosstab(pd.Categorical.from_codes(y_train, newsgroups_train.target_names),
                  y_pred, rownames=['Categoria original'], colnames=['Cluster']))

# Palabras representativas: TF-IDF promedio de los documentos de cada cluster.
# Se usa el espacio TF-IDF original, antes de SVD, para identificar palabras.
X_tfidf = pipe.named_steps['text_representation'].transform(X_train)
terms = pipe.named_steps['text_representation'].get_feature_names_out()
print('\nPalabras relevantes por cluster (entrenamiento):')
for cluster_id in range(4):
    indices = np.where(y_pred == cluster_id)[0]
    if len(indices) == 0:
        continue
    mean_tfidf = np.asarray(X_tfidf[indices].mean(axis=0)).ravel()
    top = np.argsort(mean_tfidf)[-10:][::-1]
    print('  Cluster', cluster_id, ':', ', '.join(terms[top]))

# Asignacion de documentos nuevos a los clusters aprendidos.
# El pipeline aplica el mismo TF-IDF, SVD y Normalizer ajustados con X_train.
y_test_pred = pipe.predict(X_test)

print('\nK-Means: prueba')
print('Documentos por cluster:', np.bincount(y_test_pred, minlength=4))
print('ARI (solo evaluacion externa):', round(adjusted_rand_score(y_test, y_test_pred), 3))
print('NMI (solo evaluacion externa):', round(normalized_mutual_info_score(y_test, y_test_pred), 3))
print(pd.crosstab(pd.Categorical.from_codes(y_test, newsgroups_test.target_names),
                  y_test_pred, rownames=['Categoria original'], colnames=['Cluster']))

# Palabras representativas de los documentos de prueba asignados a cada cluster.
# Usamos el vectorizador ajustado SOLO con entrenamiento (transform, no fit_transform).
X_test_tfidf = pipe.named_steps['text_representation'].transform(X_test)
print('\nPalabras relevantes por cluster (prueba):')
for cluster_id in range(4):
    indices = np.where(y_test_pred == cluster_id)[0]
    if len(indices) == 0:
        continue
    mean_tfidf = np.asarray(X_test_tfidf[indices].mean(axis=0)).ravel()
    top = np.argsort(mean_tfidf)[-10:][::-1]
    print('  Cluster', cluster_id, ':', ', '.join(terms[top]))


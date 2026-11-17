from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np

# Separacion en entrenamiento y prueba
categories = ['comp.graphics', 'rec.sport.baseball', 'sci.space', 'talk.politics.misc']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories,
                                      remove=('headers', 'footers', 'quotes'),
                                      )
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories,
                                     remove=('headers', 'footers', 'quotes'),
                                     )
X_train = newsgroups_train.data[:1200]
y_train = newsgroups_train.target[:1200]
X_test = newsgroups_test.data
y_test = newsgroups_test.target

# Pipeline: LDA aprende temas sin utilizar y_train.
# LDA utiliza conteos de palabras, no TF-IDF ni SVD.
#stop_words="english", 
pipe = Pipeline([('text_representation', CountVectorizer(stop_words='english', min_df=3, max_df=0.85, max_features=5000,)),
                 ('topic_modeling', LatentDirichletAllocation(
                     n_components=4, random_state=42,
                     learning_method='batch', max_iter=15))])
print(pipe)

# Entrenamiento no supervisado
pipe.fit(X_train)

# Palabras representativas por tema (entrenamiento)
terms = pipe.named_steps['text_representation'].get_feature_names_out()
lda = pipe.named_steps['topic_modeling']
print('\nLDA: temas descubiertos en entrenamiento')
for topic_id, weights in enumerate(lda.components_):
    top = np.argsort(weights)[-10:][::-1]
    print('  Tema', topic_id, ':', ', '.join(terms[top]))

# Evaluacion: perplejidad sobre los mismos conteos y vocabulario
# que aprendio el vectorizador durante el entrenamiento.
X_train_counts = pipe.named_steps['text_representation'].transform(X_train)
X_test_counts = pipe.named_steps['text_representation'].transform(X_test)
print('\nLDA: evaluacion')
print('Perplejidad (entrenamiento):', round(lda.perplexity(X_train_counts), 2))
print('Perplejidad (prueba):', round(lda.perplexity(X_test_counts), 2))

# Distribucion de temas para los documentos de prueba.
y_test_pred = pipe.transform(X_test)
print('\nLDA: documentos de prueba')
for i in range(3):
    print('\nDocumento', i + 1)
    print('Texto:', ' '.join(X_test[i].split())[:250], '...')
    print('Categoria original (solo referencia):', newsgroups_test.target_names[y_test[i]])
    for topic_id, proportion in enumerate(y_test_pred[i]):
        print('  Tema', topic_id, ':', f'{proportion:.1%}')

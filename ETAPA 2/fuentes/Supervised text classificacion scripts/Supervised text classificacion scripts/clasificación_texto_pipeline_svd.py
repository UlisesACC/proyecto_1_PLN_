from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

#Separación en entrenamiento y prueba
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
y_train = newsgroups_train.target
X_test = newsgroups_test.data
y_test = newsgroups_test.target

pipe = Pipeline([('text_representation', TfidfVectorizer()), ('dimensionality_reduction', TruncatedSVD(1000)), ('classifier', LogisticRegression())])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print (classification_report(y_test, y_pred))

pipe.set_params(dimensionality_reduction__n_components=300)
print (pipe)
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print (classification_report(y_test, y_pred))s








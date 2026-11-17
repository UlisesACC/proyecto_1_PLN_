from sklearn.datasets import fetch_20newsgroups
from pprint import pprint
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

#Carga y separación del corpus
newsgroups_train = fetch_20newsgroups(subset='train')
print (newsgroups_train.filenames.shape)
newsgroups_test = fetch_20newsgroups(subset='test')
print (newsgroups_test.filenames.shape)
print(newsgroups_train.target_names)
print(newsgroups_train.data[0])
print(newsgroups_train.target[0])

X_train = newsgroups_train.data
y_train = newsgroups_train.target
print (y_train)
X_test = newsgroups_test.data
y_test = newsgroups_test.target
print (y_test)

# ~ #Representación del texto
vectorizer = TfidfVectorizer()
vectors_train = vectorizer.fit_transform(X_train)
print (vectorizer.get_feature_names_out())
print (len(vectorizer.get_feature_names_out()))
print (vectors_train.toarray())

# ~ #Clasificadores
clf = MultinomialNB()
clf.fit(vectors_train, y_train)

vectors_test = vectorizer.transform(newsgroups_test.data)
print (vectors_test.toarray())
print (vectorizer.get_feature_names_out())

y_pred = clf.predict(vectors_test)
print ('*****************Naïve Bayes****************')
print (y_pred)


print (classification_report(y_test, y_pred))


clf = LogisticRegression()
clf.fit(vectors_train, y_train)
vectors_test = vectorizer.transform(newsgroups_test.data)
y_pred = clf.predict(vectors_test)
print ('*****************Logistic Regression****************')
print (y_pred)
print (classification_report(y_test, y_pred))

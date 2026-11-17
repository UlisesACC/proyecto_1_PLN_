from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os, pickle
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_validate

class data_set_polarity:
	def __init__(self, X_train, y_train, X_test, y_test):
		self.X_train = X_train
		self.y_train = y_train
		self.X_test = X_test
		self.y_test = y_test

def generate_train_test(file_name):
	pd.options.display.max_colwidth = 200				

	#Lee el corpus original del archivo de entrada y lo pasa a una DataFrame
	df = pd.read_excel(file_name)
	X = df['Opinion'].astype(str)
	y_polarity = df['Polarity'].values
	
	#~ #Separa el corpus cargado en el DataFrame en el 80% para entrenamiento y el 20% para pruebas
	X_train, X_test, y_train_polarity, y_test_polarity = train_test_split(X, y_polarity, test_size=0.2, random_state=0)
	
	return (data_set_polarity(X_train, y_train_polarity, X_test, y_test_polarity))
	
if __name__=='__main__':

	if not (os.path.exists('corpus_polarity.pkl')):
		corpus_polarity = generate_train_test('Rest_Mex_2022.xlsx')
		corpus_file = open ('corpus_polarity.pkl','wb')
		pickle.dump(corpus_polarity, corpus_file)
	else:
		corpus_file = open ('corpus_polarity.pkl','rb')
		corpus_polarity = pickle.load(corpus_file)
	corpus_file.close()
	print (corpus_polarity.X_train)
	print (corpus_polarity.X_test)

	# Representación vectorial binarizada
	vectorizador_binario = CountVectorizer(binary=True)
	X_train = vectorizador_binario.fit_transform(corpus_polarity.X_train)
	print (vectorizador_binario.get_feature_names_out())
	print (len(vectorizador_binario.get_feature_names_out()))
	
	y_train_polarity = corpus_polarity.y_train
	clf_polarity = LogisticRegression(max_iter=10000)
	clf_polarity.fit(X_train, y_train_polarity)
	y_test = corpus_polarity.y_test
	X_test = vectorizador_binario.transform (corpus_polarity.X_test)
	y_pred = clf_polarity.predict(X_test)
	print(accuracy_score(y_test, y_pred))
	print(confusion_matrix(y_test, y_pred,labels=[1,2,3,4,5]))
	target_names = ['1','2','3','4','5']
	print(classification_report(y_test, y_pred, target_names=target_names))

	# ~ print ('Realizando validación cruzada...')
	# ~ cv_results = cross_validate(clf_polarity, X_train, y_train_polarity, cv=5, scoring='f1_macro', verbose = 3, n_jobs = -1)
	# ~ print (cv_results)

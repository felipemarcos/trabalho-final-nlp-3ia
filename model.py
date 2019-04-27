import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from bayesiansets import BayesianSets

# TODO
def test_movies():
	df = pd.read_csv('datasets/movies.csv', index_col=0)

	vect = TfidfVectorizer()

	X = vect.fit_transform(df['title'])

	query = [
		'toy story',
		'the lion king',
		'alladin',
		'beauty and the best',
		'cinderella',
		'little mermaid',
		'hercules'
	]

	model = BayesianSets(X)
	ranking = np.argsort(model.compute_scores(query))[::-1]
	top10 = ranking[:10]

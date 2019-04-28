import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from bayesiansets import BayesianSets
from treattext import TreatText

# Load Movies from CSV
df = pd.read_csv('datasets/movies.csv', index_col=0)
df['title'] = df['title'].apply(TreatText().run)

# Find similar movies
query = np.array([
	'toy story',
	'the lion king',
	'alladin',
	'beauty and the best',
	'cinderella',
	'little mermaid',
	'hercules'
])

query_ids = df.loc[df['title'].isin(query)].index.tolist()
query_ids = np.array(query_ids)

X = TfidfVectorizer().fit_transform(df['title'])

model = BayesianSets(X)

# ranking = np.argsort(model.compute_scores(query_ids))[::-1]
# top10 = ranking[:10]


import numpy as np
from scipy.sparse import csr_matrix

class BayesianSets:
	def __init__(self, dataset, c=2):
		self.dataset = csr_matrix(dataset)

		m = np.mean(self.dataset, 0)

		self.alpha = c * m
		self.beta = c * (1 - m)

		self.logAlpha = np.log(self.alpha)
		self.logBeta = np.log(self.beta)
		self.logAlphaBeta = np.log(self.alpha + self.beta)

	def compute_parameters(self, query_indices):
		N = len(query_indices)
		sum_x = np.sum(self.dataset[query_indices], axis=0)

		alphaTilde = self.alpha + sum_x
		betaTilde = self.beta + N - sum_x
		logAlphaTilde = np.log(alphaTilde)
		logBetaTilde = np.log(betaTilde)
		logAlphaBetaN = np.log(self.alpha + self.beta + N)

		query = logAlphaTilde - self.logAlpha - logBetaTilde + self.logBeta
		constant = np.sum(self.logAlphaBeta - logAlphaBetaN + logBetaTilde - self.logBeta)

		return query, constant

	def compute_scores(self, query_indices):
		q, nc = self.compute_parameters(query_indices)
		Xq = self.dataset * q.transpose()

		score = nc + Xq

		return np.asarray(score).flatten()

import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	bias = np.ones((X.shape[0], 1))
	X = np.concatenate([bias, X], axis=1)
	n, m = X.shape
	y = y[:,None]
	
	theta = np.zeros((m,1))
	def sigma(x):
		return 1/(1+np.exp(-x))
	losses = []
	for _ in range(iterations):
		z = X @ theta
		pred = sigma(z)
		loss = -np.sum(y*np.log(pred)+(1-y)*np.log(1-pred))
		losses.append(np.round(loss, 4))
		grad = X.T @ (pred-y)
		theta = theta - learning_rate*grad
	theta = list(np.round(theta, 4).flatten())

	return (theta, losses)


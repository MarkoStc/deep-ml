import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    # Your code here
    m, n = X.shape

    if method == 'batch':
        batch_size = m 
    elif method == 'stochastic':
        batch_size = 1
    k = m // batch_size
    for _ in range(n_epochs):  
        i = 0  
        for i in range(k):
            indices = np.arange(i*batch_size, (i+1)*batch_size)
            X_batch, y_batch = X[indices], y[indices]
            grad = 2/batch_size*X_batch.T @ (X_batch @ weights - y_batch)
            weights = weights - learning_rate*grad

    return weights             










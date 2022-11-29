import numpy as np

def hinge_loss(w, X, y, C):
    y_hat = y * np.dot(X,w)
    y_hat = np.maximum(np.zeros_like(y_hat), (1-y_hat)) # hinge function
    
    return C*sum(y_hat)

def logistic_loss(w, X, y, return_arr=None):
    y_hat = y * np.dot(X,w)
    # Logistic loss is the negative of the log of the logistic function.
    if return_arr == True:
        out = -(log_logistic(y_hat))
    else:
        out = -np.sum(log_logistic(y_hat))
    return out

def log_logistic(X):
    if X.ndim > 1: raise Exception("Array of samples cannot be more than 1-D!")
    out = np.empty_like(X) # same dimensions and data types

    idx = X>0
    out[idx] = -np.log(1.0 + np.exp(-X[idx]))
    out[~idx] = X[~idx] - np.log(1.0 + np.exp(X[~idx]))
    return out
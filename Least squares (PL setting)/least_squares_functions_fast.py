
import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

def least_squares_loss(w, X, y, la):
    assert la >= 0
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    n_0 = X.shape[0]
    return (1/n_0) * np.sum((X.dot(w) - y)**2) + la * regularizer(w)

def least_squares_grad(w, X, y, la):
    """
    Returns full gradient
    :param w:
    :param X:
    :param y:
    :param la:
    :return:
    """
    assert la >= 0
    assert (y.shape[0] == X.shape[0])
    assert (w.shape[0] == X.shape[1])

    loss_grad = X.transpose() @ (X.dot(w) - y)

    assert len(loss_grad) == len(w)
    n_0 = X.shape[0]
    return (1/n_0) * loss_grad + la * regularizer_grad(w)

def regularizer_grad(w):
    return 2*w

def regularizer(w: np.ndarray):
    return np.sum(w**2)

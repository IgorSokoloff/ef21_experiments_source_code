"""
Auxiliary functions for logistic regression problem.
"""

import numpy as np
import random
import time

from numpy.linalg import norm
from scipy.sparse import csr_matrix
from scipy.optimize import minimize

def logreg_loss(w, X, y, la):
    assert la >= 0
    assert len(y) == X.shape[0]
    assert len(w) == X.shape[1]
    l = np.log(1 + np.exp(-X.dot(w) * y))
    m = y.shape[0]
    return np.mean(l) + la * regularizer(w)

def logreg_grad(w, X, y, la):
    """
    Returns full gradient
    :param w:
    :param X:
    :param y:
    :param la:
    :return:
    """
    assert la >= 0
    #print (f"w.shape[0]: {w.shape[0]}; X.shape[1]: {X.shape[1]}")
    assert (y.shape[0] == X.shape[0])
    assert (w.shape[0] == X.shape[1])
    
    numerator = np.multiply(X, y[:, np.newaxis])
    denominator = 1 + np.exp(np.multiply(X@w,y))

    loss_grad = - np.mean(numerator/denominator[:,np.newaxis], axis = 0)
    
    #loss_grad = - np.sum(np.multiply(X, y[:, np.newaxis]), axis = 0) / (1 + np.exp(y_i * np.dot(x_i, w)))
    
    assert len(loss_grad) == len(w)
    return loss_grad + la * regularizer_grad(w)

def logreg_part_grad(w, X, y, la, ids):
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
    
    loss_part_grad = np.zeros(w.shape)
    
    numerator = np.multiply(X[:,ids], y[:, np.newaxis])
    denominator = 1 + np.exp(np.multiply(X@w, y))

    matrix = numerator/denominator[:,np.newaxis]
    
    loss_part_grad[ids] = - np.mean(matrix, axis = 0)
    
    assert len(loss_part_grad) == len(w)
    return loss_part_grad + la * regularizer_part_grad(w,ids)

def regularizer_grad(w):
    return 2*w /(1 + w**2)**2

def regularizer_part_grad(w, ids):
    reg_part_grad = np.zeros(w.shape)
    reg_part_grad[ids] = 2*w[ids] /(1 + w[ids]**2)**2
    
    return reg_part_grad

def regularizer(w: np.ndarray):
    return np.sum(w**2/(1 + w**2))
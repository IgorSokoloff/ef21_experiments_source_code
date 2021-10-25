"""
A script for the data preprocessing before launching an algorithm
It takes the given dataset and outcomes the partition
"""

import numpy as np
from sklearn.model_selection import train_test_split
import time
import sys
import os
import argparse
from numpy.random import normal, uniform
from sklearn.datasets import make_spd_matrix, make_sparse_spd_matrix, load_svmlight_file, dump_svmlight_file
from numpy.linalg import norm
import itertools
from scipy.special import binom
from scipy.stats import ortho_group
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import pandas as pd
from matplotlib import pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
#from logreg_functions import *

parser = argparse.ArgumentParser(description='Generate data and provide information about it for workers and parameter server')

parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms', help='The name of the dataset')
parser.add_argument('--loss_func', action='store', dest='loss_func', type=str, default="log-reg",
                    help='loss function ')
parser.add_argument('--num_starts', action='store', dest='num_starts', type=int, default=1, help='Number of starts from different points')
parser.add_argument('--shared_start', action='store', dest='shared_start', type=int, default=1, help='Whether or not each launch experiments from differnt point') # if shared_start == 1 then each launch starts from the same point
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')


args = parser.parse_args()

dataset = args.dataset
num_starts = args.num_starts
loss_func = args.loss_func
shared_start = args.shared_start
num_workers = args.num_workers

#debug section

"""
num_workers = 20
dataset = 'a9a'
loss_func = 'log-reg' 
num_starts = 1

"""

if num_starts is None:
    raise ValueError("num_starts has to be specified")

if loss_func is None:
    raise ValueError("loss_func has to be specified")
        
assert (num_starts > 0)

def nan_check (lst):
    """
    Check whether has any item of list np.nan elements
    :param lst: list of datafiles (eg. numpy.ndarray)
    :return:
    """
    for i, item in enumerate (lst):
        if np.sum(np.isnan(item)) > 0:
            raise ValueError("nan files in item {0}".format(i))


data_name = dataset + ".txt"

user_dir = os.path.expanduser('~/')
RAW_DATA_PATH = os.getcwd() +'/data/'

project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

#TODO: assert these values below

train_d = None

enc_labels = np.nan
data_dense = np.nan
    
#if not (os.path.isfile(data_path + 'X.npy') and os.path.isfile(data_path + 'y.npy')):
if os.path.isfile(RAW_DATA_PATH + data_name):
    data, labels = load_svmlight_file(RAW_DATA_PATH + data_name)
    enc_labels = labels.copy()
    data_dense = data.todense()
    if not np.array_equal(np.unique(labels), np.array([-1, 1], dtype='float')):
        min_label = min(np.unique(enc_labels))
        max_label = max(np.unique(enc_labels))
        enc_labels[enc_labels == min_label] = -1
        enc_labels[enc_labels == max_label] = 1
    print (enc_labels.shape, enc_labels[-5:])
else:
    raise ValueError("cannot load " + data_name + ".txt")

assert (type(data_dense) == np.matrix or type(data_dense) == np.ndarray)
assert (type(enc_labels) == np.ndarray)

if np.sum(np.isnan(enc_labels)) > 0:
    raise ValueError("nan values of labels")

if np.sum(np.isnan(data_dense)) > 0:
    raise ValueError("nan values in data matrix")

print ("Data shape: ", data_dense.shape)

X_0 = data_dense
y_0 = enc_labels
assert len(X_0.shape) == 2
assert len(y_0.shape) == 1
data_len = enc_labels.shape[0]
train_d = X_0.shape[0]
nan_check([X_0,y_0])
np.save(data_path + 'X', X_0)
np.save(data_path + 'y', y_0)

#partition of data for each worker

num_samles_per_worker = data_len//num_workers

X = []
y = []

for i in range(num_workers):
    X.append(X_0[num_samles_per_worker*i:num_samles_per_worker*(i+1)])
    y.append(y_0[num_samles_per_worker*i:num_samles_per_worker*(i+1)])
    
#print ("shape y:", y_0.shape)
#raise ValueError("!")

X[-1] = np.vstack((X[-1], X_0[num_samles_per_worker*(num_workers):]))
y[-1] = np.hstack((y[-1], y_0[num_samles_per_worker*(num_workers):]))

nan_check(y)
nan_check(X)

for i in range (len(X)):
    print (f"worker {i} has {X[i].shape[0]} datasamples; class 1: {X[i][np.where(y[i] == 1)].shape[0]}; class -1: {X[i][np.where(y[i]==-1)].shape[0]}")
    np.save(data_path + 'X_{0}_nw{1}_{2}'.format(dataset, num_workers, i), X[i])
    np.save(data_path + 'y_{0}_nw{1}_{2}'.format(dataset, num_workers, i), y[i].flatten())
    
if not os.path.isfile(data_path + 'data_info.npy'):
    # if data_info is already esxist we pass this branch
    la = 0.1 #regularization parameter
    data_info = [la]
    np.save(data_path + 'data_info', data_info)

if train_d is None:
    X = np.load(data_path + 'X.npy')
    train_d = X.shape[1]

if shared_start:
    x_0 = np.random.normal(loc=0.0, scale=1.0, size=train_d)
    for i in range (num_starts):
        np.save(data_path + 'w_init_{0}_{1}.npy'.format(i, loss_func), x_0)
else:
    for i in range (num_starts):
        # create a new w_0
        x_0 = np.random.normal(loc=10.0, scale=1.0, size=train_d)
        np.save(data_path + 'w_init_{0}_{1}.npy'.format(i, loss_func), x_0)








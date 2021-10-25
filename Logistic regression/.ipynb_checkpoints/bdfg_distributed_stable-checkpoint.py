"""
experiment for logistic regression function with non-convex regularizer
EF21
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
import datetime
from IPython import display
from logreg_functions_fast import *

#np.random.seed(23)
def stopping_criterion(sq_norm, eps, it, Nsteps):
    #return (R_k > eps * R_0) and (it <= Nsteps)
    return (it <= Nsteps) and (sq_norm >=eps)

def top_k_matrix (X,k):
    output = np.zeros(X.shape)
    for i in range (X.shape[0]):
        output[i] = top_k_compressor(X[i],k)
    return output
        
def top_k_compressor(x, k):
    output = np.zeros(x.shape)
    x_abs = np.abs(x)
    idx = np.argpartition(x_abs, -k)[-k:]  # Indices not sorted
    inds = idx[np.argsort(x_abs[idx])][::-1]
    output[inds] = x[inds]
    return output

def compute_full_grads (A, x, b, la,n_workers):
    grad_ar = np.zeros((n_workers, x.shape[0]))
    for i in range(n_workers):
        grad_ar[i] = logreg_grad(x, A[i], b[i], la).copy()
    return grad_ar

def biased_diana_top_k_gd_estimator(A, x, b, la, k, g_ar, n_workers):
    grads = compute_full_grads(A, x, b, la, n_workers)
    assert(grads.shape==(n_workers,x.shape[0]))
    g_ar_new = np.zeros((n_workers, x.shape[0]))
    delta = grads - g_ar
    g_ar_new = g_ar + top_k_matrix(delta, k)
    size_value_sent = sys.getsizeof(delta[0,0])
    size_value_sent = 32
    #print ("size_value_sent ", size_value_sent)
    return g_ar_new, size_value_sent, np.mean(grads, axis=0)

def biased_diana_top_k_gd(x_0, A, b, A_0, b_0, stepsize, eps,la,k, n_workers, Nsteps=100000):
    print(f"biased_diana_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers) 
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    PRINT_EVERY = 1000
    
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize*g 
        g_ar, size_value_sent, grad = biased_diana_top_k_gd_estimator(A, x, b, la, k, g_ar, n_workers)
        g = np.mean(g_ar, axis=0)
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        it += 1
        it_ar.append(it*k*size_value_sent)
        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x

def save_data(its, f_grad_norms, x_solution, k_size, experiment_name, project_path, dataset):
    experiment = '{0}_{1}'.format(experiment_name, k_size)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)
    
    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")
    
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))
    


parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')
parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=int, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='mushrooms',help='Dataset name for saving logs')

args = parser.parse_args()
nsteps = args.max_it
k_size = args.k
num_workers = args.num_workers
dataset = args.dataset
loss_func = "log-reg"

n_ar = np.array([num_workers])
k_ar = np.array([k_size])
factor = args.factor
eps = args.tol
la = 0.1

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

data_path = project_path + "data_{0}/".format(dataset)

if not os.path.exists(data_path):
    os.mkdir(data_path)

#data_info = np.load(data_path + 'data_info.npy')

X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')

n_0, d_0 = X_0.shape  

#print(f"n_0: {n_0}; d_0{d_0}")

hess_f_0 = (1 / (4*n_0)) * (X_0.T @ X_0) + 2*la*np.eye(d_0)
L_0 = np.max(np.linalg.eigvals(hess_f_0))
L_0 = L_0.astype(np.float)

for i in range(len(n_ar)):
    #c = subprocess.call(f"python3 generate_data.py --dataset mushrooms --num_starts 1 --num_workers {n_ar[i]} --loss_func log-reg --is_homogeneous 0", shell=True)
    X = []
    y = []    
    L = np.zeros(n_ar[i])
    n = np.zeros(n_ar[i], dtype=int)
    d = np.zeros(n_ar[i], dtype=int)
    for j in range(n_ar[i]):
        X.append(np.load(data_path + 'X_{0}_nw{1}_{2}.npy'.format(dataset, n_ar[i], j)))
        y.append(np.load(data_path + 'y_{0}_nw{1}_{2}.npy'.format(dataset, n_ar[i], j)))
        n[j], d[j] = X[j].shape        
        
        currentDT = datetime.datetime.now()
        print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
        print (X[j].shape)

        hess_f_j = (1 / (4*n[j])) * (X[j].T @ X[j]) + 2*la*np.eye(d[j])
        L[j] = np.max(np.linalg.eigvals(hess_f_j))
    L = L.astype(np.float)

    if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
        # create a new w_0
        x_0 = np.random.normal(loc=0.0, scale=1.0, size=d_0)
        np.save(data_path + 'w_init_{0}.npy'.format(loss_func), x_0)
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
    else:
        # load existing w_0
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
    
    if k_ar[0] == d_0:
        step_size_diana_tpc = np.array([(1/L_0)*factor])
    else:    
        al_ar = k_ar/d_0
        #theory
        t_ar = -1 + np.sqrt(1/(1-al_ar))
        theta_ar = 1 - (1 - al_ar)*(1 + t_ar)
        beta_ar = (1 - al_ar)*(1 + 1/t_ar)
        Lt = np.sqrt (np.mean (L**2))
        step_size_diana_tpc = (1/(L_0 + Lt*np.sqrt(beta_ar/theta_ar)))*factor
    
    experiment_name = "biased-diana-full-grad_nw-{0}_{1}x".format(n_ar[i], factor)
    its_bdfg_tpc = []
    norms_bdfg_tpc = []
    sol_bdfg_tpc = []

    #for i in range (len(step_size)-3):
    for k in range (len(k_ar)):
        
        results = biased_diana_top_k_gd(x_0, X, y, X_0, y_0, step_size_diana_tpc[k], eps,la, k_ar[k], n_ar[i], Nsteps=nsteps)
        print (experiment_name + f" with k={k_ar[k]} finished in {results[0].shape[0]} iterations" )
        its_bdfg_tpc.append(results[0])
        norms_bdfg_tpc.append(results[1])
        sol_bdfg_tpc.append(results[2])
        
        save_data(its_bdfg_tpc[k], norms_bdfg_tpc[k], sol_bdfg_tpc[k], k_ar[k], experiment_name, project_path,dataset)

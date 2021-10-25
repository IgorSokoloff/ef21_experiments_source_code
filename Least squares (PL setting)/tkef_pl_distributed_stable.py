
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
#from logreg_functions_fast import *
from least_squares_functions_fast import *



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
    g_ar = np.zeros((n_workers, x.shape[0]))
    for i in range(n_workers):
        g_ar[i] = least_squares_grad(x, A[i], b[i], la).copy()
    return g_ar

def compute_full_funcs (A, x, b, la,n_workers):
    funcs_ar = np.zeros((n_workers, 1))
    for i in range(n_workers):
        funcs_ar[i] = least_squares_loss(x, A[i], b[i], la).copy()
    return funcs_ar

def compute_compensated_estimators(A, x, b, la, k, e_ar, n_workers, stepsize):
    grads = compute_full_grads(A, x, b, la, n_workers)
    v_ar_new = np.zeros((n_workers, x.shape[0]))
    estimator = e_ar + stepsize*grads
    v_ar_new = top_k_matrix(estimator, k)
    size_value_sent = sys.getsizeof(estimator[0,0])
   # print ("size_value_sent ", size_value_sent)
    return v_ar_new, grads, size_value_sent

def update_error (v_ar ,grads, k, e_ar, n_workers, stepsize):
    return e_ar + stepsize*grads - v_ar

def biased_diana_top_k_gd_ef(x_0, A, b, A_0, b_0, stepsize, eps,la,k, n_workers, Nsteps=100000):
    print(f"topk_ef-{n_workers}_k-{k}")

    e_ar = np.zeros((n_workers, x_0.shape[0])) # init error
    v_ar, grads, size_value_sent = compute_compensated_estimators(A, x_0, b, la, k, e_ar, n_workers, stepsize)

    v = np.mean(v_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=np.mean(grads, axis=0), ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]

    PRINT_EVERY = 1000

    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):

        x = x - v
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))

        e_ar = update_error (v_ar, grads, k, e_ar, n_workers, stepsize)#update error
        v_ar, grads, size_value_sent = compute_compensated_estimators(A, x, b, la, k, e_ar, n_workers, stepsize)

        v = np.mean(v_ar, axis=0)
        sq_norm_ar.append(np.linalg.norm(x=np.mean(grads, axis=0), ord=2) ** 2)
        it += 1
        it_ar.append(it*k*size_value_sent)
        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar)

def save_data(its, f_grad_norms, x_solution, f_ar, k_size, experiment_name, project_path, dataset):

    experiment = '{0}_{1}'.format(experiment_name, k_size)

    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'solution' + '_' + experiment, x_solution)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))
    np.save(logs_path + 'f_ar' + '_' + experiment, np.array(f_ar))

user_dir = os.path.expanduser('~/')
project_path = os.getcwd() + "/"

parser = argparse.ArgumentParser(description='Run top-k algorithm')
parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=None, help='Maximum number of iteration')
parser.add_argument('--k', action='store', dest='k', type=int, default=1, help='Sparcification parameter')
parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=1, help='Number of workers that will be used')
parser.add_argument('--factor', action='store', dest='factor', type=int, default=1, help='Stepsize factor')
parser.add_argument('--tol', action='store', dest='tol', type=float, default=1e-5, help='tolerance')
parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='phishing', help='name_of_dataset')

args = parser.parse_args()

nsteps = args.max_it
k_size = args.k
num_workers = args.num_workers

n_ar = np.array([num_workers])
k_ar = np.array([k_size])
factor = args.factor
eps = args.tol
dataset = args.dataset
loss_func = "least-squares"
data_path = project_path + "data_{0}/".format(dataset)
if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')

if not os.path.exists(data_path):
    os.mkdir(data_path)

data_info = np.load(data_path + 'data_info.npy')

#assert (type(la) == np.float64)

#n_ar = np.array([5, 20], dtype=int)
#n_ar = np.array([20], dtype=int)
#n_ar = np.array([5], dtype=int)

#k_ar = np.array([1, 5, 10, 25 ,50], dtype=int)
#k_ar = np.array([1, 5, 10, 25], dtype=int)
#k_ar = np.array([1, 5, 10],dtype=int)
#k_ar = np.array([1, 5], dtype=int)
#k_ar = np.array([1], dtype=int)


la = 0
X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')
n_0, d_0 = X_0.shape

hess_f_0 = (1 / (n_0)) * (X_0.T @ X_0) + 2*la*np.eye(d_0)
L_0 = np.max(np.linalg.eigvals(hess_f_0))
#L_0 = L_0.astype(float)
mu_0 = np.linalg.svd(X_0)[1][-1]**2
print(f"mu_0 = {mu_0}")

for i in range(len(n_ar)):
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
        #print (currentDT.strftime("%Y-%m-%d %H:%M:%S"))
        #print (X[j].shape)

        hess_f_j = (1 / (n[j])) * (X[j].T @ X[j]) + 2*la*np.eye(d[j])
        L[j] = np.max(np.linalg.eigvals(hess_f_j))
    L = L.astype(np.float)

    if not os.path.isfile(data_path + 'w_init_{0}.npy'.format(loss_func)):
        # create a new w_0
        x_0 = np.random.normal(loc=0.0, scale=2.0, size=d_0)
        np.save(data_path + 'w_init_{0}.npy'.format(loss_func), x_0)
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))
    else:
        # load existing w_0
        x_0 = np.array(np.load(data_path + 'w_init_{0}.npy'.format(loss_func)))


    ##################

    #x_0 = np.ones(d_0)

    ##################

    #print ("sqnorm(x_0): ", np.linalg.norm(x_0, ord=2))
    #sys.exit()
    #al_ar = k_ar/d_0

    #step_size_diana_ef1 = (al_ar/(10*L_0))*factor

    al_ar = k_ar/d_0
    #theory
    if k_ar[0] == d_0:
        theta_ar = 1 + 0*k_ar
        beta_ar = 0*k_ar
    else:
        t_ar = -1 + np.sqrt(1/(1-al_ar))
        theta_ar = 1 - (1 - al_ar)*(1 + t_ar)
        beta_ar = (1 - al_ar)*(1 + 1/t_ar)
    Lt = np.sqrt (np.mean (L**2))

    step_size_diana_ef = np.minimum( (1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))), theta_ar / (2*mu_0))*factor

    print('step_size_diana_ef: ', step_size_diana_ef)

    if theta_ar / (2*mu_0) < (1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))):
        print(f"PL stepsize works! Improvement in {(1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))) / (theta_ar / (2*mu_0))} times!")

    #print (f"step_size_ef: {step_size_diana_ef1}; step_size_tpc: {step_size_diana_ef}")
    #raise ValueError("")

    experiment_name = "full-grad-ef_nw-{0}_{1}x".format(n_ar[i], factor)

    #for i in range (len(step_size)-3):
    for k in range (len(k_ar)):

        results = biased_diana_top_k_gd_ef(x_0, X, y, X_0, y_0, step_size_diana_ef[k], eps,la, k_ar[k], n_ar[i], Nsteps=nsteps)
        print (experiment_name + f" with k={k_ar[k]} finished in {results[0].shape[0]} iterations" )
        its = results[0]
        norms = results[1]
        sols = results[2]
        f_ar = results[3]

        save_data(its, norms, sols, f_ar, k_ar[k], experiment_name, project_path,dataset)

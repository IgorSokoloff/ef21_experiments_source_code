

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
    grad_ar = np.zeros((n_workers, x.shape[0]))
    for i in range(n_workers):
        grad_ar[i] = least_squares_grad(x, A[i], b[i], la).copy()
    return grad_ar


def compute_full_funcs (A, x, b, la,n_workers):
    funcs_ar = np.zeros((n_workers, 1))
    for i in range(n_workers):
        funcs_ar[i] = least_squares_loss(x, A[i], b[i], la).copy()
    return funcs_ar

def choose_estimator(g_cg_ar, g_ef21_ar, grads, n_workers):
    g_ar_new = np.zeros_like(g_cg_ar)
    cg_upd, ef_upd = 0, 0
    for i in range(n_workers):
        #print(f"Q {norm(g_cg_ar[i] - grads[i])}; T {norm(g_ef21_ar[i] - grads[i])}")
        if norm(g_cg_ar[i] - grads[i]) < norm(g_ef21_ar[i] - grads[i]):
            g_ar_new[i] = g_cg_ar[i]
            cg_upd += 1
        else:
            g_ar_new[i] = g_ef21_ar[i]
            ef_upd += 1
    return g_ar_new, cg_upd, ef_upd

def biased_diana_top_k_gd_estimator(A, x, b, la, k, g_ar, n_workers):
    grads = compute_full_grads(A, x, b, la, n_workers)
    g_ef21_ar = np.zeros((n_workers, x.shape[0]))
    delta = grads - g_ar
    g_ef21_ar = g_ar + top_k_matrix(delta, k)
    #########
    g_cg_ar = top_k_matrix(grads, k)
    g_ar_new, cg_upd, ef_upd = choose_estimator(g_cg_ar, g_ef21_ar, grads, n_workers)
    #########
    size_value_sent = 32
    return g_ar_new, size_value_sent, grads, cg_upd, ef_upd

def biased_diana_top_k_gd(x_0, A, b, A_0, b_0, stepsize, eps,la,k, n_workers, theta, Nsteps=100000):
    print(f"biased_diana_gd_nw-{n_workers}_k-{k}")
    g_ar = compute_full_grads(A, x_0, b, la, n_workers)
    g = np.mean(g_ar, axis=0)
    sq_norm_ar = [np.linalg.norm(x=g, ord=2) ** 2]
    it_ar = [0]
    x = x_0.copy()
    it = 0
    f_ar = [np.mean(compute_full_funcs(A, x_0, b, la, n_workers))]
    p_ar = [0]
    PRINT_EVERY = 1000
    ef_upd_ar = [0]
    while stopping_criterion(sq_norm_ar[-1], eps, it, Nsteps):
        x = x - stepsize*g
        f_ar.append(np.mean(compute_full_funcs(A, x, b, la, n_workers)))

        g_ar, size_value_sent, grad_ar, cg_upd, ef_upd = biased_diana_top_k_gd_estimator(A, x, b, la, k, g_ar, n_workers)
        g = np.mean(g_ar, axis=0)
        grad = np.mean(grad_ar, axis=0)
        sq_norm_ar.append(np.linalg.norm(x=grad, ord=2) ** 2)
        p_ar.append((stepsize / theta) * np.linalg.norm(x= g - grad, ord=2) ** 2)
        it += 1
        it_ar.append(it*k*size_value_sent)
        if it%PRINT_EVERY ==0:
            display.clear_output(wait=True)
            print(it, sq_norm_ar[-1])

        ef_upd_ar.append(ef_upd)
    return np.array(it_ar), np.array(sq_norm_ar), x, np.array(f_ar), np.array(p_ar), np.array(ef_upd_ar)

def save_data(its, f_grad_norms, sol, f_ar, p_ar, ef_upd_ar, k_size, experiment_name, project_path, dataset):
    experiment = '{0}_{1}'.format(experiment_name, k_size)
    logs_path = project_path + "logs/logs_{0}_{1}/".format(dataset, experiment)

    if not os.path.exists(project_path + "logs/"):
        os.makedirs(project_path + "logs/")

    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    np.save(logs_path + 'iteration' + '_' + experiment, np.array(its))
    np.save(logs_path + 'sol' + '_' + experiment, sol)
    np.save(logs_path + 'norms' + '_' + experiment, np.array(f_grad_norms))
    np.save(logs_path + 'f_ar' + '_' + experiment, np.array(f_ar))
    np.save(logs_path + 'p_ar' + '_' + experiment, np.array(p_ar))
    np.save(logs_path + 'ef_upd_ar' + '_' + experiment, np.array(ef_upd_ar))

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


la = 0


X_0 = np.load(data_path + 'X.npy') #whole dateset
y_0 = np.load(data_path + 'y.npy')
n_0, d_0 = X_0.shape

hess_f_0 = (1 / (n_0)) * (X_0.T @ X_0) + 2*la*np.eye(d_0)

L_0 = np.max(np.linalg.eigvals(hess_f_0))
#L_0 = L_0.astype(np.float)
mu_0 = np.linalg.svd(X_0)[1][-1]**2
print(f"mu_0 = {mu_0}")

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

    step_size_diana_tpc = np.minimum( (1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))), theta_ar / (2*mu_0))*factor

    print('step_size_diana_tpc: ', step_size_diana_tpc)
    if theta_ar / (2*mu_0) < (1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))):
        print(f"PL stepsize works! Improvement in {(1/(L_0 + Lt*np.sqrt(2 * beta_ar/theta_ar))) / (theta_ar / (2*mu_0))} times!")

    experiment_name = "ef21-plus-full-grad_nw-{0}_{1}x".format(n_ar[0], factor)

    for k in range (len(k_ar)):

        results = biased_diana_top_k_gd(x_0, X, y, X_0, y_0, step_size_diana_tpc[k], eps,la, k_ar[k], n_ar[0],theta_ar[k], Nsteps=nsteps)
        print (experiment_name + f" with k={k_ar[k]} finished in {results[0].shape[0]} iterations" )
        its = results[0]
        norms = results[1]
        sol = results[2]
        f_ar = results[3]
        p_ar = results[4]
        ef_upd_ar = results[5]
        save_data(its, norms, sol, f_ar, p_ar, ef_upd_ar, k_ar[k], experiment_name, project_path,dataset)

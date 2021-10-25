#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import utils
import sys, os
from matplotlib.ticker import FormatStrFormatter
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as tck

class NNConfiguration: pass

plt.rcParams["lines.markersize"] = 30
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 27

size = 40
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'FreeSerif'
plt.rcParams['lines.linewidth'] = 4
# plt.rcParams['lines.markersize'] = 10
plt.rcParams['xtick.labelsize'] = size  # 40
plt.rcParams['ytick.labelsize'] = size  # 40
plt.rcParams['legend.fontsize'] = int(size*3/4)  # 30
plt.rcParams['axes.titlesize'] = int(size*1.5)  # 40
plt.rcParams['axes.labelsize'] = size  # 40
plt.rc('legend',fontsize=22)
#===================================================================================================
def get_subset(ctr, indicies):
    return [ctr[ind] for ind in indicies]
#===================================================================================================


#plt.rcParams["figure.figsize"] = [64,9]
plt.rcParams["figure.figsize"] = [64,12]
fig_four, axs_four = plt.subplots(1, 4)

#===================================================================================================
files = sys.argv[1:]
g = -1
K_ar = []
batch_size_for_worker_ar = []

for fname in files:

#===================================================================================================
    my = utils.deserialize(fname)
    transfered_bits_by_node = my["transfered_bits_by_node"]
    fi_grad_calcs_by_node   = my["fi_grad_calcs_by_node"]
    train_loss              = my["train_loss"]
    test_loss               = my["test_loss"]
    train_acc               = my["train_acc"]
    test_acc                = my["test_acc"]
    fn_train_loss_grad_norm = my["fn_train_loss_grad_norm"]
    fn_test_loss_grad_norm  = my["fn_test_loss_grad_norm"]
    nn_config               = my["nn_config"]
    current_data_and_time   = my["current_data_and_time"]
    experiment_description  = my["experiment_description"]
    compressor              = my["compressors"]
    compressors_rand_K      = my["compressors_rand_K"]
    nn_config               = my["nn_config"]
    algo_name               = my["algo_name"]
    algo_name_pure          = algo_name
    algo_name               = algo_name + f" (K$\\approx${(compressors_rand_K/nn_config.D):.3f}D)"
    K_ar.append(compressors_rand_K)
    
    freq = 10

    train_loss = [train_loss[i] for i in range(len(train_loss)) if i % freq == 0]
    test_loss  = [test_loss[i]  for i in range(len(test_loss))  if i % freq == 0]
    train_acc  = [train_acc[i]  for i in range(len(train_acc))  if i % freq == 0]
    test_acc   = [test_acc[i]   for i in range(len(test_acc))   if i % freq == 0]
    fn_train_loss_grad_norm  = [fn_train_loss_grad_norm[i]  for i in range(len(fn_train_loss_grad_norm))  if i % freq == 0]
    fn_test_loss_grad_norm   = [fn_test_loss_grad_norm[i]   for i in range(len(fn_test_loss_grad_norm))   if i % freq == 0]

    #===================================================================================================
    print("==========================================================")
    print(f"Informaion about experiment results '{fname}'")
    print(f"  Content has been created at '{current_data_and_time}'")
    print(f"  Experiment description: {experiment_description}")
    print(f"  Dimension of the optimization proble: {nn_config.D}")
    print(f"  Compressor RAND-K K: {compressors_rand_K}")
    print(f"  Number of Workers: {nn_config.kWorkers}")
    print(f"  Used step-size: {nn_config.gamma}")
    print()
    print("Whole config")
    
    for k in dir(nn_config):
        v = getattr(nn_config, k)
        if type(v) == int or type(v) == float:
            print(" ", k, "=", v)
            if k == "batch_size_for_worker": batch_size_for_worker_ar.append(v)

    print("==========================================================")

    #=========================================================================================================================
    KMax = nn_config.KMax
    mark_mult = 0.4

    fi_grad_calcs_sum      = np.sum(fi_grad_calcs_by_node, axis = 0)
    transfered_bits_sum    = np.sum(transfered_bits_by_node, axis = 0)        

    for i in range(1, KMax):
        transfered_bits_sum[i] = transfered_bits_sum[i] + transfered_bits_sum[i-1]

    transfered_bits_mean = transfered_bits_sum / nn_config.kWorkers       

    for i in range(1, KMax):
        fi_grad_calcs_sum[i] = fi_grad_calcs_sum[i] + fi_grad_calcs_sum[i-1]

    transfered_bits_mean_sampled = [transfered_bits_mean[i] for i in range(len(transfered_bits_mean)) if i % freq == 0] 

    #=========================================================================================================================

    epochs = (fi_grad_calcs_sum * 1.0) / (nn_config.train_set_full_samples)
    iterations = range(KMax)

    iterations_sampled =  [iterations[i] for i in range(len(iterations)) if i % freq == 0]
    epochs_sampled     =  [epochs[i] for i in range(len(epochs)) if i % freq == 0]

    #=========================================================================================================================
    markevery = [ int(mark_mult*KMax/4.0/freq*4.0), int(mark_mult*KMax/4.0/freq*4.0), int(mark_mult*KMax/3.5/freq), int(mark_mult*KMax/3.0/freq), 
                  int(mark_mult*KMax/4.0/freq), int(mark_mult*KMax/5.0/freq), int(mark_mult*KMax/3.0/freq), int(mark_mult*KMax/1.0/freq)]
    #[1, 1,1, 1]
    markevery_len = len(markevery)
    marker = ["d","d","^","^","^", "^", "^"]
    marker = ["o", "*", "v", "^", "<", ">", "s", "p", "p", "p", "p", "P", "h", "H", "+", "x", "X", "D", "d", "|", "_",1,2,3,4,5,6,7,8,9]
    color = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#ffff33", "#a65628", "#f781bf"]
    color_ar_1 = ['olive', 'blue', 'yellowgreen', 'orange', 'aqua', 'violet']
    color_ar_1 = ['tab:red','cornflowerblue', 'darkgreen',
                 'goldenrod', 'darkblue', 'maroon',
                  'yellowgreen', 'brown', 'coral', 'black', 'red', 'blue', 'orange', 'aqua', 'violet']
    linestyle = ["solid", "solid", "solid", "solid","solid","solid", "solid","solid"]
    #=========================================================================================================================
    #g = (g + 1)%len(color)
    print(f"algo_name: {algo_name}")
    if algo_name[:5] == "EF21+": g = 2
    elif algo_name[:4] == "EF21": g = 0
    elif algo_name[:2] == "EF": g = 1
    else: g = 3
    
    sz_factor = int(nn_config.gamma*1e3)
    l = int(np.log2(sz_factor))
    if algo_name[:5] == "EF21+": l = 2
    elif algo_name[:4] == "EF21": l = 0
    elif algo_name[:2] == "EF": l = 1
    else: l = 2
         
    if algo_name[:3] == "SGD":
        MAXIT = int(300 * 500_000 / 11689512)
    else:
        MAXIT = 150
        
       #========================================================================================================================
    #plot all four nicely
#=======================================================================================================================#======================================================================================================================  #=========================================================================================================================
    print(f"transfered_bits_mean_sampled: {len(transfered_bits_mean_sampled)}")
    transfered_bits_mean_sampled = np.array(transfered_bits_mean_sampled[:MAXIT])
    fn_train_loss_grad_norm = fn_train_loss_grad_norm[:MAXIT]
    train_loss = train_loss[:MAXIT]
    train_acc = train_acc[:MAXIT]
    test_acc = test_acc[:MAXIT]
    epochs_sampled = epochs_sampled[:MAXIT]
    
    axs_four[0].semilogy(transfered_bits_mean_sampled*1e-9, fn_train_loss_grad_norm, color=color_ar_1[l], marker=marker[g],
                markevery=markevery[(g+l)%markevery_len], 
                linestyle=linestyle[g], label=algo_name_pure+'; '+str(int(nn_config.gamma*1e3))+r"$\times$", markeredgecolor = "black")

    axs_four[0].set_xlabel('#Gbits/n')
    axs_four[0].set_ylabel('$||\\nabla f(x)||^2$')
    axs_four[0].grid(True)
    axs_four[0].legend(loc='best')
    #=========================================================================================================================


    #g = (g + 1)%len(color)
    axs_four[1].semilogy(transfered_bits_mean_sampled*1e-9, train_loss, color=color_ar_1[l], marker=marker[g], 
                markevery=markevery[(g+l)%markevery_len], 
                linestyle=linestyle[g], label=algo_name_pure+'; '+str(int(nn_config.gamma*1e3))+r"$\times$", markeredgecolor = "black")
    #ax.semilogy(transfered_bits_mean_sampled, test_loss, color=color[5], marker=marker[5], markevery=markevery[5], linestyle=linestyle[5], label="test")

    axs_four[1].set_xlabel('#Gbits/n')
    axs_four[1].set_ylabel('f(x)')
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    axs_four[1].grid(True)
    axs_four[1].legend(loc='best')
    
    
    
    axs_four[2].semilogy(transfered_bits_mean_sampled*1e-9, train_acc, color=color_ar_1[l], marker=marker[g], 
                markevery=markevery[(g+l)%markevery_len], 
                linestyle=linestyle[0], label=algo_name_pure +'; '+str(int(nn_config.gamma*1e3))+r"$\times$; train", markeredgecolor = "black")
    print(f"train_acc: {train_acc[-1]}")
    axs_four[2].set_xlabel('#Gbits/n')
    axs_four[2].set_ylabel('Accuracy')
    axs_four[2].set_yscale('linear')
    axs_four[2].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    axs_four[2].grid(True)
    axs_four[2].legend(loc='best')
    #=========================================================================================================================

    axs_four[3].semilogy(transfered_bits_mean_sampled*1e-9, test_acc, color=color_ar_1[l], marker=marker[g], 
                markevery=markevery[(g+l)%markevery_len], 
                linestyle=linestyle[0], label=algo_name_pure +'; '+str(int(nn_config.gamma*1e3))+r"$\times$; test", markeredgecolor = "black")
    print(f"test_acc: {test_acc[-1]}")
    axs_four[3].set_xlabel('#Gbits/n')
    axs_four[3].set_ylabel('Accuracy')
    axs_four[3].set_yscale('linear')
    axs_four[3].yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    axs_four[3].grid(True)
    axs_four[3].legend(loc='best')###was 25 here    #=========================================================================================================================
    
for column in range(4):
    axs_four[column].locator_params(axis='x', nbins=4)
    axs_four[column].xaxis.set_minor_locator(tck.AutoMinorLocator(5))
for column in range(2):
    locmin = tck.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=20)
    axs_four[column].yaxis.set_minor_locator(locmin)
    axs_four[column].yaxis.set_minor_formatter(tck.NullFormatter())

if len(batch_size_for_worker_ar) > 1:
    for i in range(1, len(batch_size_for_worker_ar)):
        assert batch_size_for_worker_ar[i] == batch_size_for_worker_ar[i-1]
batch_size_for_worker = batch_size_for_worker_ar[0]
        
plt.suptitle(f'{experiment_description} with $k\\approx${(compressors_rand_K/nn_config.D):.2f}$D$, $\\tau = {batch_size_for_worker}$')

if len(K_ar) > 1:
    for i in range(1, len(K_ar)):
        assert K_ar[i] == K_ar[i-1]
    

if True:
    plt.show()

fig_four.tight_layout()

save_to = f"9_nice_plot_bsz_{batch_size_for_worker}_K_{(100*compressors_rand_K/nn_config.D):.0f}.pdf"

fig_four.savefig(save_to, bbox_inches='tight')
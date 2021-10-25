#!/usr/bin/env python3.9
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import random, sys, os
import asyncio, time
import threading
from datetime import datetime
import argparse

# Utils class
import utils
import compressors

# PyTorch modules
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from torch.utils.data import TensorDataset, DataLoader

#====================================================================================ds
class NNConfiguration: pass
class WorkersConfiguration: pass
#====================================================================================
transfered_bits_by_node = None
fi_grad_calcs_by_node   = None
train_loss              = None
test_loss               = None
train_acc               = None
test_acc                = None
fn_train_loss_grad_norm = None
fn_test_loss_grad_norm  = None

def add_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] + y[i])
    return z

def sub_params(x, y):
    z = []
    for i in range(len(x)):
        z.append(x[i] - y[i])
    return z

def mult_param(alpha, x):
    z = []
    for i in range(len(x)):
        z.append(alpha*x[i])
    return z

def norm_of_param(x):
    z = 0
    for i in range(len(x)):
        z += torch.norm(x[i].flatten(0))
    return z

#====================================================================================

print_lock = threading.Lock()

def dbgprint(wcfg, *args):
    printing_dbg = True
    if printing_dbg == True:
        print_lock.acquire()
        print(f"Worker {wcfg.worker_id}/{wcfg.total_workers}:", *args, flush = True)
        print_lock.release()

def rootprint(*args):
    print_lock.acquire()
    print(f"Master: ", *args, flush = True)
    print_lock.release()

def getAccuracy(model, trainset, batch_size, device):
    avg_accuracy = 0

    dataloader = DataLoader(
                trainset,                  # dataset from which to load the data.
                batch_size=batch_size,     # How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)           
    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             # move to device
        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
        avg_accuracy += (logits.data.argmax(1) == outputs).sum().item()

    avg_accuracy /= len(trainset)
    model.train(prev_train_mode)

    return avg_accuracy

def getLossAndGradNorm(model, trainset, batch_size, device):
    total_loss = 0
    grad_norm = 0
    #print("~~ trainset: ", type(trainset))

    one_inv_samples = torch.Tensor([1.0/len(trainset)]).to(device)

    dataloader = DataLoader(
                trainset,                  # Dataset from which to load the data.
                batch_size=batch_size,     # How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

    prev_train_mode = torch.is_grad_enabled()  
    model.train(False)

    for p in model.parameters():
        p.grad = None

    for inputs, outputs in dataloader:
        inputs, outputs = inputs.to(device), outputs.to(device)                             # move to device

        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network

        #print("~~logits", logits.shape)
        #print("~~outputs",outputs.shape)

        loss = one_inv_samples * F.cross_entropy(logits, outputs, reduction='sum')          # compute objective
        loss.backward()                                                                     # compute the gradient (backward-pass)
        total_loss += loss

    for p in model.parameters(): 
        grad_norm += torch.norm(p.grad.data.flatten(0))**2
        p.grad = None

    model.train(prev_train_mode)
    return total_loss, grad_norm

#======================================================================================================================================
class WorkerThreadMarina(threading.Thread):
  def __init__(self, wcfg, ncfg):
    threading.Thread.__init__(self)
    self.wcfg = wcfg
    self.ncfg = ncfg

    self.model = utils.getModel(ncfg.model_name, wcfg.train_set_full, wcfg.device)
    self.model = self.model.to(wcfg.device)                    # move model to device
    utils.setupAllParamsRandomly(self.model)
    wcfg.model = self.model
 
  def run(self):
    wcfg = self.wcfg
    ncfg = self.ncfg

    global transfered_bits_by_node
    global fi_grad_calcs_by_node
    global train_loss
    global test_loss
    global fn_train_loss_grad_norm
    global fn_test_loss_grad_norm

    # wcfg - configuration specific for worker
    # ncfg - general configuration with task description

    dbgprint(wcfg, f"START WORKER. IT USES DEVICE", wcfg.device, ", AND LOCAL TRAINSET SIZE IN SAMPLES IS: ", len(wcfg.train_set))
    #await init_workers_permission_event.wait()

    model = self.model

    loaders = (wcfg.train_loader, wcfg.test_loader)  # train and test loaders

    #one_div_trainset_all_len = torch.Tensor([1.0/len(wcfg.train_set_full)]).to(wcfg.device)
    one_div_trainset_len    = torch.Tensor([1.0/len(wcfg.train_set)]).to(wcfg.device)

    if ncfg.i_use_vr_marina or ncfg.i_use_vr_biased_diana:
        one_div_batch_prime_len = torch.Tensor([1.0/(ncfg.batch_size_for_worker)]).to(wcfg.device)
    else:
        one_div_batch_prime_len = torch.Tensor([1.0/len(wcfg.train_set)]).to(wcfg.device)

    delta_flatten = torch.zeros(ncfg.D).to(wcfg.device)
    #For EF
    #=========================================================================================
    prev_train_mode = torch.is_grad_enabled()  
    model.train(True)

    for inputs, outputs in wcfg.train_loader:
        inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
        logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
        loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')     # compute objective
        loss.backward()                                                                     # compute the gradient (backward-pass)
       
    e_i = []
    v_i = []
    for p in model.parameters(): 
        v_i.append(p.grad.data.detach().clone())
        e_i.append(0*p.grad.data.detach().clone())
        p.grad = None

    model.train(prev_train_mode)
    #=========================================================================================
    
    #=========================================================================================
    iteration = 0
    #=========================================================================================

    while True:
        wcfg.input_cmd_ready.acquire()
        if wcfg.cmd == "exit":
            wcfg.output_of_cmd = ""
            wcfg.cmd_output_ready.release()
            break

#        if wcfg.cmd == "curr_x":
#            wcfg.output_of_cmd = []
#            for p in model.parameters(): 
#                wcfg.output_of_cmd.append(p.data.detach().clone())
#            wcfg.cmd_output_ready.release()

        if wcfg.cmd == "bcast_g_c1":
            wcfg.output_of_cmd = []
            k = 0
            for p in model.parameters(): 
                p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
                k = k + 1

            prev_train_mode = torch.is_grad_enabled()
            model.train(True)

            for inputs, outputs in wcfg.train_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
                logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
                loss = one_div_trainset_len * F.cross_entropy(logits, outputs, reduction='sum')     # compute objective
                loss.backward()                                                                     # compute the gradient (backward-pass)

            for p in model.parameters(): 
                wcfg.output_of_cmd.append(p.grad.data.detach().clone())
                p.grad = None

            model.train(prev_train_mode)

            # Case when we send to master really complete gradient
            transfered_bits_by_node[wcfg.worker_id, iteration] = ncfg.D * ncfg.component_bits_size
            # On that "c1" mode to evaluate gradient we call oracle number of times how much data is in local "full" batch
            fi_grad_calcs_by_node[wcfg.worker_id, iteration] = len(wcfg.train_set)

            wcfg.cmd_output_ready.release()
            iteration += 1

        if wcfg.cmd == "bcast_g_c0":
            wcfg.output_of_cmd = []

            #================================================================================================================================
            # Generate subsample with b' cardinality
            indicies = None
            if ncfg.i_use_vr_marina or ncfg.i_use_vr_biased_diana:
                indicies = torch.randperm(len(wcfg.train_set))[0:ncfg.batch_size_for_worker]
                subset = torch.utils.data.Subset(wcfg.train_set, indicies)
            else:
                subset = wcfg.train_set
            #================================================================================================================================
            minibatch_loader = DataLoader(
                subset,                    # dataset from which to load the data.
                batch_size=ncfg.batch_size,# How many samples per batch to load (default: 1).
                shuffle=False,             # Set to True to have the data reshuffled at every epoch (default: False)
                drop_last=False,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
                pin_memory=False,          # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
                collate_fn=None,           # Merges a list of samples to form a mini-batch of Tensor(s)
            )

            # Evaluate in previous point SGD within b' batch 
            prev_train_mode = torch.is_grad_enabled()  
            model.train(True)

            #================================================================================================================================
            #previous point skipped
            #================================================================================================================================
            # Change xk: move from x(k) to x(k+1)
            k = 0
            for p in model.parameters(): 
                p.data = p.data - ncfg.gamma * wcfg.input_for_cmd[k]
                k = k + 1

            # Evaluate SGD in the new point within b' batch 
            #================================================================================================================================
            for inputs, outputs in minibatch_loader:
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)                   # move to device
                logits = model(inputs)                                                              # forward-pass: Make a forward pass through the network
                loss = one_div_batch_prime_len * F.cross_entropy(logits, outputs, reduction='sum')     # compute objective
                loss.backward()                                                                     # compute the gradient (backward-pass)
            g_batch_next = []
            for p in model.parameters():
                g_batch_next.append(p.grad.data.detach().clone())
                p.grad = None

            u_i = add_params(e_i, g_batch_next)

            delta_offset = 0
            for t in range(len(u_i)):
                offset = len(u_i[t].flatten(0))
                delta_flatten[(delta_offset):(delta_offset + offset)] = u_i[t].flatten(0)
                delta_offset += offset

            delta_flatten = wcfg.compressor.compressVector(delta_flatten)

            delta_offset = 0
            for t in range(len(v_i)):
                offset = len(v_i[t].flatten(0))
                v_i[t].flatten(0)[:] = delta_flatten[(delta_offset):(delta_offset + offset)]
                delta_offset += offset

            e_i = sub_params(u_i, v_i)
            wcfg.output_of_cmd = v_i
            model.train(prev_train_mode)

            transfered_bits_by_node[wcfg.worker_id, iteration] = wcfg.compressor.last_need_to_send_advance * ncfg.component_bits_size
            fi_grad_calcs_by_node[wcfg.worker_id, iteration] = ncfg.batch_size_for_worker
            iteration += 1

            wcfg.cmd_output_ready.release()

        if wcfg.cmd == "full_grad":
            wcfg.output_of_cmd = []
            prev_train_mode = torch.is_grad_enabled()  
            model.train(True)

            for inputs, outputs in wcfg.train_loader:
                #dbgprint(wcfg, ">>>>", len(inputs))
                inputs, outputs = inputs.to(wcfg.device), outputs.to(wcfg.device)     # move to device
                logits = model(inputs)                                                # forward-pass: Make a forward pass through the network
                loss = one_div_trainset_len * F.cross_entropy(logits, outputs)        # compute objective
                loss.backward()                                                       # compute the gradient (backward-pass)

            for p in model.parameters(): 
                wcfg.output_of_cmd.append(p.grad.data.detach().clone())
                p.grad = None

            model.train(prev_train_mode)
            wcfg.cmd_output_ready.release()

    # Signal that worker has finished initialization via decreasing semaphore
    #completed_workers_semaphore.acquire()
    dbgprint(wcfg, f"END")
#======================================================================================================================================

def main():
    global transfered_bits_by_node
    global fi_grad_calcs_by_node
    global train_loss
    global test_loss
    global train_acc
    global test_acc
    global fn_train_loss_grad_norm
    global fn_test_loss_grad_norm

    
    ##############################################################################
    ##############################################################################
    parser = argparse.ArgumentParser(description='Run top-k algorithm')
    parser.add_argument('--max_it', action='store', dest='max_it', type=int, default=3000, help='Maximum number of iteration')
    parser.add_argument('--k', action='store', dest='k', type=int, default=100000, help='Sparcification parameter')
    parser.add_argument('--num_workers', action='store', dest='num_workers', type=int, default=5, help='Number of workers that will be used')
    parser.add_argument('--factor', action='store', dest='factor', type=int, default=1, help='Stepsize factor')
    parser.add_argument('--batch_size', action='store', dest='batch_size', type=int, default=1024, help='Batch size per worker and for GPU')
    parser.add_argument('--model', action='store', dest='model', type=str, default='resnet18', help='Name of NN architechture')
    parser.add_argument('--dataset', action='store', dest='dataset', type=str, default='CIFAR10', help='Name of dataset')
    args = parser.parse_args()
    ##############################################################################     ############################################################################## 
    
    
    utils.printTorchInfo()
    print("******************************************************************")
    cpu_device   = torch.device("cpu")      # CPU device
    gpu_device_0 = torch.device('cuda:0')   # Selected GPU (index 0)
    gpu_device_1 = torch.device('cuda:1')   # Selected GPU (index 1)

    available_devices = [gpu_device_1]
    master_device = available_devices[0]
    print("******************************************************************")
    global nn_config, workers_config

    # Configuration for NN
    torch.manual_seed(1)        # Set the random seed so things involved torch.randn are predictable/repetable 

    nn_config = NNConfiguration()
    nn_config.dataset = args.dataset #"CIFAR10"          # Dataset
    nn_config.model_name = args.model #"resnet18"      # NN architecture
    nn_config.load_workers = 0             # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
    nn_config.batch_size = args.batch_size#1024#128             # Technical batch size for training (due to GPU limitations)
    nn_config.KMax = args.max_it                  # Maximum number of iterations

    # Number of workers
    kWorkers = args.num_workers                                # Number of workers

    #=======================================================================================================
    # Load data
    train_sets, train_set_full, test_set, train_loaders, test_loaders, classes = utils.getSplitDatasets(nn_config.dataset, nn_config.batch_size,
                                                                                                        nn_config.load_workers, kWorkers)

    print(f"Start training {nn_config.model_name}@{nn_config.dataset} for K={nn_config.KMax} iteration. EF", available_devices)
    master_model = utils.getModel(nn_config.model_name, train_set_full, master_device)
    master_model = master_model.to(master_device)
    utils.setupAllParamsRandomly(master_model)

    utils.printLayersInfo(master_model, nn_config.model_name)
    nn_config.D = utils.numberOfParams(master_model)
    nn_config.component_bits_size = 32

    # Statistics/Metrics during training
    transfered_bits_by_node = np.zeros((kWorkers, nn_config.KMax)) # Transfered bits
    fi_grad_calcs_by_node   = np.zeros((kWorkers, nn_config.KMax)) # Evaluate number gradients for fi
    train_loss              = np.zeros((nn_config.KMax))           # Train loss
    test_loss               = np.zeros((nn_config.KMax))           # Validation loss
    train_acc               = np.zeros((nn_config.KMax))           # Train accuracy
    test_acc                = np.zeros((nn_config.KMax))           # Validation accuracy

    fn_train_loss_grad_norm = np.zeros((nn_config.KMax))           # Gradient norm for train loss
    fn_test_loss_grad_norm  = np.zeros((nn_config.KMax))           # Gradient norm for test loss

    nn_config.kWorkers = kWorkers         # Number of workers
    nn_config.i_use_vr_marina = False      # Algorithm for test
    nn_config.i_use_marina    = False     #
    nn_config.i_use_diana     = False     #
    nn_config.i_use_vr_diana  = False     #
    nn_config.i_use_vr_biased_diana  = True     #

    # TUNABLE PARAMS
    #=======================================================================================================
    K = args.k #100000
    gamma = args.factor*1e-3   #512*1e-3
    batch_size_for_worker = args.batch_size   #1024#256
    #=======================================================================================================
    nn_config.train_set_full_samples = len(train_set_full)
    nn_config.train_sets_samples = [len(s) for s in train_sets]
    nn_config.test_set_samples = len(test_set)
    #=======================================================================================================
 
    #c = compressors.Compressor()
    #c.makeRandKCompressor(int(K), nn_config.D)
    #w = c.getW() 

    # Fixed parameters for NN settings
    if nn_config.i_use_vr_marina:
      nn_config.gamma = gamma                                    # Gamma for VR-MARINA
      nn_config.batch_size_for_worker = batch_size_for_worker    # Batch for VR-MARINA
      # p for VR-MARINA
      nn_config.p     =  min(1.0/(1+w), (nn_config.batch_size_for_worker)/(nn_config.batch_size_for_worker + len(train_sets[0])))
    elif nn_config.i_use_vr_biased_diana:
      c_top = compressors.Compressor()
      c_top.makeTopKCompressor(int(K), nn_config.D)
      alpha = c_top.getAlpha() 
      nn_config.gamma = gamma                                    # Gamma for VR-BIASED-DIANA
      nn_config.batch_size_for_worker = batch_size_for_worker 
        
    elif nn_config.i_use_marina:
      nn_config.gamma = gamma                                    # Gamma for MARINA
      nn_config.p     = 1.0/(1+w)                                # p for MARINA
    elif nn_config.i_use_diana:
      nn_config.gamma = gamma                                    # Gamma for DIANA
      nn_config.fixed_alpha_diana = 1.0/(1+w)                    # Alpha for DIANA
    elif nn_config.i_use_vr_diana:
      nn_config.gamma = gamma                                    # Gamma for VR-DIANA
      nn_config.fixed_alpha_diana = 1.0/(1+w)                    # Alpha for VR-DIANA
      nn_config.batch_size_for_worker = batch_size_for_worker    # Batch for VR-DIANA

      m = (len(train_sets[0]))/(nn_config.batch_size_for_worker)
      nn_config.p     = 1.0/m                   # p for VR-DIANA


    worker_tasks = []                           # Worker tasks
    worker_cfgs = []                            # Worker configurations

    for i in range(kWorkers):
        worker_cfgs.append(WorkersConfiguration())
        worker_cfgs[-1].worker_id = i
        worker_cfgs[-1].total_workers = kWorkers
        worker_cfgs[-1].train_set = train_sets[i]
        worker_cfgs[-1].test_set = test_set
        worker_cfgs[-1].train_set_full = train_set_full

        worker_cfgs[-1].train_loader = train_loaders[i]
        worker_cfgs[-1].test_loader = test_loaders[i]
        worker_cfgs[-1].classes = classes
        worker_cfgs[-1].device = available_devices[i % len(available_devices)]                # Device used by worker
        worker_cfgs[-1].compressor = compressors.Compressor()
        #worker_cfgs[-1].compressor.makeRandKCompressor(int(K), nn_config.D)
        worker_cfgs[-1].compressor.makeTopKCompressor(int(K), nn_config.D) 

        worker_cfgs[-1].input_cmd_ready  = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd_output_ready = threading.Semaphore(value=0)
        worker_cfgs[-1].cmd = "init"
        worker_cfgs[-1].input_for_cmd = ""
        worker_cfgs[-1].output_of_cmd = ""

        worker_tasks.append(WorkerThreadMarina(worker_cfgs[-1], nn_config))

    for i in range(kWorkers):
        worker_tasks[i].start()
    #===================================================================================
    if nn_config.i_use_marina or nn_config.i_use_vr_marina or nn_config.i_use_vr_biased_diana:
        # Evaluate g0
        for i in range(kWorkers):
            worker_cfgs[i].cmd = "full_grad"
            worker_cfgs[i].input_cmd_ready.release()

        for i in range(kWorkers):
            worker_cfgs[i].cmd_output_ready.acquire()

        g0 = worker_cfgs[0].output_of_cmd
        worker_cfgs[0].output_of_cmd = None
        for i in range(1, kWorkers):
            for j in range(len(worker_cfgs[i].output_of_cmd)):
                g0[j] = g0[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
            worker_cfgs[i].output_of_cmd = None
        g0 = mult_param(1.0/kWorkers, g0)
        gk = g0

        rootprint(f"Start {nn_config.KMax} iterations of algorithm")

        for iteration in range(0, nn_config.KMax):
            #if k % 2 == 0:
            rootprint(f"Iteration {iteration}/{nn_config.KMax}. Completed by ", iteration/nn_config.KMax * 100.0, "%")
            #====================================================================
            # Collect statistics
            #====================================================================
            if iteration % 10 == 0:
                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
                train_loss[iteration] = loss
                fn_train_loss_grad_norm[iteration] = grad_norm

                loss, grad_norm = getLossAndGradNorm(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
                test_loss[iteration] = loss
                fn_test_loss_grad_norm[iteration] = grad_norm

                train_acc[iteration] = getAccuracy(worker_cfgs[0].model, train_set_full, nn_config.batch_size, worker_cfgs[0].device)
                test_acc[iteration]  = getAccuracy(worker_cfgs[0].model, test_set, nn_config.batch_size, worker_cfgs[0].device)
                print(f"  train accuracy: {train_acc[iteration]}, test accuracy: {test_acc[iteration]}, train loss: {train_loss[iteration]}, test loss: {test_loss[iteration]}")
                print(f"  grad norm train: {fn_train_loss_grad_norm[iteration]}, test: {fn_test_loss_grad_norm[iteration]}")
                print(f"  used step-size: {nn_config.gamma}")

            else:
                train_loss[iteration]              = train_loss[iteration - 1] 
                fn_train_loss_grad_norm[iteration] = fn_train_loss_grad_norm[iteration - 1]
                test_loss[iteration]               = test_loss[iteration - 1]
                fn_test_loss_grad_norm[iteration]  = fn_test_loss_grad_norm[iteration - 1] 
                train_acc[iteration]               = train_acc[iteration - 1]
                test_acc[iteration]                = test_acc[iteration - 1]
            #====================================================================

            ck = 0 #dont need this, it was from Bernoulli draw, effectlively it is always 0 for biased diana method

            # Broadcast gk wti command to workers
            gk_for_device = {}
            for d_id in range(len(available_devices)):
                gk_loc = []
                for gk_i in gk:
                    gk_loc.append(gk_i.to(available_devices[d_id]))
                gk_for_device[available_devices[d_id]] = gk_loc

            if ck == 0:
                for i in range(kWorkers):
                    worker_cfgs[i].cmd = "bcast_g_c0"
                    worker_cfgs[i].input_for_cmd = gk_for_device[worker_cfgs[i].device]
                    worker_cfgs[i].input_cmd_ready.release()

            # Obtain workers gi and average result
            for i in range(kWorkers): 
                worker_cfgs[i].cmd_output_ready.acquire()

            gk_next = worker_cfgs[0].output_of_cmd
            worker_cfgs[0].output_of_cmd = None
            for i in range(1, kWorkers): 
                for j in range(len(worker_cfgs[i].output_of_cmd)):
                    gk_next[j] = gk_next[j] + worker_cfgs[i].output_of_cmd[j].to(master_device)
                worker_cfgs[i].output_of_cmd = None
            gk_next = mult_param(1.0/kWorkers, gk_next)
            gk = gk_next

    #===================================================================================

    # Finish all work of nodes
    for i in range(kWorkers):
        worker_cfgs[i].cmd = "exit"
        worker_cfgs[i].input_cmd_ready.release()
    #==================================================================================
    for i in range(kWorkers):
        worker_tasks[i].join()
    print(f"Master has been finished")
    #==================================================================================
    my = {}
    my["transfered_bits_by_node"] = transfered_bits_by_node
    my["fi_grad_calcs_by_node"] = fi_grad_calcs_by_node

    my["train_loss"] = train_loss
    my["test_loss"] = test_loss
    my["train_acc"] = train_acc
    my["test_acc"]  = test_acc

    my["fn_train_loss_grad_norm"] = fn_train_loss_grad_norm
    my["fn_test_loss_grad_norm"] = fn_test_loss_grad_norm
    my["nn_config"] = nn_config
    my["current_data_and_time"] = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    my["experiment_description"] = f"Training {nn_config.model_name}@{nn_config.dataset}"
    my["compressors"] = worker_cfgs[0].compressor.fullName()
    my["algo_name"] = f"EF"
    if hasattr(worker_cfgs[0].compressor, "K"):
        my["compressors_rand_K"] = worker_cfgs[0].compressor.K

    if nn_config.i_use_vr_marina: prefix4algo = "vr_marina"
    if nn_config.i_use_marina:    prefix4algo = "marina"
    if nn_config.i_use_vr_diana:  prefix4algo = "vr_diana"
    if nn_config.i_use_diana:     prefix4algo = "diana"
    if nn_config.i_use_vr_biased_diana: prefix4algo = "EF"
    ser_fname = f"experiment_{prefix4algo}_K_{K}_sz_{int(gamma/1e-3)}_nw_{kWorkers}_bsz_{batch_size_for_worker}_{nn_config.model_name}_at_{nn_config.dataset}.bin"
    utils.serialize(my, ser_fname)
    print(f"Experiment info has been serialised into '{ser_fname}'")
    #==================================================================================

if __name__ == "__main__":
    main()


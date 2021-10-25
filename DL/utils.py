#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

import sys, os, pickle, math
import psutil 
import time

import torch
import torchvision

import torchvision.transforms as transforms

from torch.utils.collect_env import get_pretty_env_info
from torch.utils.data import TensorDataset, DataLoader

from pathlib import Path

import_time = time.time()

plt.rcParams["lines.markersize"] = 20
plt.rcParams["lines.linewidth"] = 2
plt.rcParams["font.size"] = 27

def serialize(obj, fname):
    with open(fname, "wb") as f:
        pickle.dump(obj, f)

def deserialize(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)

def printSystemInfo():
    print("")
    print("*********************************************************************************************")
    print("Path to python interpretator:", sys.executable)
    print("Version:", sys.version)
    print("Platform name:", sys.platform)
    print("Physical CPU processors: ", psutil.cpu_count(logical=False))
    print("Logical CPU processors: ", psutil.cpu_count(logical=True))
    print("Current CPU Frequncy: ", psutil.cpu_freq().current, "MHz")
    print("Installed Physical available RAM Memory: %g %s" % (psutil.virtual_memory().total/(1024.0**3), "GBytes"))
    print("Available physical available RAM Memory: %g %s" % (psutil.virtual_memory().available/(1024.0**3), "GBytes"))
    print("")
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()

    print("Process Resident Set (Working Set) Size: ", mem_info.rss/(1024.0 * 1024.0), "MBytes")
    print("Virtual Memory used by process: ", mem_info.vms/(1024.0 * 1024.0), "MBytes")
    print("*********************************************************************************************")
    print("Time since program starts: ", str(time.time() - import_time), " seconds")
    print("*********************************************************************************************")
    print("Script name: ", sys.argv[0])
    print("*********************************************************************************************")

def printTorchInfo():
    print("******************************************************************")
    print("Is CUDA avilable:", torch.cuda.is_available())
    print("GPU devices: ", torch.cuda.device_count())
    print("Torch hub(cache) for loaded datasets: ", torch.hub.get_dir())
    print("******************************************************************")
    print("")
    print(get_pretty_env_info())
    print("******************************************************************")

def numberOfParams(model):
    total_number_of_scalar_parameters = 0
    for p in model.parameters(): 
        total_items_in_param = 1
        for i in range(p.data.dim()):
            total_items_in_param = total_items_in_param * p.data.size(i)
        total_number_of_scalar_parameters += total_items_in_param
    return total_number_of_scalar_parameters

def printLayersInfo(model,model_name):
    # Statistics about used modules inside NN
    max_string_length = 0
    basic_modules = {}

    for module in model.modules():
        class_name = str(type(module)).replace("class ", "").replace("<'", "").replace("'>", "")
        if class_name.find("torch.nn") != 0:
            continue
        max_string_length = max(max_string_length, len(class_name))
 
        if class_name not in basic_modules:
            basic_modules[class_name]  = 1
        else:
            basic_modules[class_name] += 1

    print(f"Summary about layers inside {model_name}")
    print("=============================================================")
    for (layer, count) in basic_modules.items():
        print(f"{layer:{max_string_length + 1}s} occured {count:02d} times")
    print("=============================================================")
    print("Total number of parameters inside '{}' is {:,}".format(model_name, numberOfParams(model)))
    print("=============================================================")
#=======================================================================================================

def getModel(model_name, dataset, device):
    model_class = getattr(torchvision.models, model_name)
    model = model_class(pretrained=True).to(device)

    model.train(False)
    max_class = 0
    min_class = 0

    samples = len(dataset)
    for sample_idx in range(samples):
        input_sample, target = dataset[sample_idx]
        max_class = max(target, max_class)
        min_class = min(target, min_class)

    number_of_classes_in_dataset = max_class - min_class + 1 
    print("number_of_classes_in_dataset: ", number_of_classes_in_dataset)

    #print(f"dataset[0][0]: {dataset[0][0]}")
    out_one_hot_encoding = model(dataset[0][0].unsqueeze(0).to(device)).numel()
    #out_one_hot_encoding = model(dataset[0][0].to(device)).numel() added by Ilyas
    print("number of output class in original model: ", out_one_hot_encoding)

    final_model = torch.nn.Sequential(model,
                                      torch.nn.ReLU(),
                                      torch.nn.Linear(out_one_hot_encoding, number_of_classes_in_dataset, bias = False)).to(device)

    # For test only
    #final_model = torch.nn.Sequential(torch.nn.Flatten(1), 
    #                                  torch.nn.Linear(32*32*3, number_of_classes_in_dataset, bias = False)).to(device)

    final_model = model

    #out_one_hot_encoding = final_model(dataset[0][0].unsqueeze(0).to(device)).numel()
    #print("number of output class in a final model: ", out_one_hot_encoding)

    return final_model

def getDatasets(dataset_name, batch_size, load_workers):
    root_dir  = Path(torch.hub.get_dir()) / f'datasets/{dataset_name}'
    ds = getattr(torchvision.datasets, dataset_name)
    transform = transforms.Compose([
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape 
                # (C x H x W) in the range [0.0, 1.0]  
                # https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.ToTensor
                transforms.ToTensor(),
                #  https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Normalize
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_set = ds(root=root_dir, 
                   train=True, 
                   download=True, 
                   transform = transform
                  )

    test_set = ds(root=root_dir, 
                  train=False, 
                  download=True, 
                  transform = transform
                  )

    train_loader = DataLoader(
        train_set,                # dataset from which to load the data.
        batch_size=batch_size,    # How many samples per batch to load (default: 1).
        shuffle=True,             # Set to True to have the data reshuffled at every epoch (default: False)
        num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        drop_last=True,           # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
        pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
    )

    test_loader = DataLoader(
        test_set,                 # dataset from which to load the data.
        batch_size=batch_size,    # How many samples per batch to load (default: 1).
        shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
        num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
        drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
        pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
        collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
    )
    #==================================================================================================================================   
    classes = None

    if dataset_name == "CIFAR10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_set, test_set, train_loader, test_loader, classes 

def getSplitDatasets(dataset_name, batch_size, load_workers, train_workers):
    root_dir  = Path(torch.hub.get_dir()) / f'datasets/{dataset_name}'
    ds = getattr(torchvision.datasets, dataset_name)
    transform = transforms.Compose([
                # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape 
                # (C x H x W) in the range [0.0, 1.0]  
                # https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.ToTensor
                transforms.ToTensor(),
                #  https://pytorch.org/vision/0.8/transforms.html#torchvision.transforms.Normalize
                # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])

    train_set = ds(root=root_dir, 
                   train=True, 
                   download=True, 
                   transform = transform
                  )

    test_set = ds(root=root_dir, 
                  train=False, 
                  download=True, 
                  transform = transform
                  )

    lengths = [batch_size*math.ceil(len(train_set)/(train_workers*batch_size))] * (train_workers - 1)
    lengths.append(len(train_set) - sum(lengths))
    train_sets = torch.utils.data.random_split(train_set, lengths)
    train_loaders = []
    test_loaders = []

    print(f"Total train set size for '{dataset_name}' is ", len(train_set))

    for t in range(train_workers):
        train_loader = DataLoader(
            train_sets[t],            # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        test_loader = DataLoader(
            test_set,                 # dataset from which to load the data.
            batch_size=batch_size,    # How many samples per batch to load (default: 1).
            shuffle=False,            # Set to True to have the data reshuffled at every epoch (default: False)
            num_workers=load_workers, # How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.
            drop_last=False,          # Set to True to drop the last incomplete batch, if the dataset size is not divisible by the batch size
            pin_memory=False,         # If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
            collate_fn=None,          # Merges a list of samples to form a mini-batch of Tensor(s)
        )

        print(f"  Train set(shard) size for worker {t}: ", lengths[t])
        print(f"  Train set(shard) size for worker {t} in batches (with batch_size={batch_size}): ", len(train_loader))

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
    #==================================================================================================================================   
    classes = None

    if dataset_name == "CIFAR10":
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_sets, train_set, test_set, train_loaders, test_loaders, classes 

#=======================================================================================================
# VISUALIZE
#=======================================================================================================
def lookAtImage(index, dataset, classes):
    image, label = dataset[index]
    plt.imshow(image.permute((1, 2, 0)), cmap='gray')
    plt.title(f'Class: {classes[label]}')
    plt.axis('off')
    plt.show()

def plotMetrics(metrics, model_name, dataset_name):
    figure, axes = plt.subplots(1, 2, figsize=(25, 5))

    # Number of epochs
    epochs = range(len(metrics['loss']))

    color = ["#e41a1c", "#377eb8", "#4daf4a", "#e41a1c", "#377eb8", "#4daf4a"]
    linestyle = ["solid", "solid", "solid", "dashed","dashed","dashed"]

    axes[0].plot(epochs, metrics['loss'], marker='o', label='train', linestyle=linestyle[0], color=color[0])
    axes[0].plot(epochs, metrics['v_loss'], marker='o', label='valid', linestyle=linestyle[1], color=color[1])
    axes[0].set_xlabel('Epochs', fontdict = {'fontsize':35})
    #axes[0].set_ylabel('Loss', fontdict = {'fontsize':35})        
    axes[0].set_title(f'Loss {model_name}@{dataset_name}')
    axes[0].legend(loc='best', fontsize=25)
    axes[0].grid()

    axes[1].plot(epochs, metrics['accuracy'], marker='o', label='train', linestyle=linestyle[2], color=color[2])
    axes[1].plot(epochs, metrics['v_accuracy'], marker='o', label='valid', linestyle=linestyle[3], color=color[3])

    axes[1].set_xlabel('Epochs', fontdict = {'fontsize':35})
    #axes[1].set_ylabel('Accuracy', fontdict = {'fontsize':35})        
    axes[1].set_title(f'Accuracy {model_name}@{dataset_name}')
    axes[1].legend(loc='best', fontsize=25)
    axes[1].grid()

    plt.show(figure)
    best = max(metrics['v_accuracy']) * 100
    print(f'Best validation accuracy {best:.2f}%')
    
    figure.tight_layout()
    save_to = f"plots-{model_name}-{dataset_name}.pdf"
    figure.savefig(save_to, bbox_inches='tight')
    print("Image is saved into: ", save_to)

#=======================================================================================================
# OPERATE ON NN PARAMETERS
#=======================================================================================================
def setupToZeroAllParams(model):
    for p in model.parameters(): 
        p.data.zero_()

def setupAllParamsRandomly(model):
    #setupToZeroAllParams(model)

    seed = 12
    torch.manual_seed(seed)

    #torch.nn.init.xavier_normal_(model.parameters)
    #for p in model.parameters():
        #sz = p.data.flatten(0).size()
        #torch.nn.init.xavier_normal_(p.weight)
        #p.data.flatten(0)[:] = 2 * (torch.rand(size = sz) - 0.5)

def setupAllParams(model, params):
    i = 0
    for p in model.parameters():
        p.data.flatten(0)[:] = params[i].flatten(0)
        i += 1

def getAllParams(model):
    params = []
    for p in model.parameters(): 
        params.append(p.data.detach().clone())
    return params

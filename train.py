import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
import argparse

import functions

ap = argparse.ArgumentParser(description='Train.py')
# Command Line ardguments

ap.add_argument('--gpu', dest="gpu", action="store", default="gpu")
ap.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
ap.add_argument('--epochs', dest="epochs", action="store", type=int, default=5)
ap.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001)
ap.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5)
ap.add_argument('--arch', dest="arch", action="store", default="densenet121", type = str)
ap.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120)

pa = ap.parse_args()
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs


trainloader, validationloader, testloader,train_data = functions.data()


model, optimizer, criterion = functions.setup(structure,dropout,hidden_layer1,lr,power)


functions.train_network(model, criterion, optimizer, trainloader,validationloader,epochs,20, power)


functions.save_checkpoint(train_data,epochs,structure,hidden_layer1,dropout,lr,path)


print("The Model is trained")
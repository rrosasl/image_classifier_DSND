import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms.functional as FF
from torchvision import datasets, transforms, models
from collections import OrderedDict
import time
import numpy.random as random
import os
import argparse
from PIL import Image
import utils

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', nargs='*', action='store', help='directory with images', default="flowers/")
parser.add_argument('--save_dir', action='store', help='directory to save checkpoint', default="checkpoint.pth" )
parser.add_argument('--arch', action='store', help='pre trained architecture type', default='vgg16')
parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
parser.add_argument('--epochs', action='store', help='# of training ', type=int, default=3)
parser.add_argument('--learning_rate', action='store', help='learning_rate', type=float, default=0.01)
parser.add_argument('--output_size', action='store', help='# of classes to output', type=int, default=102)

args = parser.parse_args()

data_dir = args.data_dir
train_data, valid_data, test_data, trainloader, validloader, testloader = utils.img_preprocess(data_dir)

criterion, model, optimizer = utils.nn_setup(structure = args.arch, lr = args.learning_rate, output_size = args.output_size)

#utils.train_model(model=model, criterion=criterion, optimizer=optimizer, epochs = args.epochs, loader=trainloader)
#utils.save_checkpoint(path = args.save_dir, data=train_data, structure = args.arch, model = model, optimizer = optimizer, lr = args.learning_rate, epochs=args.epochs, output_size = args.output_size)

utils.train_and_save(path = args.save_dir, model= model, criterion = criterion, optimizer = optimizer, epochs = args.epochs, loader=trainloader, data = train_data, structure = args.arch)


print("done. Your model has been trained and saved. You may go home")

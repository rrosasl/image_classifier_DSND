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
import json
import utils

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', nargs='*', action='store', help='directory with images', default="flowers/")
parser.add_argument('--im_path', default='flowers/test/16/image_06657.jpg', action = 'store', help = 'a sample test image')
parser.add_argument('checkpoint', default='checkpoint.pth', nargs = '*', action = 'store', help = 'The checkpoint for the pre-trained model')
parser.add_argument('--top_k', default = 5, action = 'store', type = int, help = 'How many top categories of prediction you want')

args = parser.parse_args()
data_dir = args.data_dir

train_data, valid_data, test_data, trainloader, validloader, testloader = utils.img_preprocess(data_dir)

model_checkpoint = utils.load_checkpoint(args.checkpoint)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

probs, labels = utils.predict(image_path = args.im_path, model = model_checkpoint, cat_to_name = cat_to_name, topk=5)

for i in range(args.top_k):
    print('probabily: {} - class: {}'.format(probs[i], labels[i]))

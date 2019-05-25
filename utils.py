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
from PIL import Image
import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device # This enables us to use faster processing when available and use the CPU if not

def img_preprocess(data_dir):
    '''return the data separated into training, validaiton, and testing in a way that machines can understand
    i.e after pre-processing'''

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    data_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64,shuffle=True)

    valid_data = datasets.ImageFolder(valid_dir,transform = data_transforms)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64,shuffle = True)

    test_data = datasets.ImageFolder(test_dir, transform = data_transforms)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle=True)
    image_datasets = [train_data,valid_data,test_data]

    return train_data, valid_data, test_data, trainloader, validloader, testloader

def nn_setup(structure='vgg16', lr= 0.02, output_size = 102):
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
        n_input = 9216
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
        n_input = 25088
    elif structure == 'densenet121':
        model = models.densenet121(pretained=True)
        n_input = 1024
    else:
        print('model not recognized, only vgg16, alexnet, and densenet121 possible')

    for param in model.parameters():
        param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(0.5)),
            ('inputs', nn.Linear(n_input, 824)),
            ('relu1', nn.ReLU()),
            ('hidden_1', nn.Linear(824,625)),
            ('relu2', nn.ReLU()),
            ('hidden_2', nn.Linear(625,400)),
            ('relu3', nn.ReLU()),
            ('hidden_3', nn.Linear(400, output_size)),
            ('output', nn.LogSoftmax(dim=1))
            ]))

        criterion =  nn.NLLLoss()
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(),lr)
        model.to(device)

        return criterion, model, optimizer

def train_model(model, criterion, optimizer, epochs, loader):

    count = 0
    for e in range(epochs):
        running_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            count+=1
            if count % 25 == 0: print("training loss (run{}, epoch{})):{}".format(count, e+1, running_loss/count))
        model.train()
    #eval_st = 'training_loss : ' + str(running_loss / count) + 'epochs: ' + str(e) + 'runs : ' + str(count)
    #model_t = model
    #optimizer_t = optimizer
    #return model_t, optimizer_t
def save_checkpoint(path, data, structure, model, optimizer, lr = 0.02, epochs=5, output_size = 102):
    model.class_to_idx = data.class_to_idx
    model.cpu

    if structure == 'alexnet':
        n_input = 9216
    elif structure == 'vgg16':
        n_input = 25088
    elif structure == 'densenet121':
        n_input = 1024

    checkpoint = {'input_size': n_input,
              'output_size': output_size,
              'pre_mod': structure,
              'learning_rate': lr,
              'batch_size': 64,
              'classifier' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)

def train_and_save(path, model, criterion, optimizer, epochs, loader, data, structure, lr = 0.02, output_size = 102):
    # Train
    count = 0
    model.to(device)
    model.train()
    for e in range(epochs):
        running_loss = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            count+=1
            if count % 25 == 0:
                print("training loss (run{}, epoch{})):{}".format(count, e+1, running_loss/count))


    # Save
    model.class_to_idx = data.class_to_idx
    model.cpu
    if structure == 'alexnet':
        n_input = 9216
    elif structure == 'vgg16':
        n_input = 25088
    elif structure == 'densenet121':
        n_input = 1024

    checkpoint = {'input_size': n_input,
              'output_size': output_size,
              'pre_mod': structure,
              'learning_rate': lr,
              'batch_size': 64,
              'classifier' : model.classifier,
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path)
def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    structure = checkpoint['pre_mod']
    #criterion, model, optimizer = nn_setup(structure=checkpoint['pre_mod'], lr= checkpoint['learning_rate'], output_size = checkpoint['output_size'])
    if structure == 'alexnet':
        model = models.alexnet(pretrained=True)
    elif structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretained=True)
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']

    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model

def process_image2(image):
    im = Image.open(image)
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                                    ])
    ima = transform(im)
    return ima

def predict(image_path, model, cat_to_name, topk=5):
    model.cpu()
    model.eval()

    imz = process_image2(image_path)
    imz = imz.unsqueeze_(0)
    imz = imz.float()

    log_ps = model.forward(imz)
    ps = torch.exp(log_ps)
    probs, classes = ps.topk(5,dim=1)

    probs = probs.detach().numpy()[0]
    classes = classes.numpy()

    labels = []
    for cl in classes[0]:
        labels.append(cat_to_name[str(cl)])

    return probs, labels

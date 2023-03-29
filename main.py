from __future__ import division, print_function

import copy
import numpy
import matplotlib.pyplot
import os
import time
import torch
import torch.backends.cudnn
import torch.nn
import torch.optim
import torchvision

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

def show_image(input_data, title=None):
    input_data = input_data.numpy().transpose((1, 2, 0))
    mean = numpy.array([0.485, 0.456, 0.406])
    std = numpy.array([0.229, 0.224, 0.225])

    input_data = std * input_data + mean
    input_data = numpy.clip(input_data, 0, 1)

    matplotlib.pyplot.imshow(input_data)
    if title is not None:
        matplotlib.pyplot.title(title)
    matplotlib.pyplot.show()

if __name__ ==  '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    torch.backends.cudnn.benchmark = True

    data_dir = 'data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
        for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) 
        for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    if torch.cuda.is_available():
        print("Using CUDA-enabled GPU")
        device = torch.device("cuda:0")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    # get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    out = torchvision.utils.make_grid(inputs)

    show_image(out, title=[class_names[x] for x in classes])

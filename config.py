from tqdm import tqdm
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import h5py
import csv
import os
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from torchvision import utils
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# Compatibility with MPS FOR Apple Silicon Macs
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')

                     #Use this if you Nvidia cuda GPU
###->-> device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') <-<-###


MAPS_PATH = 'maps'
SAVE_PATH = 'results/'
NUM_EPOCHS = 100
BATCH_SIZE = 1   #Suggested By the paper ######https://arxiv.org/pdf/1611.07004.pdf######

#Hyperparameters
lr = 0.0002
beta_1 = 0.5
'''The generator is updated via a weighted sum of both the adversarial loss and the L1 loss, 
where the authors of the model recommend a weighting of 100 to 1 in favor of the L1 loss. '''
L1_weight = 100


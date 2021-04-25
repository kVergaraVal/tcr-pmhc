import glob
#!pip install -q sklearn
!pip install PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn import metrics
from sklearn.metrics import f1_score, accuracy_score
from sklearn.metrics import roc_curve, confusion_matrix
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")

# ML architecture
###############################
###    Define network       ###
###############################

print("Initializing network")

# Hyperparameters
input_size = 528
num_classes = 1
learning_rate = 0.01

class Net(nn.Module):
    def __init__(self,  num_classes):
        super(Net, self).__init__()
        self.layers = nn.Sequential(      
        nn.Conv1d(in_channels=7, out_channels=100, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
        nn.BatchNorm1d(100),
        
        nn.Conv1d(in_channels=100, out_channels=100, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm1d(100),

        nn.Flatten(),
        nn.Linear(6600, 128),
        nn.ReLU(),
        nn.Linear(128, 16),
        nn.ReLU(),
        nn.Linear(16,num_classes),
        nn.Sigmoid()
        )

    def forward(self, x):      
        x = self.layers(x)
        return x
    
# Initialize network
net = Net(num_classes=num_classes).to(device)

def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    #print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()
  #print('Reset trainable parameters for model')

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate)

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

###############################
###    Load data            ###
###############################


data_list = []
target_list = []

import glob
for fp in glob.glob("data/train/*input.npz"):
    data = np.load(fp)["arr_0"]
    targets = np.load(fp.replace("input", "labels"))["arr_0"]
    
    data_list.append(data)
    target_list.append(targets)

dataset_raw = np.concatenate(data_list)
dataset_ros = dataset_raw[:,:,20:27]

#min-max scaling
#for i in range(0,len(dataset[0,0,:])):
#  min = np.min(dataset[:,:,i])
#  max = np.max(dataset[:,:,i])
#  dataset[:,:,i] = (dataset[:,:,i]-min)/(max-min)
target = np.concatenate(target_list)
nsamples, nx, ny = dataset_ros.shape
print("Training set shape:", nsamples,nx,ny)

p_pos = len(target[target == 1])/len(target)*100
print("Percent positive samples in dataset:", p_pos)


# Fourier Transform of per-residue Rosetta scoring terms

MHC = dataset_ros[:,0:180,:]
peptide = dataset_ros[:,180:192,:]
TCR = dataset_ros[:,192:421,:]

len_MHC = len(MHC[0,:,0])
len_peptide = len(peptide[0,:,0])
len_TCR = len(TCR[0,:,0])

num_attr = len(dataset_ros[0,0,:])
num_samp = len(dataset_ros[:,0,0])

zeropad_MHC = np.zeros((num_samp,256-len_MHC,num_attr))
zeropad_peptide = np.zeros((num_samp,16-len_peptide,num_attr))
zeropad_TCR = np.zeros((num_samp,256-len_TCR,num_attr))

print(np.shape(MHC))
print(np.shape(zeropad_MHC))

MHC_padded = np.concatenate((MHC,zeropad_MHC),axis=1)
peptide_padded = np.concatenate((peptide,zeropad_peptide),axis=1)
TCR_padded = np.concatenate((TCR,zeropad_TCR),axis=1)
print(np.shape(MHC_padded))

nyq_lim_MHC = len(MHC_padded[0,:,0])
nyq_lim_pep = len(peptide_padded[0,:,0])
nyq_lim_TCR = len(TCR_padded[0,:,0])

MHC_fft = np.zeros(np.shape(MHC_padded))
peptide_fft = np.zeros(np.shape(peptide_padded))
TCR_fft = np.zeros(np.shape(TCR_padded))

for k in range(0,num_samp):
  for j in range(0,num_attr):
    MHC_fft_vec = np.abs(sc.fft.fft(MHC_padded[k,:,j]))
    MHC_fft_vec = MHC_fft_vec[0:int(nyq_lim_MHC)]
    MHC_fft_vec = [2*elem/(2*nyq_lim_MHC) for elem in MHC_fft_vec]
    MHC_fft[k,:,j] = MHC_fft_vec

    pep_fft_vec = np.abs(sc.fft.fft(peptide_padded[k,:,j]))
    pep_fft_vec = pep_fft_vec[0:int(nyq_lim_pep)]
    pep_fft_vec = [2*elem/(2*nyq_lim_pep) for elem in pep_fft_vec]
    peptide_fft[k,:,j] = pep_fft_vec

    TCR_fft_vec = np.abs(sc.fft.fft(TCR_padded[k,:,j]))
    TCR_fft_vec = TCR_fft_vec[0:int(nyq_lim_TCR)]
    TCR_fft_vec = [2*elem/(2*nyq_lim_TCR) for elem in TCR_fft_vec]
    TCR_fft[k,:,j] = TCR_fft_vec


dataset_fft = np.concatenate((MHC_fft,peptide_fft,TCR_fft),axis=1)

# make the data set into one dataset that can go into dataloader
dataset_ds = []
for i in range(len(dataset_fft)):
    dataset_ds.append([np.transpose(dataset_fft[i]), target[i]])

print(np.shape(dataset_fft))

dataset_ds_raw = []
for i in range(len(dataset_raw)):
    dataset_ds_raw.append([np.transpose(dataset_raw[i]), target[i]])
    
# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")

###############################
###     TRAIN WITH CV       ###
###############################

print("Training")

# Configuration options
k_folds = 4
num_epochs = 30
bat_size = 64
  
# For fold results
#results = {}
  
# Set fixed random number seed
torch.manual_seed(42)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=k_folds, shuffle=True)
    
# Start print
print('--------------------------------')


# K-fold Cross Validation model evaluation
val_loss_mean = []
val_MCC_mean = []
for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset_ds)):
  net.apply(reset_weights)
  train_acc, train_loss = [], []
  valid_acc, valid_loss = [], []
  losses = []
  val_losses = []
  best_MCC = 0
  # Print
  print(f'FOLD {fold}')
  print('--------------------------------')
    
  # Sample elements randomly from a given list of ids, no replacement.
  train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
  test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
  # Define data loaders for training and testing data in this fold
  train_ldr = torch.utils.data.DataLoader(
                    dataset_ds, 
                    batch_size=bat_size, sampler=train_subsampler)
  val_ldr = torch.utils.data.DataLoader(
                    dataset_ds,
                    batch_size=bat_size, sampler=test_subsampler)

  for epoch in range(num_epochs):
      cur_loss = 0
      val_loss = 0
    
      net.train()
      train_preds, train_targs = [], [] 
      for batch_idx, (data, target) in enumerate(train_ldr):
          X_batch =  data.float().detach().requires_grad_(True)
          target_batch = torch.tensor(np.array(target), dtype = torch.float).unsqueeze(1)
        
          optimizer.zero_grad()
          output = net(X_batch)
        
          batch_loss = criterion(output, target_batch)
          batch_loss.backward()
          optimizer.step()
        
          preds = np.round(output.detach().cpu())
          train_targs += list(np.array(target_batch.cpu()))
          train_preds += list(preds.data.numpy().flatten())
          cur_loss += batch_loss.detach()

      losses.append(cur_loss / len(train_ldr.dataset))        
    
      net.eval()
      ### Evaluate validation
      val_preds, val_targs = [], []
      with torch.no_grad():
          for batch_idx, (data, target) in enumerate(val_ldr): ###
              x_batch_val = data.float().detach()
              y_batch_val = target.float().detach().unsqueeze(1)
            
              output = net(x_batch_val)
            
              val_batch_loss = criterion(output, y_batch_val)
            
              preds = np.round(output.detach())
              val_preds += list(preds.data.numpy().flatten()) 
              val_targs += list(np.array(y_batch_val))
              val_loss += val_batch_loss.detach()
            
          val_losses.append(val_loss / len(val_ldr.dataset))
          print("\nEpoch:", epoch+1)
        
          train_acc_cur = accuracy_score(train_targs, train_preds)  
          valid_acc_cur = accuracy_score(val_targs, val_preds) 

          train_acc.append(train_acc_cur)
          valid_acc.append(valid_acc_cur)

          MCC_val = matthews_corrcoef(val_targs, val_preds)
          print("Training loss:", losses[-1].item(), "Validation loss:", val_losses[-1].item(), end = "\n")
          print("MCC Train:", matthews_corrcoef(train_targs, train_preds), "MCC val:", MCC_val)
      
      cur_MCC = MCC_val
      if cur_MCC > best_MCC:
        #best_model = copy.deepcopy(model)  # Will work
        best_MCC = cur_MCC
        torch.save(net.state_dict(), 'src/model.pt')
  
  break
  val_loss_mean += [val_losses[-1].item()]
  val_MCC_mean += [MCC_val]
print('--------------------')
print('Mean val loss: ', np.mean(val_loss_mean))
print('Mean val MCC: ', np.mean(val_MCC_mean))   


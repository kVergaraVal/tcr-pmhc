import argparse
import glob, os, tempfile, zipfile
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
from sklearn.model_selection import KFold
from sklearn.metrics import matthews_corrcoef
# Needed to predict
from model import Net

parser = argparse.ArgumentParser()
parser.add_argument('--input-zip')
args = parser.parse_args()
print(args)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device (CPU/GPU):", device)
#device = torch.device("cpu")

model_param = "scr/model.pt"
out_path = "predictions.csv"

# Load files
filenames = []
dfs = []
with tempfile.TemporaryDirectory() as tmpdir:
    with zipfile.ZipFile(args.input_zip) as zip:
        files = [file for file in zip.infolist() if file.filename.endswith(".npz")]
        for file in files:
            zip.extract(file, tmpdir)
            
        # Load npz files
        data_list = []

        for fp in glob.glob(tmpdir + "/*input.npz"):
            data = np.load(fp)["arr_0"]

            data_list.append(data)

#data_list = []
#target_list = []

#data_list.append(data)
#target_list.append(targets)

X_values = np.concatenate(data_list[:])
#y_values = np.concatenate(target_list[:])

X_ros = X_values[:,:,20:27]

# Fourier Transform of per-residue Rosetta scoring terms

MHC = X_ros[:,0:180,:]
peptide = X_ros[:,180:192,:]
TCR = X_ros[:,192:421,:]

len_MHC = len(MHC[0,:,0])
len_peptide = len(peptide[0,:,0])
len_TCR = len(TCR[0,:,0])

num_attr = len(X_ros[0,0,:])
num_samp = len(X_ros[:,0,0])

zeropad_MHC = np.zeros((num_samp,256-len_MHC,num_attr))
zeropad_peptide = np.zeros((num_samp,16-len_peptide,num_attr))
zeropad_TCR = np.zeros((num_samp,256-len_TCR,num_attr))
MHC_padded = np.concatenate((MHC,zeropad_MHC),axis=1)
peptide_padded = np.concatenate((peptide,zeropad_peptide),axis=1)
TCR_padded = np.concatenate((TCR,zeropad_TCR),axis=1)

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

dataset_ds = []
for i in range(len(dataset_fft)):
  dataset_ds.append([np.transpose(dataset_fft[i]), y_values[i]])

# load model

net.load_state_dict(torch.load(model_param))

# model predict
Outputs = []
for k in range(0,len(dataset_ds)):
  input = np.reshape(dataset_ds[k][0],(1,7,528))
  input = torch.FloatTensor(input)

  output = net(input)
  output = output.detach().numpy()
  output2 = output.tolist()
  Outputs += output2[0]
  bin_Outputs = [int(round(i,0)) for i in Outputs]

# output file
rows = [i for i in range(0,len(bin_Outputs))]
output_df = pd.DataFrame(bin_Outputs, index=rows, columns=['prediction'])
outfilename = out_path + 'predictions.csv'
outfile = open(outfilename, 'wb')
output_df.to_csv(outfilename, sep = ',', encoding = 'utf-8')
outfile.close()

#targs = y_values.tolist()
#acc_final = accuracy_score(targs, bin_Outputs)  
#MCC_final = matthews_corrcoef(targs, bin_Outputs)
#print("Confusion matrix final:", confusion_matrix(targs, bin_Outputs), sep = "\n")

#acc_final, MCC_final = prediction(in_path, model_param, out_path)
#print('test accuracy: ', acc_final)
#print('test MCC: ', MCC_final)

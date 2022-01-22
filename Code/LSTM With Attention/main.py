import torch
from torch import nn
import torch.nn.functional as F

import os
import pickle
import numpy as np
import math

# from dataset import Dataset
from model import Encoder, Decoder, Seq2Seq

import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from train import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("configured device: ", device)

path = "../../../datasets/data_preprocessed_python"

#getting the encoder layer with below units
enc = Encoder(40, 256, 1).to(device)

#getting the decoder layer
dec = Decoder(256, 1).to(device)

#connecting them with seq2seq and getting the final model out
model = Seq2Seq(enc, dec).to(device)

model_params = {
    "MODEL": "Soft-attention", # Self-attention, Soft-attention
    "STIM": "Arousal", #Arousal/ Valence/ All
    "BATCH_SIZE": 12,  # batch size
    "EPOCHS": 15,  # number of training epochs
    "LEARNING_RATE": 0.001,  # learning rate
}


trainer(path = path, model = model, model_params=model_params, device = device,)
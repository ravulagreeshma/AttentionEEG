import torch
from torch import nn
import torch.nn.functional as F

import os
import pickle
import numpy as np
import math

#defining the models and their architectures
class Encoder(nn.Module):

#this class will initialize the models with the desired architecture
    def __init__(self, input_size, embed_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        
# defining lstm and using bidirectional LSTM'S
        self.lstm = nn.LSTM(input_size, embed_size, n_layers,
                          dropout=dropout, bidirectional=True)
# feed forward layer;s
    def forward(self, x):       
        output, (hn, cn) = self.lstm(x)
        
# sum bidirectional outputs
        output = (output[:, :, :self.embed_size] +
                   output[:, :, self.embed_size:])
        return output, hn
#encoder output is returned and passed to the decoder


class Attn_(nn.Module):
    def __init__(self, hidden_size):
        super(Attn_, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)
        

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  
        attn_energies = self.score(h, encoder_outputs)      
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    
    def score(self, hidden, encoder_outputs):
   
        temp = torch.cat([hidden, encoder_outputs], dim=2)
        energy = F.relu(self.attn(temp))
        energy = energy.transpose(1, 2)  
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  
        energy = torch.bmm(v, energy)  
        return energy.squeeze(1)  


#Main Self attention class 
class Attn(nn.Module):
    def __init__(self, h_dim,c_num):
        super(Attn_, self).__init__()
        self.h_dim = h_dim
        self.v = nn.Parameter(torch.rand(h_dim))
        self.out = nn.Linear(self.h_dim, c_num)

        self.main = nn.Sequential(
            nn.Linear(h_dim, c_num),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

#Actual process
    def forward(self, hidden , encoder_outputs):
        b_size = encoder_outputs.size(0)

#atten_energies are calculated using encoder outputs and hidden layers
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim)) 


#Multiplying q*k
        attn_applied = torch.bmm(attn_ene.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0)) 
        
#scaling:sqrt(size(h_dim))     
        output=attn_applied[0]/math.sqrt(self.v.size(0))
        
#softmax
        output = F.log_softmax(self.out(output[0]), dim=1).unsqueeze(2)
        return output
    
#Decoder class 
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        
#         self.dropout = nn.Dropout(dropout, inplace=True)
        
        ## attention layer
        self.attention = Attn_(hidden_size)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size * 2, output_size) 
        self.sig = nn.Sigmoid()

    def forward(self, last_hidden, encoder_outputs):

# Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
#context vector=attention weights ,ecnoder outputs

#[q*k]*v
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  
        context = context.transpose(0, 1)  
        output = self.fc(last_hidden.view(-1, 2*self.hidden_size))
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
#output = F.log_softmax(output, dim=1)
        return self.sig(output), attn_weights

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src):

        encoder_output, hidden = self.encoder(src) 
        output, attn_weights = self.decoder(hidden, encoder_output)

        return output
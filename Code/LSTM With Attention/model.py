import torch
from torch import nn
import torch.nn.functional as F

# import os
# import pickle
# import numpy as np
import math

from attention_module import *

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
    
#Decoder class 
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size, attention_type,
                 dropout=0.2):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention_type = attention_type
#         self.dropout = nn.Dropout(dropout, inplace=True)
        
        ## attention layer
        if self.attention_type == "Self-attention":
            self.attention = SelfAttention(hidden_size)
        elif self.attention_type == "Soft-attention":
            self.attention = SoftAttention(hidden_size)
        elif self.attention_type == "Hard-attention":
            self.attention = HardAttention(hidden_size)
        else:
            pass
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
    def __init__(self, input_dim, hidden_dim, output_dim,  attention_type):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(hidden_dim, output_dim, attention_type)

    def forward(self, src):

        encoder_output, hidden = self.encoder(src) 
        output, attn_weights = self.decoder(hidden, encoder_output)

        return output
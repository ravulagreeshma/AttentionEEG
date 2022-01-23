import torch
from torch import nn
import torch.nn.functional as F
import math

from attention_module import *

#defining the models and their architectures
class Encoder(nn.Module):

#this class will initialize the models with the desired architecture
    def __init__(self, input_dim, hidden_dim, output_dim, attention_type, n_layers=1, lstm_dropout=0.5): # decoder_dropout=0.2
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
# defining lstm and using bidirectional LSTM'S
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                          dropout=lstm_dropout, bidirectional=True)
        
        
        self.attention_type = attention_type
#         self.dropout = nn.Dropout(dropout, inplace=True)
        
        ## attention layer
        if self.attention_type == "Self-attention":
            self.attention = SelfAttention(hidden_dim)
        elif self.attention_type == "Soft-attention":
            self.attention = SoftAttention(hidden_dim)
        elif self.attention_type == "Hard-attention":
            self.attention = HardAttention(hidden_dim)
        elif self.attention_type == "Hierarchical-attention":
            self.attention = HierarchicalAttention(hidden_dim, output_dim)    
        else:
            raise ValueError('Undefined attention type')
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out = nn.Linear(hidden_dim * 2, output_dim) 
        self.sig = nn.Sigmoid()
        
# feed forward layer;s
    def forward(self, x):       
        output, (hn, cn) = self.lstm(x)
        
# sum bidirectional outputs
        output = (output[:, :, :self.hidden_dim] +
                   output[:, :, self.hidden_dim:])
        
        encoder_outputs = output
        last_hidden = hn
    
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
#context vector=attention weights ,ecnoder outputs

#[q*k]*v
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  
        context = context.transpose(0, 1)  
        output = self.fc(last_hidden.view(-1, 2*self.hidden_dim))
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
#output = F.log_softmax(output, dim=1)

        self.attn_weights = attn_weights

        return self.sig(output)
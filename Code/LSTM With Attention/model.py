import torch
from torch import nn
import torch.nn.functional as F
import math

from attention_module import *

#defining the models and their architectures
class LSTM_Encoder(nn.Module):

#this class will initialize the models with the desired architecture
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_merge_mode, attention_type, n_layers=1, lstm_dropout=0.5): # decoder_dropout=0.2
        super(LSTM_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lstm_merge_mode = lstm_merge_mode
        
# defining lstm and using bidirectional LSTM'S
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                          dropout=lstm_dropout, bidirectional=True)
        
        
        self.attention_type = attention_type
#         self.dropout = nn.Dropout(dropout, inplace=True)
        
        ## attention layer
        if self.attention_type == "Self-attention":
            self.attention = SelfAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Soft-attention":
            self.attention = SoftAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Hard-attention":
            self.attention = HardAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Hierarchical-attention":
            self.attention = HierarchicalAttention(hidden_dim, output_dim, lstm_merge_mode)    
        else:
            raise ValueError('Undefined attention type')
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        if lstm_merge_mode == 'sum' :
            self.out = nn.Linear(hidden_dim * 2, output_dim) 
        if lstm_merge_mode == 'concat' :
            self.out = nn.Linear(hidden_dim * 3, output_dim) 
        
        self.sig = nn.Sigmoid()
        
# feed forward layer;s
    def forward(self, x):  

        output, (hn, cn) = self.lstm(x)
        
# sum bidirectional outputs
        if self.lstm_merge_mode == 'sum':
            output = (output[:, :, :self.hidden_dim] +
                       output[:, :, self.hidden_dim:])
            # print(output.shape) [8064, 12, 128]

# concatenate bidirectional outputs
        if self.lstm_merge_mode == 'concat':
            output = output # same as output = torch.cat([output[:, :, :self.hidden_dim], output[:, :, self.hidden_dim:]], dim=2)
            #print(output.shape) [8064, 12, 256]
    
        encoder_outputs = output
        last_hidden = hn
    
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) #[12, 1, 8064]
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
    
class CNN_LSTM_Encoder(nn.Module):

#this class will initialize the models with the desired architecture
    def __init__(self, input_dim, hidden_dim, output_dim, lstm_merge_mode, attention_type, n_layers=1, lstm_dropout=0.5): # decoder_dropout=0.2
        super(CNN_LSTM_Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.activation = nn.ReLU()
        self.conv1 = nn.Sequential(    nn.Conv2d(40, 32,       kernel_size=(123,5),   padding=(0,0), stride=(1,1))  ,  self.activation )
        self.conv2 = nn.Sequential(    nn.Conv2d(32, 64,        kernel_size=(1,3) ,  padding=(0,0), stride=(1,1))  ,  self.activation )
        self.conv3 = nn.Sequential(    nn.Conv2d(64, 128,       kernel_size=(1,3) ,  padding=(0,0), stride=(1,1))  ,  self.activation )
        self.lstm = nn.LSTM(128, 128, num_layers=1, dropout=0.5, bidirectional=True)
        
        self.lstm_merge_mode = lstm_merge_mode
        
        
        self.attention_type = attention_type
#         self.dropout = nn.Dropout(dropout, inplace=True)
        
        ## attention layer
        if self.attention_type == "Self-attention":
            self.attention = SelfAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Soft-attention":
            self.attention = SoftAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Hard-attention":
            self.attention = HardAttention(hidden_dim, lstm_merge_mode)
        elif self.attention_type == "Hierarchical-attention":
            self.attention = HierarchicalAttention(hidden_dim, output_dim, lstm_merge_mode)    
        else:
            raise ValueError('Undefined attention type')
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        
        if lstm_merge_mode == 'sum' :
            self.out = nn.Linear(hidden_dim * 2, output_dim) 
        if lstm_merge_mode == 'concat' :
            self.out = nn.Linear(hidden_dim * 3, output_dim) 
        
        self.sig = nn.Sigmoid()
        
# feed forward layer;s
    def forward(self, x):  
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = (x.squeeze(2)).permute((2, 0, 1))
        output, (hn, cn) = self.lstm(x)
        
# sum bidirectional outputs
        if self.lstm_merge_mode == 'sum':
            output = (output[:, :, :self.hidden_dim] +
                       output[:, :, self.hidden_dim:])
            # print(output.shape) [8064, 12, 128]

# concatenate bidirectional outputs
        if self.lstm_merge_mode == 'concat':
            output = output # same as output = torch.cat([output[:, :, :self.hidden_dim], output[:, :, self.hidden_dim:]], dim=2)
            #print(output.shape) [8064, 12, 256]
    
        encoder_outputs = output
        last_hidden = hn
    
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) #[12, 1, 8064]
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
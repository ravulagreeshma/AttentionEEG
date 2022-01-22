import torch
from torch import nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        print("SELF ATTENTION")
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

class SoftAttention(nn.Module):
    def __init__(self, encoder_hidden_dim):
        super().__init__()
        print("SOFT ATTENTION")
        # The input dimension will the the concatenation of
        # encoder_hidden_dim (hidden) and  decoder_hidden_dim(encoder_outputs)
        self.attn_hidden_vector = nn.Linear(encoder_hidden_dim *2,encoder_hidden_dim)
 
        # We need source len number of values for n batch as the dimension
        # of the attention weights. The attn_hidden_vector will have the
        # dimension of [source len, batch size, decoder hidden dim]
        # If we set the output dim of this Linear layer to 1 then the
        # effective output dimension will be [source len, batch size]
        self.attn_scoring_fn = nn.Linear(encoder_hidden_dim, 1, bias=False)
 
    def forward(self, hidden, encoder_outputs):
        # hidden = [1, batch size, decoder hidden dim]
        src_len = encoder_outputs.shape[0]
 
        # We need to calculate the attn_hidden for each source words.
        # Instead of repeating this using a loop, we can duplicate
        # hidden src_len number of times and perform the operations.
        hidden = hidden.repeat(src_len, 1, 1).transpose(0,1)

        encoder_outputs = encoder_outputs.transpose(0, 1)#.permute(1,0,2)
        # Calculate Attention Hidden values
#         print(hidden.size(),encoder_outputs.size())
        #torch.Size([8064, 2, 256]) torch.Size([2, 8064, 256])
        dup=torch.cat((hidden, encoder_outputs), dim=2)

        ## weighted sum
        # dup=hidden.bmm(encoder_outputs.transpose(0, 1))

        attn_hidden = torch.tanh(self.attn_hidden_vector(dup))
        # attn_hidden = 

        # Calculate the Scoring function. Remove 3rd dimension.
        attn_scoring_vector = self.attn_scoring_fn(attn_hidden).squeeze(2)
 
        # The attn_scoring_vector has dimension of [source len, batch size]
        # Since we need to calculate the softmax per record in the batch
        # we will switch the dimension to [batch size,source len]
        attn_scoring_vector = attn_scoring_vector.permute(1, 0)
 
        # Softmax function for normalizing the weights to
        # probability distribution
        return F.softmax(attn_scoring_vector, dim=1).unsqueeze(0).permute(2,0,1)
    
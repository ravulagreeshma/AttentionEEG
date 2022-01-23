import torch
from model import Encoder
from train import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Configured device: ", device)

path = "../../../data_preprocessed_python"

model_params = {
    "MODEL": "Hard-attention", # Self-attention, Soft-attention, Hard-attention, #Hierarchical-attention
    "STIM": "Arousal", #Arousal/ Valence/ All
    "BATCH_SIZE": 12,  # batch size
    "EPOCHS": 2,  # number of training epochs
    "LEARNING_RATE": 0.001,  # learning rate
}

#getting the encoder layer with below units
input_dim = 40
hidden_dim = 128
output_dim = 1
# enc = Encoder(40, 256, 1).to(device)

# #getting the decoder layer
# dec = Decoder(256, 1).to(device)

#connecting them with seq2seq and getting the final model out
model = Encoder(input_dim, hidden_dim, output_dim, model_params["MODEL"] ).to(device)

trainer(path = path, model = model, model_params=model_params, device = device,)
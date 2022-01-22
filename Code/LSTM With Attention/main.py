import torch
from model import Encoder, Decoder, Seq2Seq
from train import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Configured device: ", device)

path = "../../../datasets/data_preprocessed_python"

#getting the encoder layer with below units
enc = Encoder(40, 256, 1).to(device)

#getting the decoder layer
dec = Decoder(256, 1).to(device)

#connecting them with seq2seq and getting the final model out
model = Seq2Seq(enc, dec).to(device)

model_params = {
    "MODEL": "Self-attention", # Self-attention, Soft-attention
    "STIM": "Arousal", #Arousal/ Valence/ All
    "BATCH_SIZE": 12,  # batch size
    "EPOCHS": 15,  # number of training epochs
    "LEARNING_RATE": 0.001,  # learning rate
}


trainer(path = path, model = model, model_params=model_params, device = device,)
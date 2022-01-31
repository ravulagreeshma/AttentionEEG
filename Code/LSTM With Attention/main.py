import torch
from model import LSTM_Encoder, CNN_LSTM_Encoder
from train import trainer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Configured device: ", device)

path = "../../../data_preprocessed_python"

model_params = {
    "DATA" : "eeg", # eeg / spectrogram
    "LSTM_MERGE_MODE" : 'concat', # concat / sum
    "STIM": "Valence", #Arousal / Valence
    "MODEL": "Self-attention", # Self-attention, Soft-attention, Hard-attention, #Hierarchical-attention
    "BATCH_SIZE": 12,  # batch size
    "EPOCHS": 15,  # number of training epochs
    "LEARNING_RATE": 0.001,  # learning rate
}

#getting the encoder layer with below units
input_dim = 40
hidden_dim = 128
output_dim = 1

# enc = Encoder(40, 256, 1).to(device)
# dec = Decoder(256, 1).to(device)

if model_params["DATA"] == "eeg":
    model = LSTM_Encoder(input_dim, hidden_dim, output_dim, model_params["LSTM_MERGE_MODE"], model_params["MODEL"] ).to(device)
if model_params["DATA"] == "spectrogram":
    model = CNN_LSTM_Encoder(input_dim, hidden_dim, output_dim, model_params["LSTM_MERGE_MODE"], model_params["MODEL"] ).to(device)

trainer(path = path, ds = model_params["DATA"], model = model, model_params=model_params, device = device,)
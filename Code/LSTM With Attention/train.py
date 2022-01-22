#Training the model
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from dataset import Dataset
from torch.utils.data import DataLoader

import warnings
warnings.filterwarnings("ignore")
from utils import classification_report

# rich: for a better display on terminal
from rich.table import Column, Table
from rich import box
from rich.console import Console

console = Console(record=True)

# training logger to log training progress
logger = Table(
    Column("Epoch", justify="center"),
    Column("Train Loss", justify="center"),
    Column("Val Loss", justify="center"),
    title="Status",
    pad_edge=False,
    box=box.ASCII,
)

def train(model, epoch, train_loader, val_loader,  optimizer, loss_fn, device, model_params):

    model.train()
    train_loss = 0

    ## training bathces in gpu
    for i, batch in enumerate(train_loader):
        data = batch['data'].permute(2, 0, 1).to(device)
        label = batch['label'].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, label)
            
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    ## evaluating the trained model on validation set
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):

            data = batch['data'].permute(2, 0, 1).cuda()
            label = batch['label'].cuda()
            output = model(data)
            loss = loss_fn(output, label)
            
            val_loss += loss.item()
    
    logger.add_row(str(epoch), str((model_params["WEIGHT_TRAIN_LOSS"]*train_loss)/len(train_loader)),str((model_params["WEIGHT_VAL_LOSS"]*val_loss)/len(val_loader)))
    console.print(logger)
    # print('Epoch : {} train_loss : {} val_loss : {}'.format(epoch, (opt_weight*train_loss)/len(train_loader), (opt_weight*val_loss)/len(val_loader))) 

def trainer(
    path, model, model_params, device
): #  output_dir="outputs", << to save model

    """
    trainer

    """

    # logging
    console.log(f"""[Model]: {model_params["MODEL"]}...\n""")

    model = model.to(device)

    # logging
    console.log(f"[Data]: Reading data...\n")
    console.log(f"""[STIM]: {model_params["STIM"]}\n""")

    # Importing the raw dataset
    dataset = Dataset(path, model_params["STIM"])
    
    # Creation of Dataset and Dataloader
    # Defining the train size. So 80% of the data will be used for training and the rest for validation.
    torch.manual_seed(1)
    
    indices = torch.randperm(len(dataset)).tolist()
    train_ind = int(0.8 * len(dataset))
    train_set = torch.utils.data.Subset(dataset, indices[:train_ind])
    val_set = torch.utils.data.Subset(dataset, indices[train_ind:])

    console.log(f"FULL Dataset: {len(dataset)}")
    console.log(f"TRAIN Dataset: {len(train_set)}")
    console.log(f"VAL Dataset: {len(val_set)}\n")

    # Creating the Training and Validation dataset for further creation of Dataloader
    # Defining the parameters for creation of dataloaders
    train_params = {
        "batch_size": model_params["BATCH_SIZE"],
        "shuffle": True,
        "pin_memory": True,
    }

    val_params = {
        "batch_size": model_params["BATCH_SIZE"],
        "shuffle": True,
        "pin_memory": True,
    }

    # Creation of Dataloaders for training and validation. 
    train_loader = DataLoader(train_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    # Defining the optimizer and loss function that will be used to tune the weights of the network in the training session.
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=model_params["LEARNING_RATE"])
    loss_fn = nn.BCELoss() 
    
    if model_params["MODEL"] == "Self-attention" or model_params["MODEL"] == "Hierarchical-attention":
        model_params["WEIGHT_TRAIN_LOSS"] = model_params["WEIGHT_VAL_LOSS"] = -0.001
        model_params["BEST_CLASS_WEIGHTS"] = [10, 8, 94, 48]
    elif model_params["MODEL"] == "Soft-attention":
        model_params["WEIGHT_TRAIN_LOSS"] = -0.002
        model_params["WEIGHT_VAL_LOSS"] = -0.001
        model_params["BEST_CLASS_WEIGHTS"] = [9.5, 0.8, 9.5, 5.3]
    elif model_params["MODEL"] == "Hard-attention":
        model_params["WEIGHT_TRAIN_LOSS"] = 1
        model_params["WEIGHT_VAL_LOSS"] = 1
        model_params["BEST_CLASS_WEIGHTS"] = [1.5, 1.35, 1.35, 1.35]     
    else:
        pass

    
    # print(model_params)
    # Training loop
    console.log(f"[Initiating Training]...\n")

    for epoch in range(model_params["EPOCHS"]):
        train(model, epoch, train_loader, val_loader,  optimizer, loss_fn, device, model_params)
        
#     console.log(f"[Saving Model]...\n")
#     # Saving the model after training
#     # path = os.path.join(output_dir, "model_files")

    evaluate(model, val_loader, model_params["BEST_CLASS_WEIGHTS"])

def evaluate(model, loader, best_class_weights):
    #Calculating the metrics
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for i, batch in enumerate(loader):

            data = batch['data'].permute(2, 0, 1).cuda()
            label = batch['label']
            output = model(data)
            fin_targets.append(np.asarray(label.numpy(),dtype=np.int))
            fin_outputs.append(np.asarray((output.cpu().detach().numpy()>0.5), dtype=np.int))
    acc,precision,recall,f1score=classification_report(fin_outputs,fin_targets,best_class_weights)
    print('Accuracy : {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1score: {}'.format(f1score))
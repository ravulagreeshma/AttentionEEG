import torch
import os
import pickle
import numpy as np

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path, stim):
        _, _, filenames = next(os.walk(path))
        filenames = sorted(filenames)
        all_data = []
        all_label = []
        for dat in filenames:
            temp = pickle.load(open(os.path.join(path,dat), 'rb'), encoding='latin1')
            all_data.append(temp['data'])
            
            if stim == "Valence":
                all_label.append(temp['labels'][:,:1])
            elif stim == "Arousal":
                all_label.append(temp['labels'][:,1:2]) # Arousal
            else:
                all_label.append(temp['labels'][:,:2]) # All
                
        self.data = np.vstack(all_data)
        self.label = np.vstack(all_label)
        del temp, all_data, all_label

    def __len__(self):
        return self.data.shape[0]

   
    def __getitem__(self, idx):
        single_data = self.data[idx]
        single_label = self.label[idx].astype(float)
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }

        return batch

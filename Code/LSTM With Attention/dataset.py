import torch
import os
import pickle
import numpy as np
from scipy import signal
from IPython.display import clear_output
import matplotlib.pyplot as plt

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
        single_label = (self.label[idx] > 5).astype(float)
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }

        return batch

class DatasetSpectogram(torch.utils.data.Dataset):
    
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
        print(self.data.shape)
        self.data_spectogram = self.spectogram(self.data)
        # print(self.data_spectogram.shape)
        self.label = np.vstack(all_label)
        del temp, all_data, all_label

    def __len__(self):
        return self.data.shape[0]

   
    def __getitem__(self, idx):
        single_data = self.data[idx]
        
        single_label = (self.label[idx] > 5).astype(float)
        
        batch = {
            'data': torch.Tensor(single_data),
            'label': torch.Tensor(single_label)
        }

        return batch
    
    def spectogram(self, data):
        fs = 512
        nperseg =64
        noverlap = None
        window = 'hann'
        data_spectogram = np.zeros((data.shape[0], data.shape[1], 33, 253))
        
        N = data[1,1,:].shape[0]
        time_0 = np.arange(N) / float(fs)
        
        fig,ax = plt.subplots(41,1,figsize=(20,100))
        for m in range(data.shape[0]):
            print("done1")
            for channel in range(data.shape[1]):
                print("done2")
                x = data[m,channel,:]

                # clear_output(wait=True)
                #y axis = fs/2
                #nperseg = width of yout chunk on the graph
                #15 temporal bins per segment
                f, t, Zxx = signal.stft(x, fs = fs, nperseg=nperseg,noverlap=noverlap)
                print("done3")
                print(Zxx.shape)
                data_spectogram[m,channel,:,:] = np.abs(Zxx) #or np.abs(Zxx)
                #time.sleep(1)  
                ax[channel].pcolormesh(t, f, np.abs(Zxx), vmin=0, vmax=np.max(x),shading='auto') #np.log(np.abs(Zxx)), vmin=np.min(x)
                ax[channel].set_ylabel(str(channel))
                # ax[channel].plot(time_0,x)
                plt.show()
                
            # ax[40].plot(time_0,x)
            # plt.show()
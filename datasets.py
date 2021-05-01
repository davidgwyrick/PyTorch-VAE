import glob, os, sys
import pandas as pd
import numpy as np
import h5py

from PIL import Image
from functools import partial
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# sys.path.insert(0, os.path.join(os.path.expanduser('~/Research/Github/'),'PyTorch-VAE'))

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
    
class WCDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_name = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],self.data_paths.iloc[idx,2])
        sample = pil_loader(img_name)
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

class WCShotgunDataset(Dataset):
    def __init__(self, csv_file, N_fm, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.N_fm = N_fm

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample=[]
        for n in range(self.N_fm):
            img_name = eval(self.data_paths['FileName'][idx])[n]
            img_path = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
        sample = torch.cat(sample,dim=0)
        return sample

class WC3dDataset(Dataset):
    def __init__(self, csv_file, N_fm, root_dir, transform=None):
        
        self.data_paths = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.N_fm = N_fm

    def __len__(self):
        return(len(self.data_paths))
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample=[]
        for n in range(self.N_fm):
            img_name = self.data_paths['N_{:02d}'.format(n)].iloc[idx]
            img_path = os.path.join(self.root_dir,self.data_paths.iloc[idx,1],img_name)
            img = pil_loader(img_path)
            if self.transform:
                img = self.transform(img)
            sample.append(img)
        sample = torch.cat(sample,dim=0).unsqueeze(0) # To put into B x C x D x H x W format
        return sample

class WFDataset(Dataset):
    def __init__(self, h5_file, root_dir,xval='train',transform=None):

        #Read in h5 file
        with h5py.File(os.path.join(root_dir,h5_file),'r') as h5file:
            #Save calculated AR matrices and transition matrix
            dfof_list = list(h5file['dfof_list'])
        
        #Take the first 8400 frames (14minutes) of each session 
        if xval == 'train':
            train_list = [data[:,:,:8400] for data in dfof_list]
            data = np.concatenate(train_list)
        elif xval == 'test':
            test_list = [data[:,:,8400:] for data in dfof_list]
            data = np.concatenate(test_list)
        else:
            data = np.concatenate(dfof_list)
            
        data = np.transpose(data,[-1,0,1])
        img_size = data.shape[-1]
        self.data = data.reshape((-1,1,img_size,img_size))
        self.filename = h5_file
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return(self.data.shape[0])
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = torch.Tensor(self.data[idx])   
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
    
class WF3dDataset(Dataset):
    def __init__(self, h5_file, root_dir, N_fm,xval='train',transform=None):

        #Read in h5 file
        with h5py.File(os.path.join(root_dir,h5_file),'r') as h5file:
            #Save calculated AR matrices and transition matrix
            dfof_list = list(h5file['dfof_list'])
        
        #Take the first 8400 frames (14minutes) of each session 
        if xval == 'train':
            train_list = [data[:,:,:8640] for data in dfof_list]
            data = np.concatenate(train_list,axis=-1)
        elif xval == 'test':
            test_list = [data[:,:,8640:] for data in dfof_list]
            data = np.concatenate(test_list,axis=-1)
        else:
            data = np.concatenate(dfof_list,axis=-1)
            
        data = np.transpose(data,[-1,0,1])
        img_size = data.shape[-1]
        data = data.reshape((-1,1,img_size,img_size))
        self.data = data
        self.filename = h5_file
        self.root_dir = root_dir
        self.transform = transform
        self.N_fm = N_fm
        
    def __len__(self):
        return(self.data.shape[0])
    
    def __getitem__(self,idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        if idx >= self.N_fm:
            sample_list = []
            for n in range(self.N_fm):
                img = torch.Tensor(self.data[idx-n]) 
                if self.transform:
                    img = self.transform(img)
                sample_list.append(img)
            sample = torch.cat(sample_list,dim=0).unsqueeze(0) # To put into B x C x D x H x W format
        elif idx < self.N_fm:
            #Just copy the first frame (N_fm - idx) number of times
            sample_list = []
            for ii in range(self.N_fm-idx):
                img = torch.Tensor(self.data[0]) 
                if self.transform:
                    img = self.transform(img)
                sample_list.append(img)
            for ii in range(idx):
                img = torch.Tensor(self.data[ii]) 
                if self.transform:
                    img = self.transform(img)
                sample_list.append(img)
            sample = torch.cat(sample_list,dim=0).unsqueeze(0)
        
        return sample
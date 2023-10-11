import numpy as np
import pandas as pd
import os
import random

from torch.utils.data import Dataset, Subset
import torch
from pathlib import Path

import random
import torchvision.transforms as T
from torch.nn.functional import unfold
from skimage.transform import resize




class ProcasBags(Dataset):
    def __init__(self, csf, partition, seed = 1, tile_size = 224, train_percent = 0.7, train=True):
        self.train_percent = train_percent
        self.csf = csf
        self.train = train
        self.t= tile_size
        self.partition = partition
        if self.csf == 'y':
            self.root = '/mnt/iusers01/ct01/k16296sr/scratch/datasets/PROCAS_matched'
            self.data = pd.read_csv(Path(self.root, 'PROCAS_matched_sr.csv'))
        elif self.csf == 'n':
            self.root = 'Z:/Datasets/PROCAS_matched'
            self.data = pd.read_csv(Path(self.root, 'PROCAS_matched_sr.csv'))
        self.data['DiagnosisOfCancer'] = pd.Series(np.where(self.data['DiagnosisOfCancer'] == 'Yes', 1, 0), self.data.index)
        self.data['cancer_diagnosis_class'] = pd.Categorical(self.data['cancer_diagnosis_class'], 
                                                        categories=['SDC', 'Interval 1', 'FSDC 1', 'Interval 2', 'FSDC 2', 'Interval 3', 'FSDC 3', 'Distant'],
                                                        ordered=True)
        if self.partition == 'sdcs':
            print('Loaded sdcs')
            self.pat_data = self.data[self.data['CC_set']=='sdcs']
        if self.partition == 'priors':
            print('Loaded priros')
            self.pat_data = self.data[self.data['CC_set']=='priors']
        if self.partition == 'sdcs+oc':
            print('Loaded sdcs+oc')
            self.pat_data = self.data[(self.data['CC_set']=='sdcs')|(self.data['CC_set']=='sr')]
        if self.partition == 'sdcs+I2':
            print('Loaded sdcs+I2')
            self.pat_data = self.data[(self.data['matched_group_id'].isin(self.data[self.data['cancer_diagnosis_class'] <= 'Interval 2']['matched_group_id'].unique())) & (self.data['CC_set'] != 'priors')]
        if self.partition == 'all_sdcs':
            print('Loaded all sdcs')
            self.pat_data = self.data[(self.data['matched_group_id'].isin(self.data[self.data['cancer_diagnosis_class'] == 'SDC']['matched_group_id'].unique())) & (self.data['CC_set'] != 'priors')]
        if self.partition == 'no_sdcs':
            print('Loaded no sdcs')
            self.pat_data = self.data[(self.data['matched_group_id'].isin(self.data[self.data['cancer_diagnosis_class'] > 'SDC']['matched_group_id'].unique())) & (self.data['CC_set'] != 'priors')]
        if self.partition == 'fsdc2':
            print('Loaded fsdcs 2')
            self.pat_data = self.data[(self.data['matched_group_id'].isin(self.data[(self.data['cancer_diagnosis_class'] > 'SDC') & 
                                                     (self.data['cancer_diagnosis_class'] <= 'FSDC 2')]['matched_group_id'].unique())) & (self.data['CC_set'] != 'priors')]
        if self.partition == 'from_I2':
            print('Loaded from I2')
            self.pat_data = self.data[(self.data['matched_group_id'].isin(self.data[self.data['cancer_diagnosis_class'] >= 'Interval 2']['matched_group_id'].unique())) & (self.data['CC_set'] != 'priors')]
        self.seed = seed
        self.tile_csv = pd.read_csv(Path(self.root, 'ALL_tile_amounts_proc_matched.csv'))
        self.mamm_list = self._form_list()
        

        

    def _form_list(self):
        mamm_list = []
        # Change the folder name correspoding to the exact folder we want to use
        file_path = Path(self.root,'data_full_proc')
        file_dirs = os.listdir(file_path)
        side_list = []
        for mamm_files in file_dirs:
            image_id = int(mamm_files[-5:])
            if image_id in self.pat_data['ASSURE_PROCESSED_ANON_ID'].unique():
                
                mamms = os.listdir(Path(file_path,mamm_files))

                random.seed(a=self.seed)
                label = int(self.pat_data[self.pat_data['ASSURE_PROCESSED_ANON_ID']==image_id]['DiagnosisOfCancer'])
                if label == 1:
                    cancer_side = self.pat_data.loc[self.pat_data['ASSURE_PROCESSED_ANON_ID']==image_id, 'Side'].item()
                else:
                    cancer_side = random.choice(['Right', 'Left'])
                side_list.append(cancer_side)    
                for mamm in mamms:
                    if cancer_side == 'Right' and any(k in mamm for k in ('LCC', 'LMLO')):
                       continue
                    if cancer_side == 'Left' and any(k in mamm for k in ('RCC', 'RMLO')):
                       continue 
                    tile_amount = self.tile_csv['Tile_amount'].loc[self.tile_csv['ASSURE_ID']==Path(file_path, mamm_files, mamm).stem]
                    mamm_list.append([Path(file_path, mamm_files, mamm), mamm_files, label, tile_amount])
        return mamm_list
               
    
    def _form_bag(self, mamm_path, combine=False):
        bag = np.load(mamm_path[0]) 
        bag = torch.as_tensor(bag.copy()).float()
        # Flips the Left mammos if they are present
        if 'RMLO' in mamm_path[0].stem or 'RCC' in mamm_path[0].stem:
            bag = T.functional.hflip(bag)
        bag = bag.unsqueeze(0)
        norm = T.Normalize((0.56718950817, 0.56718950817 , 0.56718950817),(0.20914499727207, 0.20914499727207 , 0.20914499727207))
        trans = T.RandomAffine(degrees = 10, translate = (0.05,0.05), scale = (0.9,1.1), shear = (-10,10,-10,10))
        if self.train:
            bag = trans(bag)
    
        tile_amount = mamm_path[3].item()
        if tile_amount > 100:
            bag = bag.unfold(1,self.t,int(self.t-24)) # dim, size, step
            bag = bag.unfold(2,self.t,int(self.t-24))
        elif tile_amount > 50:
            bag = bag.unfold(1,self.t,int(self.t*3/4)) # dim, size, step
            bag = bag.unfold(2,self.t,int(self.t*3/4))
            
        elif tile_amount > 15:
            bag = bag.unfold(1,self.t,int(self.t/2)) # dim, size, step
            bag = bag.unfold(2,self.t,int(self.t/2))
            
        else:
            bag = bag.unfold(1,self.t,int(self.t-150)) # dim, size, step
            bag = bag.unfold(2,self.t,int(self.t-150))

        
        bag = bag.reshape(bag.shape[1]*bag.shape[2],self.t,self.t)
        #bag = bag[torch.count_nonzero(bag,(1,2)) > (self.t*self.t)/1.05].unsqueeze(1)
        # bag = bag[torch.count_nonzero(bag,(1,2)) > 10].unsqueeze(1)
        bag = bag[torch.count_nonzero(bag,(1,2)) > (self.t*self.t)/20].unsqueeze(1)
        
        bag = torch.cat((bag, bag, bag), 1)
        bag = norm(bag)
        bag = bag.contiguous()
        
        # Labels need to be repeated the same number as patches since they are treated as batch size
        label = mamm_path[2]
        label = np.repeat(label, bag.shape[0])
        label = torch.as_tensor(label).float().contiguous()
  
                
        return bag, label
    
        
    
    def __len__(self):
        return len(self.mamm_list)
        
        
    def __getitem__(self, index):
        bag, label = self._form_bag(self.mamm_list[index])
        return bag, [max(label), label], self.mamm_list[index][0].stem
    
if __name__ == "__main__":   
    dataset = ProcasBags(csf='n', seed = 1, partition = 'fsdc2')
    
    im = dataset.__getitem__(2387)
    print(im[2])
    print(im[0].shape)
    
    
    
#!/usr/bin/env python
# coding: utf-8

# In[2]:


import random
from torch.utils.data import Dataset, DataLoader

import import_ipynb
from text_process import *


# In[ ]:


class GANDataset(Dataset):
    def __init__(self,data): 
        self.data=data
        
    def __getitem__(self,idx): # idx item 반환
        return self.data[idx]
        
    def __len__(self): # data size 반환
        return len(self.data)


# In[ ]:


class GenDataIter:
    def __init__(self, samples, if_test_data=False, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shuffle else shuffle
        if cfg.if_real_data:
            self.word2idx, self.idx2word = load_dict(cfg.dataset)
        if if_test_data: # used for the classifier
            self.word2idx, self.idx2word = load_test_dict(cfg.dataset)
        self.loader = DataLoader(
            dataset = GANDataset(self.read_data(samples)),
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last=True        
        )
        
        self.input = self.all_data('input')
        self.target = self.all_data('target')
        
    def read_data(self, samples):
        '''
        input : same as target, but start with start_letter
        '''
        if isinstance(samples,torch.Tensor): # isinstance : samples가 Tensor이면 True
            inp, target = self.prepare(samples)
            all_data = [{'input':i, 'target':t} for (i,t) in zip(inp,target)]
        elif isinstance(samples,str):
            inp, target = self.load_data(samples) # samples : filename
            all_data = [{'input':i, 'target':t} for (i,t) in zip(inp,target)]
        else:
            all_data =None
            
        return all_data
    
    @staticmethod
    def prepare(samples, gpu=False):
        ''' add start_letter to sampels as inp, target same as samples'''
        inp = torch.zeros(samples.size()).long()
        target = samples
        inp[:,0] = cfg.start_letter
        inp[:,1:] = target[:, :cfg.max_seq_len-1]
        
        if gpu:
            return inp.cuda(), target.cuda()
        return inp, target
    
    def load_data(self,filename):
        self.tokens = get_tokenized(filename)
        samples_idx = tokens_to_tensor(self.tokens, self.word2idx) # add padding and token -> tensor
        return self.prepare(samples_idx)


    #잘 모르겠으니 체크해보자
    #각각 데이터 들에 dim 0 추가하고 다시 원래대로 만든거같긴한데.
    def all_data(self, col):
        return torch.cat([data[col].unsqueeze(0) for data in self.loader.dataset.data], dim = 0)


# In[ ]:


class DisDataIter:
    def __init__(self, pos_samples, neg_samples, shuffle=None):
        self.batch_size = cfg.batch_size
        self.max_seq_len = cfg.max_seq_len
        self.start_letter = cfg.start_letter
        self.shuffle = cfg.data_shuffle if not shulffe else shuffle
        
        self.loader = DataLoader(
            dataset = GANDataset(self.__read_data__(pos_samples, neg_samples)),
            batch_size = self.batch_size,
            shuffle = self.shuffle,
            drop_last=True)
        
        def __read_data__(self, pos_samples, neg_samples):
            inp, target = self.prepare(pos_samples,neg_samples)
            all_data = [{'input' : i, 'target':t } for (i,t) in zip(inp,target)]
        
            return all_data
        
        def prepare(self, pos_samples, neg_samples, gpu=False):
            inp = torch.cat((pos_samples, neg_samples), dim=0).long().detach()
            target = torch.ones(inp.size(0)).long()
            
            #pos_samples =0 , neg_samples =1
            target[pos_samples.size(0):] = 0
            
            #순서 섞어줌
            perm = torch.randperm(inp.size(0))
            inp = inp[prem]
            target = target[perm]
            
            if gpu:
                return inp.cuda(), target.cuda()
            
            return inp, target


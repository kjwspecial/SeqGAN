#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[7]:


filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]

class Discriminator(nn.Module):
    def __init__(self,embedding_dim, vocab_size, filter_sizes, num_filters, padding_idx, gpu=False, dropout = 0.25):
        super(Discriminator,self).__init__()
        self.embedding_dim = embedding_dim
        self.voacb_size = vocab_size
        self.padding_idx = padding_idx
        self.feature_dim = sum(num_filters)
        self.gpu = gpu
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        self.conv = nn.ModuleList([
            nn.Conv2d(1, channel, (kernal_size, embedding_dim)) for (channel, kernal_size) in zip(num_filters, filter_size)])
        
        
        self.highway = nn.Linear(self.feature_dim, self.feature_dim)
        self.feature2out = nn.Linear(self.feature_dim, 2)
        self.dropout = nn.Dropout(dropout)
        
        self.init_params()
        
    
    def forward(self,inp):
        pass
    
    def get_feature(self):
        pass
        
    def init_params(self):
        for params in self.parameters():
            if params.requires_gard and len(params.shape) >0:
                stddev = 1/ math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std = stddev)


# In[ ]:





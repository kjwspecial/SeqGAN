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
        '''
        -input
            inp : batch_size x feature_dim
        -return : batch_size x 2 (probability)
        '''
        
        feature = get_feature(inp)
        pred = self.feature2out(self.dropout(feature))
        return pred
    
    def get_feature(self,inp):
        '''
        sentence 주어지면 feature vector 추출
        
        -input
            inp : batch_size x max_seq_len
        -return : batch_size x feature_dim
        '''
        emb = self.embeddings(inp).unsqueeze(1)                                # batch_size x 1 x max_seq_len x embedding_dim
        convs = [F.relu(conv(emb)).squeeze(3) for conv in self.conv]           # [batch_size x num_filter x 필터 사이즈에 따른 length]
        pools = [F.max_pool1d(conv,conv.size(2)).squeeze(2) for conv in convs] # [batch_size x num_filter], kernal size를 length로 해서 feature 1개 뽑고 squeeze
        pred = torch.cat(pools,1)                                              # batch_size x feature_dim
        highway = self.highway(pred)
        pred = torch.sigmoid(highway) * F.relu(highway) + (1. - torch.sigmoid(highway)) * pred # Highway: y= T⋅H(x,W_H)+(1−T)⋅x
        
        return pred
    
    def init_params(self):
        for params in self.parameters():
            if params.requires_gard and len(params.shape) >0:
                stddev = 1/ math.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std = stddev)


# In[ ]:





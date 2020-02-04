#!/usr/bin/env python
# coding: utf-8

# In[4]:


import torch
import torch.nn as nn
import config as cfg
from metrics.basic import Metrics


# In[5]:


class NLL(Metrics):
    def __init__(self,name, if_use = False, gpu =False):
        super(NLL, self).__init__(name)
        
        self.if_use = if_use
        self.gpu = gpu
        
        self.model = None
        self.data_loader = None
        self.criterion = nn.NLLLoss()
        
    def get_score(self):
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()'
        return self.cal_nll(self.model, self.data_loader, self.criterion, self.gpu)

    def reset(self, model=None, data_loader=None):
        self.model = model
        self.data_loader = data_loader
        
    @staticmethod
    def cal_nll(model, data_loader, criterion, gpu = cfg.CUDA):
        total_loss = 0
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
            if gpu:
                inp, target = inp.cuda(), target.cuda()
            
            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            total_loss += loss.item()
        return round(total_loss / len(data_loader), 4)
    


# In[ ]:





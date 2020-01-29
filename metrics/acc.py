#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import import_ipynb
from basic import Metrics


# In[ ]:


class ACC(Metrics):
    def __init__(self, if_use = True, gpu=True):
        super(ACC,self).__init__('clasification_acc')
        
        self.if_use = if_use
        self.gpu = gpu
        self.model =None
        self.data_loader = None
        
    def get_score(self):
        if not self.if_use:
            return 0
        assert self.model and self.data_loader, 'Need to reset() before get_score()'
        
        return self.cal_acc(self.model, self.data_loader)
    
    def reset(self):
        
    def cal_acc(self, model, data_loader):
        total_acc=0
        total_num=0
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if self.gpu:
                    inp, target =inp.cuda(), target.cuda()
                    
                pred = model.forward(inp) # softmax 안함?
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item() # batch x max_seq_len 
                total_num += inp.size(0) # batch_size
                
        return round(total_acc/total_num,4)


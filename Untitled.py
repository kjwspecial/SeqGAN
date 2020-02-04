#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.optim as optim


# In[ ]:


class SeqGANInstructor(Instructor):
    def __init__(self,opt):
        super(SeqGANInstructor, self).__init__(opt):


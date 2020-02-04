#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
from time import strftime, localtime

import os
import re
import torch


# In[3]:


data_shuffle = False  # False

dataset = "image_coco"
vocab_size = 6613
max_seq_len = 37
batch_size = 64
# ===Basic Train===
start_letter = 1
padding_idx = 0
start_token = 'BOS'
padding_token = 'EOS'



train_data = 'dataset/' + dataset + '.txt'
test_data = 'dataset/testdata/' + dataset + '_test.txt'


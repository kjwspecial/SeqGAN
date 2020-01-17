#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[6]:


class Generator(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, padding_idx,gpu=False):
        super(Generator, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_idx = pad_idx
        self.gpu = gpu
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first= True)
        self.lstm2out = nn.Linear(hidden_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        
        self.init_params()
        
    def forward(self, inp, hidden):
        '''
        embeds input and applies LSTM
        
        - input
            inp : (batch_size, seq_len)
            hidden : (h,c)
            
        - output
            pred : (batch_size*seq_len, voacb_size)
            
        '''
        emb = self.embedding(inp)                     # batch_size x seq_len x embedding_dim
        if len(inp.size()) == 1:
            emb = emb.unsqueeze(1)                    # batch_size x 1 x embedding_dim
        
        out, hidden = self.lstm(hidden)               # out : batch_size x seq_len x hidden_dim
        out = out.contiguous().view(-1, hidden_dim)   # (batch_size * seq_len) x hidden_dim
        out = self.lstm2out(out)                      # (batch_size * seq_len) x vocab_size
        pred = self.softmax(out)
        
        return pred
        
    def init_hidden(self,batch_size):
        h = torch.zeros(1, batch_size, self.hidden_size)
        c = torch.zeros(1, batch_size, self.hidden_size)
        if self.gpu:
            return h.cuda(), c.cuda()
        else:
            return h, c
        
    def init_params(self, batch_size):
        for parans in self.parameters():
            if param.requires_grad and len(param.shape) > 0:
                stddev = 1 / match.sqrt(param.shape[0])
                torch.nn.init.normal_(param, std= stddev)
    
    def init_oracle(self):
        for param in self.paramters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0, std =1)
    
    def samples(self, num_samples, batch_size, start_letter=0):
        
        num_batch = num_samples // batch_size + 1 if num_samples != batch_size else 1
        samples = torch.zeors(num_batch * batch_size, self.max_seq_len).long()
        
        #generate sentences with multinomial sampling, 각 row에서 vocab 하나씩 sampling
        #samples = torch.zeros(num_samples, self.max_seq_len).long()
        
        # variable 생성: (1, num_samples, hidden_dim)
        h = self.init_hidden(num_samples)
        
        # inp : num_samples개 만큼 문장 생성 할것이다.
        inp = torch.LongTensor([start_letter]*num_samples)
        #x = torch.LongTensor([start_letter]* num_samples))
        
        if self.gpu:
            samples = samples.cuda()
            inp = inp.cuda()
        
        # 각 sample 한 단어씩 뽑기
        for i in range(self.max_seq_len):
            out, h = self.forward(x,h)                             # out: num_samples x vocab_size
            next_token = torch.multinomial(torch.exp(out), 1)      # num_samples x 1 (sampling from each row) row마다 한개씩 sampling
            sampels[:, i] = out.view(-1).data                      # samples의 각 column마다 random 뽑힌 단어들 입력 받음.
    
            x = out.view(-1)
            
        return samples
    
    def batchPGLOSS(self, inp, target, reward):
        
        batch_size, seq_len = inp.size()
        
        h = self.init_hidden(batch_size)
        out, h = self.fowrard(x,h).view(batch_size, self.max_seq_len, self.voacb_size)
        target_onehot = F.one_hot(target, self.vocab_size).float()                     # batch_size x seq_len x vocab_size
        pred = torch.sum(out * target_onehot, dim=-1)                                  # batch_size x seq_len
        loss = -torch.sum(pred * reward)  # 많이 맞출수록 loss 감소

        return loss


# In[ ]:


class Discriminator(nn.Module):
    def __init__(self,embedding_dim, vocab_size, filters_sizes, num_filters, padding_idx, gpu=False, dropout = 0.2):
        super(Discriminator,self)__init__()
        self.embedding_dim = embedding_dim
        self.


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb
import math
import torch.nn.init as init


# In[ ]:


class Generator_old(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_seq_len, gpu=False, orcale_init = False)
    super(Generator, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.max_seq_len = max_seq_len
    self.gpu = gpu
    
    #Embedidng
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    self.gru = nn.GRU(embedding_dim, hidden_dim)         # (seq_len, batch, input_size)
    self.gru2out = nn.Linear(hidden_dim, vocab_size)
    
    #초기화 분산이 매우 작아서 => N(0,1)로 네트워크 초기화
    if oracle_init:
        for param in self.parameters():
            if param.requires_grad:
                init.normal(param, mean = 0, std =1)
    
    #필요한 이유 생각해보기. 왜 h를 따로 만들어서 관리하지?
    
    def init_hidden(self, batch_size = 1)
        h = autograd.Variable(torch.zeros(1, batch_size, self.hidden_dim))    # (num_layers * num_directions, batch, hidden_size)
        #h = torch.zeros(1, batch_size, self.hidden_dim)
        if self.gpu:
            return h.cuda()
        else:
            return h
        
    def forward(self, x, hidden):
        '''
            input 임베딩, GRU one token at a time (seq_len = 1)
            
        - input shape
            emb : LongTensor
            gru : (seq_len, batch_size, input_size(여기서는 emb_dim)), (num_layers, batch_size, hidden_dim)
            gru2out : (batch_size, hidden_dim)
            
        - output shape
            emb : (batch_size, embedding_dim)
            gru : (seq_len, batch_size, hidden_dim)
            gru2out : (batch_size, vocab_size)
            F.log_softmax( ,dim=1) : (batch_size, vocab_size)
            
        '''
        # input dim                                         # batch_size
        emb = self.embeddings(x)                            # batch_size x embedding_dim
        emb = emb.view(1, -1, self.embedding_dim)           # 1 x batch_size x embedding_dim
        out, hidden = self.gru(emb, hidden)                 # 1 x batch_size x hidden_dim(out)
        out = self.gru2out(out.view(-1, self.hidden_dim))   # batch_size x vocab_size
        out = F.log_softmax(out, dim=1)
        return out, hidden

    #MCTS 정도로 생각하면 되려나
    def sample(self, num_samples, start_letter = 0):

        '''    
        - multinomial - random sampling인지 확인 필요.
        
            input shape : (input, num_samples)
            output shape : (num_samples, 1) => input의 각 row마다 1개씩(vocab 1개) sampling  
        
        네트워크 샘플링, return max_seq_len인 num_samples개 samples
        
        Outputs : samples, hidden
                -smaple: num_samples x max_seq_len (각 row가 sampled sequence) =>batch 정도로 생각하면 될듯.
                
                
        '''
        
        samples = torch.zeros(num_samples, self.max_seq_len).type(torch.LongTensor)
        
        # variable 생성: (1, num_samples, hidden_dim)
        h = self.init_hidden(num_samples)
        
        # inp : num_samples개 만큼 문장 생성 할것이다.
        inp = autograd.Variable(torch.LongTensor([start_letter]*num_samples))
        #x = torch.LongTensor([start_letter]* num_samples))
        
        if self.gpu:
            samples = samples.cuda()
            x = x.cuda()
        
        # 각 sample 한 단어씩 뽑기
        for i in range(self.max_seq_len):
            out, h = self.forward(x,h)                      # out: num_samples x vocab_size
            out = torch.multinomial(torch.exp(out), 1)      # num_samples x 1 (sampling from each row) row마다 한개씩 sampling
            sampels[:, i] = out.view(-1).data               # samples의 각 column마다 random 뽑힌 단어들 입력 받음.
    
            x = out.view(-1)
            
        return samples
    
    def batchNLLLoss(self, x, target):
        '''
        returns the NLL Loss for predictiong target sequence.
        
        Input : x, target
            -x : batch_size x seq_len
            -target : batch_size x seq_len
            
            x should be target with <s> (start letter) prepended
        '''
    
        loss_fn = nn.NLLLoss()
        batch_size, seq_len = x.size()
        x = x.permute(1,0)                  #seq_len x batch_size
        tartget = target.permute(1,0)       #seq_len x batch_size
        h = self.init_hidden(batch_size)
        
        loss = 0
        for i in range(seq_len):
            out, h = self.forward(x[i],h)
            loss += loss_fn(out, target[i])
            
        return loss
    
    def batchPGLoss(self, x, target, reward):
        '''
        Inputs: x, target
            - x : batch_size x seq_len
            - target : batch_size x seq_len
            - reward : batch_size (D가 각 sentence마다 reward줌)
        '''
        
        batch_size, seq_len = x.size()
        x = x.permute(1,0)
        target = target.permute(1,0)
        
        h = self.init_hidden(batch_size)
        
        loss = 0 
        
        #단어 한개씩 넣기.
        for i in range(seq_len):
            # out shape : (batch_size*seq_len, vocab_size)
            # x[i] shape : seq_len
            out, h = self.forward(x[i], h)
            
            for j in range(batch_size):
                loss + = -out[j][target[i][j]]*reward[j]    # log(P(y_t|Y_1:Y_{t-1})) * Q
                
        return loss/batch_size
    
    
    def batchPGLOSS2(self, x, target, reward):
        '''
        Return a policy gradient loss
        
        :param x : batch_size x seq_len, 'x' should be target with <s> (start letter) prepended
        :target x : batch_size x seq_len
        :reward
        '''
                
        batch_size, seq_len = x.size()
        h = self.init_hidden(batch_size)
        #  (batch_size*seq_len) x vocab_size
        out, h = self.forward(x, h).view(batch_size, self.max_seq_len, self.vocab_size)
        target_onehot = F.one_hot(target, self.vocab_size).float() # batch_size * seq_len * vocab_size
        pred = torch.sum(out * target_onehot, dim=-1) # batch_* seq_len
        loss = -torch.sum(pred * reward) #많이 맞추면 커짐.
        
        return loss


# In[145]:


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


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import copy
import torch
import torch.nn.functional as F


# In[ ]:


class ROLLOUT:
    '''
    완성된 문장을 가지고, 문장의 첫단어 + sampling                * rollout_num
                         문장의 두번째 까지 단어 + sampling ...  * rollout_num
                         ....
                        
    => max_seq_len까지 sampling해서 문장 만들고, 총 batch_size * rollout_num * max_seq_len 개의 문장 생성
    
    
    '''

    def __init__(self, gen, gpu=True):
        self.gen = gen
        self.old_model = copy.deepcopy(gen)
        self.max_seq_len = gen.max_seq_len
        self.vocab_size = gen.vocab_size
        self.setp_size = gen.setp_size if gen.name == 'leakgan' else 0
        self.goal_out_size = gen.goal_out_size if gen.name == 'leakgan' else 0
        self.gpu = gpu
        
    def rollout_mc_search(self, sentences, given_num):
        '''
            나머지 token mc search로 채움.
        '''
        batch_size = sentences.size(0)
        
        #get current state init_hidden=> 만들어서 새로 할당받는거아니야?
        hidden = self.gen.init_hidden(batch_size)
        
        # inp : 현재까지 생성된 문장.
        # out :각 batch의 마지막 문장의 마지막 단어[:,-1]에 대한 확률
        inp = sentences[:, :given_num]
        out, hidden = self.gen.forward(inp, hidden, need_hidden = True) # need_hidden?
        out = out.view(batch_size, -1, self.vocab_size)[:,-1]           # batch_size x seq_len x voacb_size [:,-1]   => batch_size x vocab_size
        
        samples = torch.zeros(batch_size, self.max_seq_len).long()
        samples[:,:given_num] = sentences[:,:given_num]
        
        
        if self.gpu:
            samples = samples.cuda()
        
        # MC search
        for i in range(given_num, self.max_seq_len):
            out = torch.multinomial(torch.exp(out),1)              # row마다 한개씩 뽑음.
            samples[:,i] = out.view(-1).data                       # sampling한 단어들 이어붙이기.
            inp = out.view(-1)                                     # batch_size 개의 단어들
            
            out , hidden = self.gen.forward(inp,hidden,need_hidden=True) #앞에 생성된 단어를 입력받아서 다음단어 선택.
            
        return samples # batch_size x max_seq_len
    
    def get_reward(self, sentences, rollout_num, D, current_k=0):
        '''
        get reward via MTCS
        sentence : batch_size X max_seq_len
        rollout_num : 문장당 rollout 몇번했는지.
        current_k = current training gen
        reward = [batch_size]
        '''
        
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num * self.max_seq_len, batch_size]).float()
        if self.gpu:
            rewards = rewards.cuda()
        
        idx = 0
        for i range(rollout_num):                                               # 문장당 롤아웃 횟수
            for igiven_num in range(1, self.max_seq_len +1):                     # 현재 생성된 문장의 몇번째 단어 까지 입력으로 넣을지.
                samples = self.rollout_mc_search(sentences, given_num)          # batch_size x max_seq_len
                out = D.forward(samples)
                out = F.softmax(out, dim=-1)                                    # 각 batch당 reward , batch_size x 2(가짜,진짜)
                reward = out[:, current_k+1]
                reward[idx] = reward
                idx +=1
        
        rewards = torch.mean(rewards.view(batch_size, self.max_seq_len, rollout_num),dim = -1) # Q-function
        
        return rewards
        
    def get_token_reward(self, sentences, rollout_num, D, current_k, given_num):
        '''
        MCTS, each token reward = (batch_size x max_seq_len) / rollout_num
        
        '''
        with torch.no_grad():
            batch_size = sentences.size(0)
            rewards = torch.zeros([rollout_num, batch_size]).float()
            
            idx =0
            
            for i in range(rollout_num):
                samples = self.rollout_mc_search(sentences, given_num)
                out = D(samples)
                out = F.softmax(out, dim=-1)
                reward = out[:, current_k+1]
                rewards[idx]= reward
                idx +=1
                
            rewards = torch.Tensor(rewards).cuda()
            rewards = torch.sum(rewards,dim =0) / rollout_num
            
        return rewards


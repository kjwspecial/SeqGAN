#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn

import config as cfg
from text_process import load_dict, tensor_to_tokens, write_tokens
from data_loader import GenDataIter

from metrics.bleu import BLEU
from metrics.nll import NLL


# In[3]:


class Instructor:
    def __init__(self,opt):
        # Load Dict
        self.word2idx, self.idx2word = load_dict(cfg.dataset)
        
        # Load Data
        self.train_data = GenDataIter(cfg.train_data)
        self.test_data = GenDataIter(cfg.test_data, if_test_data=True)
        
        # Criterion
        self.MLE_criterion = nn.NLLLoss()
        self.D_criertion = nn.CrossEntropyLoss()
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.clas_opt = None
        
        # Metrics
        self.bleu = BLEU('BLEU', gram=[2,3,4,5], if_use = cfg.use_bleu)
        self.self_bleu = BLEU('Self_BLEU', gram=[2,3,4], if_use = cfg.use_self_bleu)

        self.nll_g = NLL('NLL_G',if_use = cfg.use_nll_g, gpu = cfg.CUDA)
        self.nll_d = NLL('NLL_D',if_use = cfg.use_nll_d, gpu = cfg.CUDA)
 
        self.acc = ACC(if_use = cfg.use_acc)
        self.all_metrics = [self.bleu, self.self_bleu, self.nll_g, self.nll_d]
        
    def _run(self):
        pass
    
    def _text(self):
        pass
    
    def init_model(self):
        if cfg.dis_pretrian:
            print('Load pre-train Discriminator: {}'.format(cfg.pretrained_D_path))
            self.D.load_state_dict(torch.load(cfg.pretrained_D_path))
        if cfg.gen_pretrain:
            print('Load MLE pre-train Generator: {}'.format(cfg.pretrained_G_path))
            self.G.load_state_dict(torch.load(cfg.pretrained_G_path))
            
        if cfg.CUDA:
            self.D = self.D.cuda()
            self.G = self.G.cuda()
    
    def cal_metrics(self):
        with torch.no_grad():
            samples = self.gen.samples(cfg.samples_num, 4 * cfg.batch_size)
            gen_data = GenDataIter(eval_samples)
            gen_tokens = tensor_to_tokens(eval_samples, self.idx2word)
            gen_tokens_small = tensor_to_tokens(self.gen.samples(200,200), self.idx2word)
            
            #real_text : str로 입력받아서 data_loader의 load_data를 통해 tokens으로 나눠짐.
            self.bleu.reset(test_text = gen_tokens, real_text = self.test_data.tokens)
            self.self_bleu.reset(test_text = gen_tokens_small, real_text = self.test_data.tokens)
            self.nll_g.reset(model = self.G, data_loader = self.train_data.loader)
            self.nll_d.reset(model = self.D, data_loader = gen_data.lodaer)
        return [metrics.get_score() for metrics in self.all_metrics]
    
    @staticmethod
    def optimize(opt, loss , model =None, retain_graph = False):
        opt.zero_grad()
        loss.backward(retain_graph = retain_graph) # true하면 computation graph 보존가능.
        if model is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
        opt.step()
        
    def train_G_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        for i,data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            
            hidden = model.init_hidden(data_loader.batch_size)
            pred = model.forward(inp, hidden)
            loss = criterion(pred, target.view(-1))
            self.optimize(optimizer, loss, model)
            total_loss += loss.item()
        return total_loss / len(data_loader)
    
    def trans_D_epoch(self, model, data_loader, criterion, optimizer):
        total_loss = 0
        total_acc = 0
        total_num = 0
        for i, data in enumerate(data_loader):
            inp, target = data['input'], data['target']
            if cfg.CUDA:
                inp, target = inp.cuda(), target.cuda()
            pred = model.forward(inp)
            loss = criterion(pred,target)
            self.optimize(optimizer, loss, model)
            
            total_loss += loss.item()
            total_acc += torch.sum((pred.argmax(dim = -1) == target)).item()
            total_num += inp.size(0)
        
        total_loss /= len(data_loader)
        total_acc /= total_num
        return total_loss, total_acc
    
    @staticmethod
    def eval_D(model, data_loader, criterion):
        total_loss = 0
        total_acc = 0
        total_num = 0
        
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inp, target = data['input'], data['target']
                if cfg.CUDA:
                    inp, target = inp.cuda(), target.cuda()
                pred = model.forward(inp)
                loss = criterion(pred, target)
                
                total_loss += loss.item()
                total_acc += torch.sum((pred.argmax(dim=-1) == target)).item()
                total_num += inp.size(0)
                
            total_loss /= len(data_loader)
            total_acc /= total_num
        return total_loss, total_acc
    
    def save_(self, phase, epoch):
        if phase != 'ADV':
            torch.save(self.G.state_dict(), cfg.svae_model_root + 'generator_{}_{:05d}.pt'.format(phase, epoch))
        save_sample_path = cfg.svae_model_root + 'sample_{}_{:05d}.txt'.format(phase, epoch)
        samples = self.gen.samples(cfg.batch_size, cfg.batch_size)
        write_tokens(save_sample_path, tensor_to_tokens(samples, self.idx2word))


# In[ ]:





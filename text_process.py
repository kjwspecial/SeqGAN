#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
import numpy as np
import os
import torch
import import_ipynb
import config as cfg


# In[5]:


def get_tokenized(file):
    '''file 토큰화'''
    tokenized = []
    with open(file) as data:
        for text in data:
            text = nltk.word_tokenize(text.lower())
            tokenized.append(text)
    return tokenized


# In[10]:


def get_word_set(tokens):
    ''' set() : 중복 X, 순서 X '''
    word_set = []
    for sentence in tokens:
        for word in sentence:
            word_set.append(word)
    return list(set(word_set))


# In[54]:


def get_dict(word_set):
    word2idx = {}
    idx2word = {}
    
    index = 2
    word2idx[cfg.padding_token] = cfg.padding_idx
    idx2word[cfg.padding_idx] = cfg.padding_token
    
    word2idx[cfg.start_token] = cfg.start_letter
    idx2word[cfg.start_letter] = cfg.start_token
    
    for idx,word in enumerate(word_set):
        word2idx[word] = idx+2 # +2 : pad_idx, start_token
        idx2word[idx+2] = word
    return word2idx, idx2word


# In[63]:


def get_seq_len(train_data, text_data =None):
    train_tokenized = get_tokenized(train_data)
    if text_data is None:
        text_tokenized = []
    else:
        text_tokenized = get_tokenized(text_data)
    word_set = get_word_set(train_tokenized + text_tokenized)
    word2idx, idx2word = get_dict(word_set)
    
    if text_data is None:
        max_seq_len = len(max(train_tokenized,key=len))
    else:
        max_seq_len = max(len(max(train_tokenized,key=len)), len(max(text_tokenized,key=len)))
    
    return max_seq_len, len(word2idx)#vocab_size


# In[92]:


def init_dict(dataset):
    """save training data dictionary files"""
    tokens = get_tokenized('dataset/{}.txt'.format(dataset))
    word_set = get_word_set(tokens)
    word2idx, idx2word = get_dict(word_set)
    
    with open('dataset/{}_word2idx.txt'.format(dataset), 'w' ) as data:
        data.write(str(word2idx))
    with open('dataset/{}_idx2word.txt'.format(dataset), 'w' ) as data:
        data.write(str(idx2word))

    print('vocab_size: ', len(word2idx))


# In[111]:


def load_dict(dataset):
    word2idx_path = 'dataset/{}_word2idx.txt'.format(dataset)
    idx2word_path = 'dataset/{}_idx2word.txt'.format(dataset)
    
    if not os.path.exists(word2idx_path) or not os.path.exists(idx2word_path):
        init_dict(dataset)
    '''
    .strip() : \n, 양쪽 끝 공백 제거
    '''
    with open(word2idx_path, 'r') as data:
        word2idx = eval(data.read().strip())
    with open(idx2word_path, 'r') as data:
        idx2word = eval(data.read().strip())
        
    return word2idx,idx2word


# In[120]:


def load_test_dict(dataset):
    word2idx, idx2word = load_dict(dataset)
    test_data_path = 'dataset/testdata/{}_test.txt'.format(dataset)
    
    tokenized = get_tokenized(test_data_path)
    word_set = get_word_set(tokenized)
     
    idx = len(word2idx)
    # test_data word 포함시킴.
    for word in word_set:
        if word not in word2idx:
            word2idx[word] = idx+1
            idx2word[idx+1] = word
            idx +=1
    return word2idx, idx2word


# In[ ]:


def tensor_to_tokens(tensor, word_dict):
    '''transform Tensor to tokens'''
    tokens =[]
    for sentence in tensor:
        sentence_token =[]
        for word in sentence.tolist():
            if word == cfg.padding_idx:
                break
            sentence_token.append(word_dict[word])
        tokens.append(sentence_token)
    return tokens


# In[116]:


def tokens_to_tensor(tokens,word_dict):
    '''add extra padding and transform word tokens to tensor'''
    tensor =[]
    for sentence in tokens:
        sentence_tensor = []
        for word in sent:
            if word == cfg.padding_token:
                break
            sentence_tensor.append(int(word_dict[word]))
            
        extra_padding_size = cfg.max_seq_len-len(sentence_tensor)
        for i in range(extra_padding_size):
            sentence_tensor.append(cfg.padding_idx)
        tensor.append(sentence_tensor[:cfg.max_seq_len])
    return torch.LongTensor(tensor)


# In[117]:


def padding_token(tokens):
    '''sentence padding'''
    pad_tokens= []
    for sentence in tokens:
        sentence_token = []
        for word in sentence:
            if word == cfg.padding_token:
                break
            sentence_token.append(word)
        extra_padding_size = cfg.max_seq_len-len(sentence_token)
        for i in range(extra_padding_size):
            sentence_token.append(cfg.padding_token)
        pad_tokens.append(sentence_token)
    return pad_tokens


# In[ ]:


dataset = cfg.dataset
train_data = cfg.train_data


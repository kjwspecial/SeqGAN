#!/usr/bin/env python
# coding: utf-8

# In[2]:


from multiprocessing import Pool # process 병렬처리

import nltk
import os
import random
from nltk.translate.bleu_score import SmoothingFunction

from basic import Metrics


# In[9]:


class BLEU(Metrics):
    def __init__(self, name=None, test_text=None, real_text=None, gram=3, portion=1, if_use=False):
        assert type(gram) == int or type(gram) == list, 'Gram format error'
        super(BLEU,self).__init__('%s-%s'% (name,gram))
        self.if_use = if_use
        self.test_text = test_text
        self.real_Text = real_text
        
        self.gram = [gram] if type(gram) == int else gram
        self.sample_size = 200 # BLEU scores remain nearly unchanged for self.sample_size >= 200
        self.reference = None
        self.is_first = True
        self.portion = portion  # how many portions to use in the evaluation, default to use the whole test dataset
        
    def get_score(self, is_fast= True, given_gram= None):
        '''
        is_fast : Fast mode
        given_gram : Calc specific n-gram BLEU score
        '''
        if not self.if_use:
            return 0
        if self.is_first:
            self.get_reference()
            self.is_first = False
        if is_fast:
            return self.get_bleu_fast(given_gram)
        return self.get_bleu(given_gram)
    
    def reset(self, test_text = None, real_text = None):
        self.test_text = test_text if test_text else self.test_text
        self.real_text = real_text if real_text else self.real_text
    
    # 실제 문장과 비교하기 위해 가져옴.
    def get_reference(self):
        reference = self.real_text.copy()
        # randomly choose a portion of test data
        # In-place shuffle
        random.shuffle(reference)
        len_reference = len(reference)
        reference = reference[:int(self.portion * len_reference)]
        self.reference = reference
        return reference
    
    @staticmethod  
    def cal_bleu(reference, hypothesis, weight):
        return nltk.translate.bleu_score.setence_bleu(reference, hypothesis, weight,smoothing_function=SmoothingFunction().method1)
    
    def get_bleu(self, given_gram=None): # given_gram : single gram일 경우 단어의 개수
        if given_gram is not None: # for single gram
            bleu = []
            reference = self.get_reference()
            weight = tuple((1. / given_gram for _ in range(given_gram))) # 단어 당 가중치 : (0.33 , 0.33, 0.33)
            for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                bleu.append(self.cal_bleu(reference,hypothesis, weight))
            return round(sum(bleu) / len(bleu),3)
        else: # multiple gram
            all_bleu =[]
            for ngram in self.gram:
                bleu =[]
                reference = self.get_reference()
                weight = tuple((1. / ngram for _ in range(ngram))) # gram 당 가중치.
                for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
                    bleu.append(self.cal_bleu(reference, hypothesis, weight))
                all_bleu.append(round(sum(bleu) / len(bleu), 3))
            return all_bleu
        
        
        
        
    ### additional
    def get_bleu_parallel(self, ngram, reference):
        weight = tuple((1. / ngram for _ in range(ngram)))
        pool = Pool(os,cpu_count())
        result = []
        for idx, hypothesis in enumerate(self.test_text[:self.sample_size]):
            bleu.append(pool.apply_async(self.cal_bleu,args = (reference,hypothesis,weight)))
        score = 0.0
        cnt = 0
        for i in result:
            score += i.get()
            cnt +=1
        pool.close()
        pool.join()
        return round(score / cnt, 3)
            
            
    def get_bleu_fast(self, given_num = None):
        reference = self.get_reference
        if given_num is not None:
            return self.get_bleu_parallel(ngram=given_gram, reference = reference)
        else:
            all_bleu =[]
            for ngram in self.gram:
                all_bleu.append(self.get_bleu_parallel(ngram = given_gram, reference =reference))
            return all_bleu


# In[ ]:





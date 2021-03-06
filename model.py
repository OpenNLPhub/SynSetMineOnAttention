'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-10 16:21:07
 * @desc 
'''

from pathlib import Path
from typing import Dict, List
import torch
import torch.nn as nn
from torch.nn.modules import dropout
from transformers import BertModel
from config import BertPretrainedModelPath
import numpy as np
import math

class Embedding_layer(nn.Module):
    """embedding layer and provide some method to freeze layer parameters"""
    def __init__(self, vocab_size, embedding_dim)->None:
        super(Embedding_layer,self).__init__()
        self.dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.dim, padding_idx = 0)
        # self._freeze_parameters()

    @classmethod
    def load(cls,word_file):
        """Polymorphism Contructor """
        word2vec = torch.from_numpy(np.load(word_file))
        vocab_size, embedding_dim = word2vec.shape
        layer = cls(vocab_size,embedding_dim)
        layer.embedding = nn.Embedding.from_pretrained(word2vec, padding_idx=0).float()
        return layer
    
    @classmethod
    def from_pretrained(cls,embedding):
        """Polymorphism Contructor """
        word2vec = torch.from_numpy(embedding)
        vocab_size, embedding_dim = word2vec.shape
        layer = cls(vocab_size,embedding_dim)
        layer.embedding = nn.Embedding.from_pretrained(word2vec, padding_idx=0).float()
        return layer

    def forward(self,x):
        """
        Args:
            x:  batch_size, max_word_set_size
        Returns:
            word embedding value 
            batch_size, max_word_set_size, word_emb_size 
        """
        return self.embedding(x)


    def freeze_parameters(self):
        for i in self.embedding.parameters():
            i.requires_grad = False

    def unfreeze_parameters(self):
        for i in self.embedding.parameters():
            i.requires_grad = True


def getPretrainedBertModel(key:str) -> BertModel:
    path = BertPretrainedModelPath[key]
    return BertModel.from_pretrained(path)


def getFCLayer(state_size_list:List,add_dropout:bool=False,dropout:float= 0.2) -> nn.Sequential:
    in_size = state_size_list[:-1]
    out_size = state_size_list[1:]
    size_tuple = list(zip(in_size,out_size))
    L = []
    for ix,(i,j) in enumerate(size_tuple):
        if ix != 0:
            L.append(nn.ReLU())
        bias = True if ix == len(in_size) - 1 else False
        L.append(nn.Linear(i,j,bias=bias))
        if add_dropout and ix!= len(in_size) - 1:
            L.append(nn.Dropout(dropout))
    return nn.Sequential(*L)

            

"""Model Based On Attention """     

def calculate_attention(query:torch.Tensor,key:torch.Tensor,value:torch.Tensor,mask:torch.Tensor):
    """Calclulate Attention 
    @param:
        query: torch.Tensor (Batch_size, max_seq_len, hidden_size)
        key: torch.Tensor (Batch_size, max_seq_len, hidden_size)
        value : torch.Tensor(Batch_size, max_seq_len, word_vector_size)
        mask : torch.Tensor.Boolean(Batch_size, max_seq_len);
    """
    hidden_size = query.size(-1)
    max_seq_size = mask.size(-1)
    scores = torch.matmul(query, key.transpose(-1,-2)) / math.sqrt(hidden_size)
    #scores (batch_size, max_seq_len , max_seq_len)
    mask_ = mask.unsqueeze(-1).expand(-1,-1,max_seq_size)
    scores = scores.masked_fill(mask_ == 0, -1e9)

    attention_matrix = torch.softmax(scores, dim = -1)
    # (batch_size, max_seq_len , max_seq_len)

    output = torch.matmul(attention_matrix,value)

    return output


class Attention_layer(nn.Module):
    """Attention Unit"""
    def __init__(self,kernel_input:int, kernel_output:int,dropout:int = 0.2):
        super(Attention_layer,self).__init__()
        self.key_kernel = nn.Linear(kernel_input, kernel_output)
        self.query_kernel = nn.Linear(kernel_input, kernel_output)
        self.value_kernel = nn.Linear(kernel_input, kernel_output)
        self.normalize_kernel = nn.Linear(kernel_output,1,bias = False)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x:torch.Tensor, mask:torch.Tensor):
        """
        @params:
            x: (batch_size, max_seq_len, word_emb_size)
            mask: (batch_size, max_seq_len)
        """
        key = self.key_kernel(x)
        query = self.query_kernel(x)
        value = self.value_kernel(x)
        #batch_size,max_seq_len, hidden_size
        
        att_output = calculate_attention(query, key, value, mask)
        #batch_size, max_seq_len, hidden_size
        att_output = self.dropout(att_output)
        att_output = self.normalize_kernel(att_output)
        
        att_output = torch.softmax(att_output,-1)
        #batch_size, max_seq_len, 1
        return att_output


class BinarySynClassifierBaseOnAttention(nn.Module):
    """SynSet Classifier Based on Attention"""

    def __init__(self,config:Dict):
        super(BinarySynClassifierBaseOnAttention,self).__init__()
        self.name = config['name']
        self.version = config['version']
        self.embedding = config['embedding']
        self.attention_unit = config['attention']
        self.classifier = getFCLayer([config['mapper_hidden_size'][-1] * 5, *config['classifier_hidden_size'], 1],True,dropout=config['dropout'])
        self.mapper = getFCLayer([self.embedding.dim, *config['mapper_hidden_size']], True, dropout= config['dropout'])
        #self.mapper.add_module('activate',nn.ReLU())
    
    def forward(self,word_set:torch.Tensor,mask:torch.Tensor,waiting_word:torch.Tensor):
        """
        @params:
            word_set: torch.Tensor (batch_size, max_seq_len)
            waiting_word: torch.Tensor (batch_size)
            mask: torch.Tesnor (batch_size, max_seq_len)
        """
        word_set = self.embedding(word_set)
        waiting_word = self.embedding(waiting_word)
        # batch_size max_seq_len word_vec_size, batch_size, word_vec_size
        
        word_set_weight= self.attention_unit(word_set,mask);
        # batch_size, max_seq_len,1
        word_set_weight = word_set_weight.expand(-1,-1,word_set.size(-1))
        # batch_size, max_seq_len, word_vec_size

        word_set_vec = torch.sum(word_set_weight * word_set, dim = 1)
        # batch_size, word_vec_size

        word_set_vec = self.mapper(word_set_vec)
        waiting_word = self.mapper(waiting_word)

        com_feature = self.extract_feature(word_set_vec, waiting_word)

        ans = self.classifier(com_feature)
        # batch_size , 1
        return torch.sigmoid(ans).squeeze(-1)

    
    def extract_feature(self,x:torch.Tensor,y:torch.Tensor):
        add_feature = x + y
        divide_feature = x - y
        multiply_feature = x * y
        # batch_size, word_vec_size
        ans = torch.cat([add_feature,divide_feature,multiply_feature,x,y], dim = -1)
        return ans
    
    def test_predict_attention_weights(self,word_set:torch.Tensor,mask:torch.Tensor):
        word_set = self.embedding(word_set)
        word_set_weight = self.attention_unit(word_set, mask);
        # [batch_size * word_nums] = [ 1 * max_word_nums]
        return word_set_weight.squeeze(0)
    
    def test_predict_is_wordset(self,word_set:torch.Tensor, mask:torch.Tensor, waiting_word:torch.Tensor):
        word_set_ = self.embedding(word_set)
        waiting_word_ = self.embedding(waiting_word)

        word_set_weight = self.attention_unit(word_set_, mask);
        word_set_weight = word_set_weight.expand(-1,-1,word_set_.size(-1))
        word_set_vec = torch.sum(word_set_weight * word_set_, dim = 1)
        word_set_vec = self.mapper(word_set_vec)
        waiting_word_ = self.mapper(waiting_word_)

        com_feature = self.extract_feature(word_set_vec, waiting_word_)

        ans = self.classifier(com_feature)
        # batch_size , 1
        return torch.sigmoid(ans).squeeze(-1)
    
    def save(self,dir_path:Path):
        
        name = self.name
        version = self.version
        filename = name+ '_' + version + '.pkl'
        dir_path.joinpath(filename)
        
        torch.save(self,str(dir_path))


    @classmethod
    def load(cls,path:Path):
        return torch.load(str(path))

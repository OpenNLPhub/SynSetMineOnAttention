'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-10 16:21:07
 * @desc 
'''

from typing import List
import torch
from torch._C import set_flush_denormal
import torch.nn as nn
from torch.nn.modules import dropout
from transformers import BertModel
from config import BertPretrainedModelPath
import numpy as np

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
        bias = True if ix == len(L) - 1 else False
        L.append(nn.Linear(i,j,bias=bias))
        if add_dropout:
            L.append(nn.Dropout(dropout))
    return nn.Sequential(*L)
        

class SynSetClassfier(nn.Module):
    def __init__(self,component,config):
        super(SynSetClassfier,self).__init__()
        self.name = config['name']
        self.version = config['version']
        self.bert = component['bert']
        self.embedding = component['embedding']
        embedding_state_size_list = [
            self.embedding.dim,
            *config['embed_trans_hidden_size'],
            self.bert.embeddings.word_embeddings.embedding_dim
            ]
        self.embedding_map = getFCLayer(embedding_state_size_list)

        out_state_size_list = [
            self.bert.embeddings.word_embeddings.embedding_dim,
            *config['post_trans_hidden_size'],
            1
            ]
        self.output_map = getFCLayer(out_state_size_list,add_dropout=0.2,dropout=config['dropout'])

    def forward(self,input_ids,attention_mask,token_type_ids):
        """
        Args:
            input_ids: (batch_size, max_seq_len) it include two word set, raw word set and new word set which have waiting word. 
                        Use [SEP] to split two word set. Just Like predict two sentence
            attention_mask: (batch_size, max_seq_len) 
            token_type_ids: (batch_size, max_seq_len) it label the world whether it belongs to new wordset or old wordset
        """

        x = self.embedding(input_ids)
        # ( batch_size, max_seq_len, raw_emb_dim)
        x = self.embedding_map(x)
        # (batch_size, max_seq_len, bert_emb_dim)

        x = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            input_embeds=x
            )[1]
        # (batch_size, bert_emb_dim)

        x = self.output_map(x)
        # (batch_size, 1)
        x = x.squeeze(-1)
        x = torch.sigmoid(x)
        return x
        # batch_size

    def freeze_bert_paramter(self):
        for i in self.bert.parameters():
            i.requires_grad = False            
            
            

        


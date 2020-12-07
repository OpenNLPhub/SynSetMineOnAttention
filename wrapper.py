
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''

from model import Attention_layer
from os import wait, write
from pathlib import Path
from dataloader import DataSet
from datetime import datetime
import typing as __t
from numpy.lib.arraysetops import union1d
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from log import logger
from typing import Any, Callable, Dict, List, Optional, Sequence
from copy import deepcopy
from evaluate import EvalUnit,binary_confusion_matrix_evaluate
from base.basewrapper import BaseWrapper
from base.basedataloader import BaseDataLoader
from utils import set_padding

class ModelWrapper(BaseWrapper):
    """Class to wrapper training and testing of deeplearning model
    """
    def __init__(self, model:nn.Module, config:Dict):
        super(ModelWrapper,self).__init__(model,config)
        self.required_mode.append('threshold')
        self.threshold = self.getconfigattr('threshold',config=config)
        
        self.writer = SummaryWriter()

    def __check_before_work(self,keys:List[str]):
        for key in keys:
            if key not in self.__dict__:
                raise ValueError('{} is Missing in Wrapper instance')
    
    def __trans_np_to_tensor(self,item:Sequence[Any]):
        wordset, attention_mask, waiting_word, labels = item
        F = lambda x: torch.tensor(x).long().to(self.device)
        wordset = F(wordset)
        attention_mask = F(attention_mask)
        waiting_word = F(waiting_word)
        tensor_labels = torch.tensor(labels).float().to(self.device)
        return wordset, attention_mask, waiting_word, labels, tensor_labels
    
    def train_(self):
        self.__check_before_work(['dev_dataloader','train_dataloader'])
        self.model.to(self.device)
        t = range(self.start_epoch, self.epoches)
        all_step = len(self.train_dataloader)
        validation_flag = True if self.dev_dataloader is not None else False
        for epoch in t:
            self.model.train()
            epoch_unit = EvalUnit(0,0,0,0,'sum')
            ep_loss = 0.0
            for step,item in enumerate(self.train_dataloader):
                batch_word_set, attention_mask, waiting_word, labels , tensor_labels = self.__trans_np_to_tensor(item)
                pred_ans = self.model(batch_word_set, attention_mask, waiting_word)
                step_loss = self.loss_fn(pred_ans,tensor_labels) / tensor_labels.shape[0]
                ep_loss += step_loss.item()
                self.optimizer.zero_grad()
                step_loss.backward()
                self.optimizer.step()

                pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)
                unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
                epoch_unit += unit

                if (step + 1) % self.print_step == 0 or step + 1 == all_step:
                    logger.info('Training Epoch: {} step {}/{} Loss:{:.6f}'.format(
                        epoch,step + 1,all_step,step_loss
                    ))
            #Writer
            self.writer.add_scalar('Loss/train',ep_loss/all_step,epoch)
            #Train Evaluation
            logger.info('Evaluation Training Epoch:{}'.format(epoch))

            if validation_flag:
                # it will update model in validation method
                score = self.validation()
            else:
                #No Validation Model
                #update best model according to performance in Trainning DataSet
                score = epoch_unit.f1_score()
            
            self.writer.add_scalar('F1_score in Validation:',score,epoch)
            
            # using f1_score to update standard line
            if self.best_score < score:
                self.best_score = score
                self.best_model = deepcopy(self.model)
                self.save_check_point()
                logger.info("Update BEST MODEL !")
            
            logger.info("Training Evaluation Epoch :{}".format(epoch))
            logger.info(epoch_unit)

            if (epoch + 1) % self.checkpoint_epoch == 0 or epoch + 1 == self.epoches:
                self.save_check_point(epoch=epoch)


            
    def train(self,train_dataloader:BaseDataLoader, dev_dataloader:Optional[BaseDataLoader]):
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        self.train_()
  
    def validation(self):
        self.model.eval()
        all_step = len(self.dev_dataloader)
        validation_unit = EvalUnit(0,0,0,0,'Evaluation')

        for step, item in enumerate(self.dev_dataloader):
            batch_word_set, attention_mask, waiting_word, labels , tensor_labels = self.__trans_np_to_tensor(item)
            pred_ans = self.model(batch_word_set, attention_mask, waiting_word)
            step_loss = self.loss_fn(pred_ans,tensor_labels) / tensor_labels.shape[0]
            pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)
            unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
            validation_unit += unit

            if (step + 1) % self.print_step == 0 or step + 1 == all_step:
                logger.info('Validation {}/{} Loss: {:.6f}'.format(step, all_step,step_loss))

        logger.info("Validation Evaluation:")
        logger.info(validation_unit)
        return validation_unit.f1_score()

    def test_performance_(self):
        self.__check_before_work(['test_dataloader'])
        '''
        Test Performance
        '''
        self.best_model.eval()
        test_unit = EvalUnit(0,0,0,0,'Test')
        for step,item in enumerate(self.test_dataloader):
            batch_word_set, attentionn_mask, waitinng_word, labels, _ = self.__trans_np_to_tensor(item)
            pred_ans = self.best_model(batch_word_set, attentionn_mask, waitinng_word)
            pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)      
            unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
            test_unit += unit
        logger.info("Test Performance Evaluation:")
        logger.info(test_unit)

    def test_performance(self,test_dataloader:BaseDataLoader):
        self.test_dataloader = test_dataloader
        self.test_performance_()

    def cluster_predict(self,dataset:DataSet,word2id:Dict,outputfile:Optional[Path]) -> Sequence[Any]:
        """Using Binary Classifer to cluster wordset
        Args:
            dataset: it's self defined class, in DataSet, we use vocab to get all words and true cluster result
            word2id: it is got from embedding file, translate word to embedding index
            outputfile: outputfile path
        Returns:
            List of word sets
        """
        self.best_model.eval()
        vocab = dataset.vocab
        words = vocab.keys()
        wordset_list = []
        for word in words:
            wordid = word2id[word]
            if not wordset_list:
                #Empty
                wordset_list.append([wordid])
                continue
            itemsum = len(wordset_list)
            tmp_best_scores = 0
            index = 0
            for ix in range(0,itemsum,self.batch_size):
                batch_word_set = wordset_list[ix:ix+self.batch_size]
                batch_waiting_word = [wordid] * len(batch_word_set)
                batch_word_set, attention_mask = set_padding(batch_word_set)
                batch_word_set, attention_mask, batch_waiting_word, _, _ = self.__trans_np_to_tensor(
                    [batch_word_set, attention_mask, batch_waiting_word, 0]
                )
                scores = self.best_model(batch_word_set, attention_mask, batch_waiting_word)
                best_scores = torch.max(scores).item()
                
                if best_scores >= tmp_best_scores:
                    tmp_best_scores = best_scores
                    index = ix + torch.argmax(scores).item()
            
            if tmp_best_scores > self.threshold:
                wordset_list[index].append(wordid)
            else:
                wordset_list.append([wordid])
        
        #id2word
        # import pdb;pdb.set_trace()
        id2word = { j:i for i,j in word2id.items()}
        F = lambda x:[ id2word[i] for i in x]
        pred_word_sets = [ F(wordset) for wordset in wordset_list]
        
        if outputfile is not None:
            with open(outputfile, 'w', encoding='utf-8') as f:
                for pred_word_set in pred_word_sets:
                    for word in pred_word_set:
                        f.write(word+' ')
                    f.write('\n')

        return pred_word_sets

    def evaluate(self, dataset:DataSet, pred_word_sets:Sequence[Any], function_list:Sequence[Callable[...,float]])->Sequence[Any]:
        """ Use Evaluating Function to Evaluate the final result
        Args:
            dataset: it's self defined class, we use vocab attribute to get true cluster result
            pred_word_set: the output of cluster_predict method | List of word sets
            function_list: the list of evaluating function which have two input pred_cluster and target_cluster
        """
        #trans datatype 
        clusters = set(dataset.vocab.values())
        cluster2id = {cluster:idx for idx,cluster in enumerate(clusters)}
        target_cluster = {key:cluster2id[value] for key,value in dataset.vocab.items()}
        pred_cluster = {}
        # import pdb;pdb.set_trace()
        for idx,pred_word_set in enumerate(pred_word_sets):
            for word in pred_word_set:
                pred_cluster[word] = idx
        # import pdb;pdb.set_trace()
        ans = []
        for func in function_list:
            ans.append(func(pred_cluster = pred_cluster,target_cluster = target_cluster))
        self.writer.add_hparams(metric_dict={ name:f for name,f in ans})
        
        return ans
        
    """
    Below Methods are used to Test
    """
    def Test_predict_wordset_attention(self,word_set:List[str],word2id:Dict):
        self.best_model.eval();
        word_set_ = [ word2id[i] for i in word_set]
        word_set_, attention_mask = set_padding([word_set_])
        word_set_tensor = torch.tensor(word_set_).long().to(self.device)
        attention_mask = torch.tensor(attention_mask).long().to(self.device)
        attention_weight = self.best_model.test_predict_attention_weights(word_set_tensor, attention_mask)
        attention_weight = attention_weight.cpu().detach().numpy()
        d = {i:j for i,j in zip(word_set,attention_weight)}
        return d
    
    def Test_predict_is_wordset(self,word_set:List[str],waiting_word:str,word2id:Dict):
        self.best_model.eval();
        word_set_ = [ word2id[i] for i in word_set]
        waiting_word_ = word2id[waiting_word]
        word_set_, attentionn_mask = set_padding([word_set_])
        word_set_tensor = torch.tensor(word_set_).long().to(self.device)
        attention_mask = torch.tensor(attentionn_mask).long().to(self.device)
        waiting_word_tensor = torch.tensor([waiting_word_]).long().to(self.device)

        y = self.best_model(word_set_tensor,attention_mask,waiting_word_tensor)
        # batch_size , 1
        y = y.squeeze(0).cpu().detach().numpy()
        return y
        
        

    
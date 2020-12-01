
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''

from datetime import datetime
import typing as __t
from numpy.lib.arraysetops import union1d
import torch.nn as nn
import torch
import numpy as np
from log import logger
from typing import Any, Callable, Dict, List, Optional, Sequence
from copy import deepcopy
from evaluate import EvalUnit,binary_confusion_matrix_evaluate
from base.basewrapper import BaseWrapper
from base.basedataloader import BaseDataLoader

class ModelWrapper(BaseWrapper):
    """Class to wrapper training and testing of deeplearning model
    """
    def __init__(self, model:nn.Module, config:Dict):
        super(ModelWrapper,self).__init__(model,config)
        self.required_mode.append('threshold')
        self.threshold = self.getconfigattr('threshold',config=config)

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
                        epoch,step,all_step,step_loss
                    ))
            
            #Train Evaluation
            logger.info('Evaluation Training Epoch:{}'.format(epoch))

            if validation_flag:
                # it will update model in validation method
                score = self.validation()
            else:
                #No Validation Model
                #update best model according to performance in Trainning DataSet
                score = epoch_unit.f1_score()

            # using f1_score to update standard line
            if self.best_score < score:
                self.best_model = score
                self.best_model = deepcopy(self.model)
            logger.info("Training Evaluation Epoch :{}".format(epoch))
            logger.info(epoch_unit)
            

            
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





    
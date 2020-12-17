'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-12-17 14:11:20
 * @desc 
'''
'''
@author Waldinsamkeit
@email Zenglz_pro@163.com
@create date 2020-10-20 15:58:45
@desc [description]
'''

from pathlib import Path
import os

from dataloader import DataSet
import torch.nn as nn
import torch
import numpy as np
from log import logger
from typing import Any, Callable, Dict, ItemsView, List, Optional, Sequence
from copy import deepcopy
from evaluate import EvalUnit,binary_confusion_matrix_evaluate,cluster_metrics_eval
from base.basewrapper import BaseWrapper
from base.basedataloader import BaseDataLoader
from utils import set_padding
import pickle

class TransformerModelWrapper(BaseWrapper):
    """Class to wrapper training and testing of deeplearning model
    """
    def __init__(self, model:nn.Module, config:Dict):
        super(TransformerModelWrapper,self).__init__(model,config)
        self.append_require_mode('threshold')
        self.threshold = self.getconfigattr('threshold',config=config)


    def __check_before_work(self,keys:List[str]):
        for key in keys:
            if key not in self.__dict__:
                raise ValueError('{} is Missing in Wrapper instance')
    

    def __trans_np_to_tensor(self,item:Sequence[Any]):
        old_word_set, old_mask, new_word_set, new_mask, labels, attention_pos = item
        F = lambda x: torch.tensor(x).long().to(self.device)
        old_word_set_ = F(old_word_set)
        old_mask_ = F(old_mask)
        new_word_set_ = F(new_word_set)
        new_mask_ = F(new_mask)
        tensor_attention_pos = torch.tensor(attention_pos).float().to(self.device)
        tensor_labels = torch.tensor(labels).float().to(self.device)
        return old_word_set_, old_mask_, new_word_set_, new_mask_, tensor_attention_pos, tensor_labels, labels, attention_pos

    def train_(self):
        self.__check_before_work(['dev_dataloader','train_dataloader'])
        self.model.to(self.device)
        t = range(self.start_epoch, self.epoches)
        all_step = len(self.train_dataloader)
        validation_flag = True if self.dev_dataloader is not None else False
        for epoch in t:
            self.model.train()
            epoch_unit = EvalUnit(0,0,0,0,'Training')
            ep_loss = 0.0
            ep_attention_loss = 0.0

            for step,item in enumerate(self.train_dataloader):
                old_word_set, old_mask, new_word_set, new_mask, tensor_attention_pos, tensor_labels, labels, attention_pos = self.__trans_np_to_tensor(item)
                
                pred_ans = self.model(old_word_set, old_mask, new_word_set, new_mask)
                step_loss = self.loss_fn(pred_ans,tensor_labels) / tensor_labels.shape[0]
                # attention_pred_ans = self.model.cal_attention_weight(new_word_set, new_mask)
                #cal attention weight loss
                mask = new_mask == 1
                # p = torch.masked_select(attention_pred_ans,mask)
                # t = torch.masked_select(tensor_attention_pos,mask)
                # step_attention_loss = self.loss_fn(p,t) / p.shape[0]
                
                # loss = step_loss + step_attention_loss
                loss = step_loss
                ep_loss += step_loss.item()
                # ep_attention_loss += step_attention_loss.item()
                

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)
                unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
                epoch_unit += unit

                if (step + 1) % self.print_step == 0 or step + 1 == all_step:
                    logger.info('Training Epoch: {} step {}/{} Loss:{:.6f}'.format(
                        epoch,step + 1,all_step, loss.item()
                    ))
            #Train Evaluation
            logger.info("Training Evaluation Epoch :{}".format(epoch))
            logger.info(epoch_unit)

            val_loss = 0.0
            val_attention_loss = 0.0

            val_unit = EvalUnit(0,0,0,0,'validation')
            val_attention_unit = EvalUnit(0,0,0,0,'validation attention Unit')
            cluster_unit = {'FMI':0.0, 'NMI':0.0, 'ARI':0.0}
            score = 0

            if validation_flag:
                # it will update model in validation method
                val_loss, val_attention_loss, val_unit, val_attention_unit = self.validation()
                if epoch > self.epoches / 10:
                    cluster_unit = self.validation_cluster_metrics()
                    score = cluster_unit['ARI']
            else:
                #No Validation Model
                #update best model according to performance in Trainning DataSet
                score = epoch_unit.f1_score()
                  
            # using f1_score to update standard line
            if self.best_score < score:
                self.best_score = score
                self.best_model = deepcopy(self.model)
                self.save_check_point()
                logger.info("Update BEST MODEL !")
            
            

            # if (epoch + 1) % self.checkpoint_epoch == 0 or epoch + 1 == self.epoches:
            #     self.save_check_point(epoch=epoch)

            yield  (
                    ep_loss/all_step, 
                    ep_attention_loss/all_step,
                    epoch_unit,
                    val_loss,
                    val_attention_loss,
                    val_unit,
                    val_attention_unit,
                    cluster_unit,
                    self.best_score
                    )
            
    def train(self,train_dataloader:BaseDataLoader, dev_dataloader:Optional[BaseDataLoader]):
        self.train_dataloader = train_dataloader
        self.dev_dataloader = dev_dataloader
        vocab = self.dev_dataloader.data.vocab
        word2id = self.dev_dataloader.word2id
        target_wordset_list = {}
        for k,v in vocab.items():
            if v not in target_wordset_list:
                target_wordset_list[v] = [word2id[k]]
            else:
                target_wordset_list[v].append(word2id[k])
        self.target_wordset_list = list(target_wordset_list.values())

        for item in self.train_():
            yield item

    def attention_check(self, pred, tag, mask):
        pass

    def validation(self):
        self.model.eval()
        all_step = len(self.dev_dataloader)
        validation_unit = EvalUnit(0,0,0,0,'Evaluation')
        validation_attention_unit = EvalUnit(0,0,0,0,'Evaludation Attention')
        loss = 0.0
        attention_loss = 0.0
        for step, item in enumerate(self.dev_dataloader):
            old_word_set, old_mask, new_word_set, new_mask, tensor_attention_pos, tensor_labels, labels, attention_pos = self.__trans_np_to_tensor(item)   
            pred_ans = self.model(old_word_set, old_mask, new_word_set, new_mask)
            step_loss = self.loss_fn(pred_ans,tensor_labels) / tensor_labels.shape[0]
            #  attention_pred_ans = self.model.cal_attention_weight(new_word_set, new_mask)
            #cal attention weight loss
            mask = new_mask == 1
            # p = torch.masked_select(attention_pred_ans,mask)
            # t = torch.masked_select(tensor_attention_pos,mask)
            # step_attention_loss = self.loss_fn(p,t) / p.shape[0]
            
            loss += step_loss.item()
            # attention_loss += step_attention_loss.item()

            # attention_p = np.where(p.cpu().detach().numpy() > self.threshold, 1, 0)
            # attention_t = t.long().cpu().detach().numpy()
            # attention_unit = binary_confusion_matrix_evaluate(attention_t, attention_p)

            pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)
            unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
            validation_unit += unit
            # validation_attention_unit += attention_unit

            if (step + 1) % self.print_step == 0 or step + 1 == all_step:
                logger.info('Validation {}/{} Loss: {:.6f}'.format(step, all_step,step_loss))

        logger.info("Validation Evaluation:")
        logger.info(validation_unit)
        logger.info(validation_attention_unit)
        return loss/all_step, attention_loss/all_step , validation_unit, validation_attention_unit
    

    def validation_cluster_metrics(self):
        self.model.eval()
        vocab = self.dev_dataloader.data.vocab
        word2id = self.dev_dataloader.word2id
        pred_wordset_list = self.__cluster_predict(self.model,vocab=vocab,word2id=word2id)
        ans = self.__evaluate_cluster(pred_wordset_list,self.target_wordset_list)
        return { i:j for i,j in ans}
    

    def test_performance_(self):
        self.__check_before_work(['test_dataloader'])
        '''
        Test Performance
        '''
        self.best_model.eval()
        test_unit = EvalUnit(0,0,0,0,'Test')
        test_attention_unit = EvalUnit(0,0,0,0,'Test Attention')
        for step,item in enumerate(self.test_dataloader):
            old_word_set, old_mask, new_word_set, new_mask, tensor_attention_pos, tensor_labels, labels, attention_pos = self.__trans_np_to_tensor(item)   
            pred_ans = self.best_model(old_word_set, old_mask, new_word_set, new_mask)
            attention_pred_ans = self.best_model.cal_attention_weight(new_word_set, new_mask)
            #cal attention weight loss
            mask = new_mask == 1
            p = torch.masked_select(attention_pred_ans,mask)
            t = torch.masked_select(tensor_attention_pos,mask)
            
            attention_p = np.where(p.cpu().detach().numpy() > self.threshold, 1, 0)
            attention_t = t.long().cpu().detach().numpy()
            attention_unit = binary_confusion_matrix_evaluate(attention_t, attention_p)

            pred_labels = np.where(pred_ans.cpu().detach().numpy() > self.threshold,1,0)
            unit = binary_confusion_matrix_evaluate(np.array(labels), pred_labels)
            test_unit += unit
            test_attention_unit += attention_unit
        logger.info("Test Performance Evaluation:")
        logger.info(test_unit)
        logger.info(test_attention_unit)
        
        return test_unit.metrics2dict()

    def test_performance(self,test_dataloader:BaseDataLoader):
        self.test_dataloader = test_dataloader
        return self.test_performance_()
    

    def __cluster_predict(self,model:nn.Module,vocab:Dict, word2id:Dict)->Sequence[Any]:
        model.eval()
        words = vocab.keys()
        wordset_list = []
        for word in words:
            wordid = word2id[word]
            if not wordset_list:
                # Empty
                wordset_list.append([wordid])
                continue
            new_wordset = [[*i, wordid] for i in wordset_list]
            old_wordset = deepcopy(wordset_list)
            itemsnum = len(new_wordset)
            # add batch operation
            tmp_best_scores = 0
            index = 0
            for ix in range(0,itemsnum, self.batch_size):
                batch_new_wordset = new_wordset[ix:ix+self.batch_size]
                batch_old_wordset = old_wordset[ix:ix+self.batch_size]
                batch_old_wordset, mask = set_padding(batch_old_wordset)
                batch_new_wordset, mask_ = set_padding(batch_new_wordset)
                #batch_size * max_word_set_size
                F = lambda x: torch.Tensor(x).long().to(self.device)
                word_set_tensor, mask, new_word_set_tensor, mask_ = [
                    F(i) for i in [batch_old_wordset, mask, batch_new_wordset, mask_]
                ]
                scores = self.best_model(word_set_tensor, mask, new_word_set_tensor, mask_)
                best_scores = torch.max(scores).item()
                if best_scores >= tmp_best_scores:
                    tmp_best_scores = best_scores
                    index = ix + torch.argmax(scores).item()
               

            if tmp_best_scores > self.threshold:
                wordset_list[index].append(wordid)
            else:
                wordset_list.append([wordid])
        return wordset_list

    def __evaluate_cluster(self,pred_wordset_list:Sequence[Any], target_wordset_list:Sequence[Any]):
        pred_cluster = {}
        # import pdb;pdb.set_trace()
        for idx,pred_word_set in enumerate(pred_wordset_list):
            for word in pred_word_set:
                pred_cluster[word] = idx
        
        target_cluster = {}
        for idx, target_word_set in enumerate(target_wordset_list):
            for word in target_word_set:
                target_cluster[word] = idx

        return cluster_metrics_eval(pred_cluster,target_cluster)


    '''Public  Method'''
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
        wordset_list = self.__cluster_predict(self.best_model,vocab=vocab,word2id=word2id)

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

    def evaluate(self, dataset:DataSet, pred_word_sets:Sequence[Any])->Sequence[Any]:
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
        return cluster_metrics_eval(pred_cluster,target_cluster)



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
        d = {i:j.item() for i,j in zip(word_set,attention_weight)}
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


    """ ---------------- OverWrite No Writer -----------------"""  
    def save(self,dir_path:Path):
        """ save this wrapper
        using pickle to save this wrapper 
        It is convinient for us to get entire wrapper without setting config
        """
        if os.path.isdir(dir_path):
            name = self.model.name
            version = self.model.version
            filename = name + "_" + version +"_wrapper.pkl" 
            filepath = dir_path.joinpath(filename)
        else:
            filepath = dir_path

        d = self.__dict__
        with open(filepath, 'wb') as f:
            pickle.dump(d, f)

    
    @classmethod
    def load(cls,dir_path:Path):
        """ load this wrapper
        using pickle to load this wrapper
        It is convinient for us to get entire wrapper without setting config
        """
        # f = open(self.filename, 'rb')
        # tmp_dict = cPickle.load(f)
        # f.close()          

        # self.__dict__.update(tmp_dict)
        if os.path.isdir(dir_path):
            flist = os.listdir(dir_path)
            if not flist:
                msg = 'No wrapper pickle file'
                raise ValueError(msg=msg)
            filepath = Path.joinpath(dir_path,max(flist))
        if os.path.isfile(dir_path):
            filepath = dir_path

        with open(dir_path, 'rb') as f:
            tmp_dict = pickle.load(f)
            return cls(tmp_dict['model'],tmp_dict)
        

    
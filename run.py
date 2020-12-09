'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-16 11:12:26
 * @desc 
'''

from typing import Any,Dict
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.tensorboard.summary import hparams
from dataloader import DataSetDir, DataSet, Dataloader, DataItemSet,select_sampler
from wrapper import ModelWrapper
from model import Embedding_layer, Attention_layer, BinarySynClassifierBaseOnAttention
from evaluate import select_evaluate_func
import config 
from config import TrainingConfig,OperateConfig,DataConfig,ModelConfig
from log import logger
from utils import set_random_seed

SEED = 2020

def test_clustertask(operateconfig:Dict,dataconfig:Dict, trainingconfig:Dict, modelconfig:Dict):
    #set registered hyper parameters
    hparams = config.register_hparams

    dir_path =  dataconfig['data_dir_path']
    comment = '_' + dir_path.name +'_'+modelconfig['name']+'_'+modelconfig['version']
    metric_dict = {}
    w = SummaryWriter(comment = comment) if operateconfig['plot'] else None
    
    if not dir_path:
        raise KeyError

    datasetdir = DataSetDir(dir_path)
    # combine model
    embedding_layer = Embedding_layer.from_pretrained(datasetdir.embedding_vec)
    
    embedding_layer.freeze_parameters()
    attenion_layer = Attention_layer(embedding_layer.dim,modelconfig['attention_hidden_size'])
    modelconfig['attention'] = attenion_layer
    modelconfig['embedding'] = embedding_layer
    model = BinarySynClassifierBaseOnAttention(
                config = modelconfig
            )
    optimizer = optim.Adam(filter(lambda x : x.requires_grad , model.parameters()),lr=trainingconfig['lr'], amsgrad=True)
    trainingconfig['optim'] = optimizer
    trainingconfig['loss_fn'] = torch.nn.BCELoss()
    wrapper = ModelWrapper(model,trainingconfig)
    
    if operateconfig['resume']:
        wrapper.load_check_point()
        # continue to trainning

    if operateconfig['train']:
        train_datasetitem = DataItemSet(
                    dataset=datasetdir.train_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['negative_sample_size']
                ) 
        dev_datasetitem = DataItemSet(
                    dataset=datasetdir.test_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )
        train_dataloader = Dataloader(
                    dataitems=train_datasetitem, 
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        dev_dataloader = Dataloader(
                    dataitems=dev_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        
        #Plot in Tensorboard
        for ix,item in enumerate(wrapper.train(train_dataloader=train_dataloader,dev_dataloader=dev_dataloader)):
            ep_loss, t_ac, t_p, t_r, t_f1, v_loss, v_ac, v_p, v_r, v_f1, b_score = item
            if w:
                w.add_scalar("Training/Loss", ep_loss ,ix)
                w.add_scalar("Training/Accuracy", t_ac, ix )
                w.add_scalar("Training/Precision", t_p, ix)
                w.add_scalar("Training/Recall", t_p, ix)
                w.add_scalar("Training/F1_score", t_f1, ix)
                w.add_scalar("Validation/Loss",v_loss, ix)
                w.add_scalar("Validation/Accuracy", v_ac, ix)
                w.add_scalar("Validation/Precision", v_p, ix)
                w.add_scalar("Validation/Recall", v_r, ix)
                w.add_scalar("Validation/F1_score", v_f1, ix)
                w.add_scalar("Best Score Update", b_score, ix)
        
    if operateconfig['test']:
        test_datasetitem = DataItemSet(
                    dataset=datasetdir.test_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )

        test_dataloader = Dataloader(
                    dataitems=test_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        d = wrapper.test_performance(test_dataloader=test_dataloader)
        metric_dict = { **metric_dict, **d}
    if operateconfig['predict']:
        func_list = select_evaluate_func(operateconfig['eval_function'])

        pred_word_set = wrapper.cluster_predict(
                    dataset=datasetdir.test_dataset,
                    word2id=datasetdir.word2id,
                    outputfile=trainingconfig['result_out_dir'].joinpath(datasetdir.name+'_result.txt')
                )

        ans = wrapper.evaluate(datasetdir.test_dataset, pred_word_set,function_list=func_list)
        logger.info("{} DataSet Cluster Prediction".format(datasetdir.train_dataset.name))
        for name,f in ans:
            logger.info("{} : {:.5f}".format(name,f))
        
        if w:
            d = {i:j for i,j in ans}
            metric_dict = {**metric_dict, **d}
            w.add_hparams(hparams, metric_dict = metric_dict)
            w.close()
    wrapper.save(config.WRAPPER_DIR_PATH)

def NYT():
    DataConfig['data_dir_path'] = config.NYT_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

def PubMed():
    DataConfig['data_dir_path'] = config.PubMed_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

def Wiki():
    DataConfig['data_dir_path'] = config.Wiki_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

if __name__ == '__main__':
    set_random_seed(seed=SEED)
    NYT()
    #PubMed()
    #Wiki()
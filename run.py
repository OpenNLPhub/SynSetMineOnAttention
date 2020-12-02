'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-16 11:12:26
 * @desc 
'''

from typing import Any,Dict
import torch
import torch.optim as optim
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
    
    dir_path =  dataconfig['data_dir_path']

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
        wrapper.train(train_dataloader=train_dataloader,dev_dataloader=dev_dataloader)
    
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
        wrapper.test_performance(test_dataloader=test_dataloader)

    if operateconfig['predict']:
        func_list = select_evaluate_func(operateconfig['eval_function'])
        # import pdb;pdb.set_trace()
        pred_word_set = wrapper.cluster_predict(
                    dataset=datasetdir.test_dataset,
                    word2id=datasetdir.word2id,
                    outputfile=trainingconfig['result_out_dir'].joinpath(datasetdir.name+'_result.txt')
                )
        # import pdb;pdb.set_trace()
        ans = wrapper.evaluate(datasetdir.test_dataset, pred_word_set,function_list=func_list)
        logger.info("{} DataSet Cluster Prediction".format(datasetdir.train_dataset.name))
        for name,f in ans:
            logger.info("{} : {:.2f}".format(name,f))


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
    PubMed()
    Wiki()
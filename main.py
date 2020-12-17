'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-12-17 15:56:43
 * @desc 
'''

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
from dataloader import DataSetDir, DataSet, DataloaderTransformer, DataItemSet,select_sampler
from transformerwrapper import TransformerModelWrapper
from model import BinarySynClassifierBaseOnTransformer, Embedding_layer, Attention_layer, BinarySynClassifierBaseOnAttention
import Tconfig  as config
from Tconfig import TrainingConfig,OperateConfig,DataConfig,ModelConfig
from Tconfig import generate_register_hparams
from log import logger
from utils import set_random_seed
from args import parser
import os
from datetime import datetime
args = parser.parse_args()
SEED = 2020

def test_clustertask(operateconfig:Dict,dataconfig:Dict, trainingconfig:Dict, modelconfig:Dict):
    #set registered hyper parameters
    dir_path =  dataconfig['data_dir_path']
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = os.path.join(config.TENSORBOADRD_DIR_PATH,now+dir_path.name)
    logger.info("Register Hyper Parameter")
    hparams = generate_register_hparams(modelconfig,trainingconfig,dataconfig)
    metric_dict = {}
    w = SummaryWriter(log_dir = log_dir) if args.p else None
    if not dir_path:
        raise KeyError
    logger.info("Load Embedding Vector")
    datasetdir = DataSetDir(dir_path,word_emb_select=dataconfig['word_emb_select'])
    # combine model
    embedding_layer = Embedding_layer.from_pretrained(datasetdir.embedding_vec)
    
    embedding_layer.freeze_parameters()
    modelconfig['embedding'] = embedding_layer
    model = BinarySynClassifierBaseOnTransformer(
                config = modelconfig
            )
    optimizer = optim.Adam(filter(lambda x : x.requires_grad , model.parameters()),lr=trainingconfig['lr'], amsgrad=True)
    trainingconfig['optim'] = optimizer
    trainingconfig['loss_fn'] = torch.nn.BCELoss()
    
    wrapper = TransformerModelWrapper(model,trainingconfig)
    
    if operateconfig['resume']:
        wrapper.load_check_point()
        # continue to trainning

    if operateconfig['train']:
        logger.info("Generate DataLoader")
        train_datasetitem = DataItemSet(
                    dataset=datasetdir.train_dataset,
                    sampler = select_sampler(dataconfig['sample_strategy']),
                    negative_sample_size = dataconfig['negative_sample_size']
                ) 
        dev_datasetitem = DataItemSet(
                    dataset=datasetdir.test_dataset,
                    sampler = select_sampler(dataconfig['test_sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )
        train_dataloader = DataloaderTransformer(
                    dataitems=train_datasetitem, 
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        dev_dataloader = DataloaderTransformer(
                    dataitems=dev_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        logger.info("Start to Train !! ")

        #Plot in Tensorboard
        for ix,item in enumerate(wrapper.train(train_dataloader=train_dataloader,dev_dataloader=dev_dataloader)):
            ep_loss, ep_attention_loss, ep_unit, val_loss, val_attention_loss, val_unit, val_attention_unit, cluster_unit,b_score = item
            if w:
                w.add_scalar("Training/Loss", ep_loss ,ix)
                w.add_scalar("Training/AttentionLoss", ep_attention_loss, ix)
                w.add_scalar("Training/Precision", ep_unit.precision(), ix)
                w.add_scalar("Training/Recall", ep_unit.recall(), ix)
                w.add_scalar("Training/F1_score", ep_unit.f1_score(), ix)
                
                w.add_scalar("Validation/Loss",val_loss, ix)
                w.add_scalar("Validation/AttentionLoss",val_attention_loss, ix)

                w.add_scalar("Validation/Precision", val_unit.precision(), ix)
                w.add_scalar("Validation/Recall", val_unit.recall(), ix)
                w.add_scalar("Validation/F1_score", val_unit.f1_score(), ix)

                w.add_scalar("Validation/Precision", val_attention_unit.precision(), ix)
                w.add_scalar("Validation/Recall", val_attention_unit.recall(), ix)
                w.add_scalar("Validation/F1_score", val_attention_unit.f1_score(), ix)
                
                w.add_scalar("Validation/FMI", cluster_unit['FMI'], ix)
                w.add_scalar("Validation/ARI",  cluster_unit['ARI'], ix)
                w.add_scalar("Validation/NMI",cluster_unit['NMI'], ix)
                w.add_scalar("Best Score Update", b_score, ix)
        
    if operateconfig['test']:
        test_datasetitem = DataItemSet(
                    dataset=datasetdir.test_dataset,
                    sampler = select_sampler(dataconfig['test_sample_strategy']),
                    negative_sample_size = dataconfig['test_negative_sample_size']
                )

        test_dataloader = DataloaderTransformer(
                    dataitems=test_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        d = wrapper.test_performance(test_dataloader=test_dataloader)
        metric_dict = {**metric_dict, **d}
    
    if operateconfig['predict']:
        pred_word_set = wrapper.cluster_predict(
                    dataset=datasetdir.test_dataset,
                    word2id=datasetdir.word2id,
                    outputfile=trainingconfig['result_out_dir'].joinpath(datasetdir.name+'_result.txt')
                )
        ans = wrapper.evaluate(datasetdir.test_dataset, pred_word_set)
        logger.info("{} DataSet Cluster Prediction".format(datasetdir.train_dataset.name))
        for name,f in ans:
            logger.info("{} : {:.5f}".format(name,f))
        
        if w:
            d = {i:j for i,j in ans}
            metric_dict = {**metric_dict, **d}
            w.add_hparams(hparams, metric_dict = metric_dict)
            w.close()
    wrapper.save(config.WRAPPER_DIR_PATH.joinpath(datasetdir.name))

def NYT():
    DataConfig['data_dir_path'] = config.NYT_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

def PubMed():
    DataConfig['data_dir_path'] = config.PubMed_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

def Wiki():
    DataConfig['data_dir_path'] = config.Wiki_DIR_PATH
    test_clustertask(OperateConfig,DataConfig,TrainingConfig,ModelConfig)

def CSKB():
    DataConfig['data_dir_path'] = config.CSKB_DIR_PATH
    DataConfig['sample_strategy'] = 'sample_enumerate_size_enumerate'
    ModelConfig['attention_hidden_size'] = 768
    ModelConfig['classifier_hidden_size'] = [4096, 1024]
    ModelConfig['mapper_hidden_size'] = [1024, 4096]
    ModelConfig['dropout'] = 0.3
    TrainingConfig['epoches'] = 500
    TrainingConfig['lr'] = 1e-5
    test_clustertask(OperateConfig, DataConfig, TrainingConfig, ModelConfig)


def OMaha():
    DataConfig['data_dir_path'] = config.OMaha_DIR_PATH
    DataConfig['sample_strategy'] = 'sample_size_repeat_size'
    ModelConfig['attention_hidden_size'] = 768
    ModelConfig['classifier_hidden_size'] = [2048, 512]
    ModelConfig['mapper_hidden_size'] = [1024, 2048]
    ModelConfig['dropout'] = 0.3
    TrainingConfig['epoches'] = 200
    TrainingConfig['lr'] = 1e-5
    test_clustertask(OperateConfig, DataConfig, TrainingConfig, ModelConfig)

def run():
    set_random_seed(seed=SEED)
    if args.dataset == 'NYT':
        NYT()
    elif args.dataset == 'PubMed':
        PubMed()
    elif args.dataset == 'Wiki':
        Wiki()
    elif args.dataset == 'CSKB':
        CSKB()
    elif args.dataset == 'OMaha':
        OMaha()
    else:
        raise KeyError

if __name__ == '__main__':
    run()
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
from dataloader import DataSetDir, DataSet, Dataloader, DataItemSet,select_sampler
from wrapper import ModelWrapper
from model import Embedding_layer, Attention_layer, BinarySynClassifierBaseOnAttention
import config 
from config import TrainingConfig,OperateConfig,DataConfig,ModelConfig,generate_register_hparams
from log import logger
from utils import set_random_seed
from args import parser

args = parser.parse_args()
SEED = 2020

def test_clustertask(operateconfig:Dict,dataconfig:Dict, trainingconfig:Dict, modelconfig:Dict):
    #set registered hyper parameters
    logger.info("Register Hyper Parameter")
    hparams = generate_register_hparams(modelconfig,trainingconfig,dataconfig)

    dir_path =  dataconfig['data_dir_path']
    comment = '_' + dir_path.name +'_'+modelconfig['name']+'_'+modelconfig['version']
    metric_dict = {}
    w = SummaryWriter(comment = comment) if args.p else None
    if not dir_path:
        raise KeyError
    logger.info("Load Embedding Vector")
    datasetdir = DataSetDir(dir_path,word_emb_select=dataconfig['word_emb_select'])
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
        logger.info("Start to Train !! ")

        #Plot in Tensorboard
        for ix,item in enumerate(wrapper.train(train_dataloader=train_dataloader,dev_dataloader=dev_dataloader)):
            ep_loss, t_ac, t_p, t_r, t_f1, v_loss, v_ac, v_p, v_r, v_f1, cluster_unit, b_score = item
            if w:
                w.add_scalar("Training/Loss", ep_loss ,ix)
                w.add_scalar("Training/Accuracy", t_ac, ix )
                w.add_scalar("Training/Precision", t_p, ix)
                w.add_scalar("Training/Recall", t_r, ix)
                w.add_scalar("Training/F1_score", t_f1, ix)
                w.add_scalar("Validation/Loss",v_loss, ix)
                w.add_scalar("Validation/Accuracy", v_ac, ix)
                w.add_scalar("Validation/Precision", v_p, ix)
                w.add_scalar("Validation/Recall", v_r, ix)
                w.add_scalar("Validation/F1_score", v_f1, ix)
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

        test_dataloader = Dataloader(
                    dataitems=test_datasetitem,
                    word2id=datasetdir.word2id,
                    batch_size=trainingconfig['batch_size']
                )
        d = wrapper.test_performance(test_dataloader=test_dataloader)
        metric_dict = { **metric_dict, **d}
    
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
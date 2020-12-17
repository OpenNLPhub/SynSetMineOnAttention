'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-10 16:24:54
 * @desc 
'''
import os
from typing import Dict 
pwd = os.getcwd()
bertroot = os.path.join(pwd,'static')
BertPretrainedModelPath = {
    'bert-base-uncased' : os.path.join(bertroot,'bert-base-uncased')
}


from pathlib import Path
cwd = Path.cwd()

""" ------------- Path Config ------------- """
NYT_DIR_PATH = Path.joinpath(cwd,'data','NYT')
PubMed_DIR_PATH = Path.joinpath(cwd, 'data', 'PubMed')
Wiki_DIR_PATH = Path.joinpath(cwd, 'data', 'Wiki')
OMaha_DIR_PATH = Path.joinpath(cwd, 'data', 'OMaha')
CSKB_DIR_PATH = Path.joinpath(cwd, 'data', 'CSKB')


#Wrapper Dir config
WRAPPER_DIR_PATH = cwd.joinpath('result','wrapper')

#check point config
CHECK_POINT_DIR_PATH = cwd.joinpath('checkpoint')

#clustering result path
RESULT_DIR_PATH = cwd.joinpath('result')


""" ---------------- Own Config ----------- """
#default training Config
TrainingConfig = {
    'threshold' :  0.5,
    'epoches' : 100,
    'checkpoint_epoch' : 5,
    'print_step' : 15,
    'lr' : 1e-4,
    'checkpoint_dir' : CHECK_POINT_DIR_PATH,
    'batch_size' : 32,
    'result_out_dir' : RESULT_DIR_PATH,
    'cuda': 'cuda:0',
    'bert_freeze': False
}

#default Operate Config
OperateConfig = {
    'resume': False,
    'train' : True,
    'test' : True,
    'predict' : True,
}

#default dataconfig
DataConfig = {
    'data_dir_path' : None,
    'sample_strategy' : 'sample_large_size_enumerate',
    'negative_sample_size' : 20,
    'test_negative_sample_size' : 10,
    'word_emb_select': 'embed',
    'test_sample_strategy': 'sample_large_size_enumerate'
}

#default modelconfig
ModelConfig = {
    'name' : 'SynSetMineOnBase',
    'version' : 'v1.1.2',
    'attention_hidden_size': 512,
    'classifier_hidden_size': [1024,256],
    'mapper_hidden_size': [256,512],
    'dropout': 0.5
}


def generate_register_hparams(modelconfig:Dict, trainingconfig:Dict, dataconfig:Dict):
    register_hparams = { **modelconfig, **trainingconfig, **dataconfig } 
    register_hparams.pop('checkpoint_dir')
    register_hparams.pop('result_out_dir')
    register_hparams.pop('data_dir_path')
    register_hparams['classifier_hidden_size'] = str(register_hparams['classifier_hidden_size'])
    register_hparams['mapper_hidden_size'] = str(register_hparams['mapper_hidden_size'])
    return register_hparams
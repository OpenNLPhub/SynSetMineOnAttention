'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-10 16:24:54
 * @desc 
'''
import os 
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

""" ---------------- Own Config ----------- """
#default training Config
TrainingConfig ={
    'loss_fn' : 'crossEntropy',
    'threshold' :  0.5,
    'epoch' : 100,
    'checkpoint_epoch' : 5,
    'print_step' : 15,
    'lr' : 1e-4,
    'checkpoint_dir' : cwd.joinpath('checkpoint'),
    'batch_size' : 32,
    'result_out_dir' : cwd.joinpath('result'),
    'cuda': 'cuda:0',
    'bert_freeze': False
}

#default Operate Config
OperateConfig = {
    'resume': False,
    'train' : True,
    'test' : True,
    'predict' : True,
    'eval_function':['ARI','NMI','FMI']
}

#default dataconfig
DataConfig = {
    'data_dir_path' : None,
    'sample_strategy' : 'sample_size_repeat_size',
    'negative_sample_size' : 20,
    'test_negative_sample_size' : 10
}

#default modelconfig
ModelConfig = {
    'name' : 'SynSetMineOnBase',
    'version' : 'v1.0.0',
    'embed_trans_hidden_size' : [250],
    'post_trans_hidden_size' : [256],
    'dropout' : 0.2,
}


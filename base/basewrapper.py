'''
 * @author Waldinsamkeit
 * @email Zenglz_pro@163.com
 * @create date 2020-11-29 15:02:50
 * @desc BaseWrapper for model wrapper in deeplearning
'''
import pickle
from typing import Any, Dict, Optional
import typing as __t
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import warnings
from base.basedataloader import BaseDataLoader

class BaseWrapper(object):
    """Base class for Wrapper
    """
    def __init__(self,model:nn.Module, config:Dict):
        super(BaseWrapper,self).__init__()
        self.required_mode = [
            'loss_fn', 'checkpoint_epoch', 'print_step', 'batch_size', 'lr', 'checkpoint_dir', 'cuda',
            'optim', 'epoches'
            ]
        F = lambda key: self.getconfigattr(key,config)
        
        self.model = model
        self.device = torch.device(config['cuda']) if torch.cuda.is_available() else torch.device('cpu')
        
        self.loss_fn = F('loss_fn')
        self.start_epoch = 0
        self.epoches = F('epoches')
        self.checkpoint_epoch = F('checkpoint_epoch')
        self.print_step = F('print_step')
        self.init_lr = F('lr')
        self.best_model = self.model
        self.batch_size = F('batch_size')
        self.checkpoint_dir = F('checkpoint_dir')

        #data_loader
        self.train_dataloader = F('train_dataloader')
        self.dev_dataloader = F('dev_dataloader')
        self.test_dataloader = F('test_dataloader')

        #subclass have to overwrite
        self.optimizer = F('optim')
        self.best_score = 0.0
        
    def getconfigattr(self,key:str,config:Dict)->Any:
        if key not in config:
            if key in self.required_mode:
                raise ValueError('Attribute {} is required in this model wrapper'.format(key))
            else:
                desc = 'Attribute {} is not in config'.format(key)
                warnings.warn(desc)
                return None
        return config[key]

    def append_require_mode(self,key:str)->Any:
        if key not in self.required_mode:
            self.required_mode.append(key)

    def train(self,train_dataloader:BaseDataLoader, dev_dataloader:Optional[BaseDataLoader])->None:
        raise NotImplementedError()       

    def train_(self):
        raise NotImplementedError()
    

    def validation(self):
        raise NotImplementedError()

    def test_performance(self,test_dataloader:BaseDataLoader)->None:
        raise NotImplementedError()

    def test_performance_(self):
        raise NotImplementedError()
    
    
    def save_check_point(self, filename:str = None, epoch:int = None) -> None:
        """ save the key attribute in this wrapper
        key attribute:
        d = {
            'epoch' : epoch,
            'state_dict' : model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'best_score' : best_score,
            'best_model' : best_pred
        }
        """
        dir_path = Path(self.checkpoint_dir)
        now = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if filename is None:
            if epoch is None:
                filename = self.model.name+'_'+self.model.version+'_'+'best_model'+'_checkpoint.pth.tar'
            filename = self.model.name+'_'+self.model.version+'_'+'epoch:'+str(epoch)+'_'+now+'_checkpoint.pth.tar'
        
        filepath = Path.joinpath(dir_path, filename)
        d = {
            'epoch':self.start_epoch,
            'state_dict' : self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'best_model' : self.best_model.state_dict(),
            'best_score' : self.best_score
        }
        torch.save(d, filepath)
    
    def load_check_point(self,file_name:Optional[str] = None) -> None:
        """load the save model file
        Args:
            file_name: filename not filepath, it will concatenate the checkpoint_dir and filename 
                to create the whole path. It filename is none, it will choose the latest checkpoint file
                under the checkpoint directory.
        """
        dir_path = Path(self.checkpoint_dir)
        if file_name == None:
            import os
            flist = os.listdir(dir_path)
            if not flist:
                msg = 'No checkpoint file'
                raise ValueError(msg=msg)
            filepath = Path.joinpath(dir_path,max(flist))
        else:
            filepath = Path.joinpath(dir_path,file_name)
        
        checkpoint = torch.load(filepath, map_location='cpu')
        self.start_epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.best_model.load_state_dict(checkpoint['best_model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.best_score = checkpoint['best_score']
    

    def save(self,dir_path:Path):
        """ save this wrapper
        using pickle to save this wrapper 
        It is convinient for us to get entire wrapper without setting config
        """
        name = self.model.name
        version = self.model.version
        filename = name + "_" + version + "_wrapper.pkl" 
        filepath = dir_path.joinpath(filename)
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        
    
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
        with open(dir_path, 'rb') as f:
            tmp_dict = pickle.load(f)
            return cls(tmp_dict['model'],tmp_dict)

        
        

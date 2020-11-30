
class BaseDataLoader(object):
    def __init__(self):
        pass
    
    def __iter__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
from pynnet.base import *

__all__ = ['StandardTrainer']

def fnone():
    pass

class StandardTrainer(BaseObject):
    def __init__(self, max_iter, stopper):
        self.max_iter = max_iter
        self.stopper = stopper

    def train(self, model, dataset, checkpoint=fnone):
        train = dataset[0]
        valid = dataset[1]
        test = dataset[2]
        
        self.stopper.init_params(model, valid)

        for e in xrange(self.max_iter):
            score = model.partial_fit(train)
            checkpoint()
            if stopper.check(e, score):
               return stopper.result()  

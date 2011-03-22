from pynnet.base import *

__all__ = ['BasicTrainer']

def _fnone():
    pass

class BasicTrainer(BaseObject):
    def __init__(self, max_iter, stopper):
        self.max_iter = max_iter
        self.stopper = stopper

    def train(self, model, dataset, checkpoint=_fnone):
        r"""
        Train `model` on `dataset` calling `checkpoint` every
        iteration.

        The dataset is a 3-tuple of (train, valid, test).  Further
        interpretation is left to the model.
        
        The model must provide the partial_fit() method with the
        appropriate semantics.

        The checkpoint function is an optional function called at a
        safe point for saving and resuming the training.

        Safe point is defined here as:
          - No C frames in the stack
          - No unwrapped dataset on the stack
        """
        train = dataset[0]
        valid = dataset[1]
        test = dataset[2]
        
        self.stopper.init_params(model, valid)

        for e in xrange(self.max_iter):
            score = model.partial_fit(train)
            checkpoint()
            if self.stopper.check(e, score):
                return

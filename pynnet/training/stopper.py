from pynnet.base import *

__all__ = ['NullStopper', 'EarlyStopper']

class NullStopper(BaseObject):
    r"""
    The simplest possible stopper.

    Also doubles as API reference.
    """
    def init_params(self, model, valid):
        r"""
        Initialize run-specific parameters.

        This function is responsible to initialize any model or
        dataset-specific parameter the stopper might have as well as
        resetting any statistics that are kept while training runs to
        their initial value.

        Stopper-specific parameters should be supplied to the __init__
        method.
        """
        pass

    def check(self, epoch, score):
        r"""
        Returns True if training should continue.
        """
        return True
    
class EarlyStopper(BaseObject):
    r"""
    Stopper that uses a early stopping criterion.
    """
    def __init__(self, patience=2000, improvement_treshold=0.995, 
                 check_every=100):
        self.patience_increase = patience
        self.check_every = check_every
        self.improvement_treshold = improvement_treshold

    def init_params(self, model, valid):
        r"""
        Prep stopper for a training session.
        """
        self.patience = self.patience_increase
        self.best_valid_score = float('inf')
        self.best_iter = 0
        self.model = model
        self.valid = valid
    
    def check(self, epoch, score):
        r"""
        Check every `check_every` epoch whether the score has improved
        enough.

        This will wait at most `patience` epochs between significant
        improvements.  It will terminate training when patience is
        exhausted.
        """
        if epoch % self.check_every != 0:
            return True

        valid_score = self.model.eval(self.valid)
        if valid_score < self.best_valid_score:
            if valid_score < self.best_valid_score * self.improvement_treshold:
                self.patience = epoch + self.patience_increase
            self.best_valid_score = valid_score

        return self.patience >= epoch
        

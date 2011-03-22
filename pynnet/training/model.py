from pynnet.base import *

__all__ = []

class StandardModel(BaseObject):
    r"""
    API reference for the common functions that should be provided by
    a model and the context of their use.

    Models are not required to inherit from this class.
    """
    def partial_fit(self, data):
        r"""
        Make one "pass" over the data and adjust parameters
        accordingly.

        In a standard gradient-based model this would correspond to
        one update of the weights.
        
        This must return the cost associated with the current
        parameters as per the `eval` method.
        """
        raise NotImplementedError('partial_fit')

    def eval(self, data):
        r"""
        Evaluate the performance of the model on the specified data.
        
        This function may be needed by the stopper but is otherwise
        not required for training.
        """
        raise NotImplementedError('eval')

class SubModel(StandardModel):
    r"""
    Convinience class for a model that has more than one phase of
    training.

    This class can serve to bundle prebuilt training function to pass
    to a particular trainer.
    """
    def __init__(self, partialf, evalf):
        if partialf:
            self.partial_fit = partialf
        if evalf:
            self.eval = evalf

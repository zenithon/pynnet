from base import *

from nlins import *
from errors import *
from layers import *

__all__ = ['SimpleNet']

class SimpleNet(BaseObject):
    r"""
    Simple 3-layers neural network to experiment with new algorithms
    or implementations.

    WARNING: Not all functionality of this class is guaranteed to
    always work.  You should do some sanity checks before trusting
    results.

    If you are not tinkering with the implementation, you should use
    the `pynnet.NNet` class which is documented and stable.
    """
    
    def __init__(self, ninputs, nhidden, noutputs, hnlin=tanh,
                 onlin=none, error=mse, build_funcs=True):
        r"""
        Parameters:
        ninputs : Dimension of input
        nhidden : Dimension of hidden layer
        noutput : Dimension of output
        hnlin : hidden transfer function
        onlin : output transfer function
        error : Cost function to optimize
        """
        self.layer1 = HiddenLayer(ninputs, nhidden, activation=hnlin)
        self.layer2 = HiddenLayer(nhidden, noutput, activation=onlin)
        self.err = error
        self._set_reg()

        if build_funcs:
            x = theano.tensor.matrix('x')
            y = theano.tensor.matrix('y')
            self.build(x, y)
            self.test = theano.function([x, y], self.error)
            self.eval = theano.function([x], self.output)
    
    def _set_reg(self):
        self.L1 = self.layer1.L1 + self.layer2.L1
        self.L2_sqr = self.layer1.L2_sqr + self.layer2.L2_sqr

    def build(self, input, target):
        self.layer1.build(input)
        self.layer2.build(self.layer1.output)
        self.output = self.layer2.output
        self.params = self.layer1.params + self.layer2.params
        self.error = self.err(self.output, target)
    
    def _save_(self, file):
        pickle.dump(self.err, file, pickle.HIGHEST_PROTOCOL)
        self.layer1.save(file)
        self.layer2.save(file)
        
    def _load_(self, file):
        self.err = pickle.load(file)
        self.layer1 = HiddenLayer.load(file)
        self.layer2 = HiddenLayer.load(file)
        self._set_reg()

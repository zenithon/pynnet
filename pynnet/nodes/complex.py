from .base import *
from pynnet.nlins import *

__all__ = ['SharedComplexNode', 'ComplexNode']

class SharedComplexNode(BaseNode):
    def __init__(self, input, W1, W2, b, nlin=tanh, rng=numpy.random, 
                 dtype=theano.config.floatX, name=None):
        BaseNode.__init__(self, [input], name)
        self.nlin = nlin
        self.W1 = W1
        self.W2 = W2
        self.b = b

    def transform(self, x):
        # I am unsure about this form
        return self.nlin(T.sqrt(T.dot(x, self.W1)**2 + T.dot(self.W2)**2)+self.b)

class ComplexNode(BaseNode):
    def __init__(self, input, n_in, n_out, nlin=tanh, rng=numpy.random, 
                 dtype=theano.config.floatX, name=None):
        w_range = numpy.sqrt(6./(n_in+n_out))
        W1_values = rng.uniform(low=-w_range, high=w_range, 
                                size=(n_in, n_out)).astype(dtype)
        W2_values = rng.uniform(low=-w_range, high=w_range, 
                                size=(n_in, n_out)).astype(dtype)
        b = theano.shared(numpy.zeros((n_out,), dtype=dtype))
        SharedComplexNode.__init__(self, input, W1, W2, b, nlin=nlin, 
                                   name=name)
        self.local_params = [self.W1, self.W2, self.b]

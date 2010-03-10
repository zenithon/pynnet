import theano.tensor as T

__all__ = ['tanh', 'sigmoid', 'none']

from theano.tensor import tanh
from theano.tensor.nnet import sigmoid

def none(x):
    return x

import theano.tensor as T

__all__ = ['tanh', 'sigmoid', 'softmax', 'none']

from theano.tensor import tanh
from theano.tensor.nnet import sigmoid, softmax

def none(x):
    return x

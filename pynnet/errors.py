r"""
This module groups functions implementing error calculations.

All function exported by this module (the names listed in __all__)
must comply with this interface:

Parameters (the names don't have to be the same):
os -- the symbolic variable for the output
y -- the symbolic variable for the targets

Returns:
A single symbolic expression that computes the mean of the error over
all exemples.
"""
from base import *
import theano.tensor as T

__all__ = ['mse', 'nll']

def nll(os, y):
    r"""
    Computes the negative log likelyhood.

    Inputs:
    os -- probabilites for each class
    y -- integer label for the good class

    Tests:
    >>> os = T.fmatrix('os')
    >>> y = T.ivector('y')
    >>> out = nll(os, y)
    >>> theano.pp(out)
    '(-(sum(<theano.tensor.basic.AdvancedSubtensor object at ...>(log(os), ARange(0, y.shape[0], 1), y)) / float32(<theano.tensor.basic.AdvancedSubtensor object at ...>(log(os), ARange(0, y.shape[0], 1), y).shape)[0]))'
    >>> f = theano.function([os, y], out)
    >>> r = f(numpy.random.random((10, 10)), numpy.random.randint(0, 10, size=(10,)))
    >>> r.shape
    ()
    >>> r.dtype
    dtype('float32')
    """
    return -T.mean(T.log(os)[T.arange(y.shape[0]),y])

def mse(os, y):
    r"""
    Mean square error between `os` and `y`.
    """
    return T.mean((os-y)**2)

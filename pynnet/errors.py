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

__all__ = ['mse', 'nll', 'class_error', 'cross_entropy']

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

def class_error(os, y):
    r"""
    Classification error.

    os -- probabilities for each class (per example)
    y -- integer vector of the correct label
    """
    return T.mean(T.neq(T.argmax(os, 1), y))

def cross_entropy(os, y):
    r"""
    Cross-entropy cost.
    
    os -- probabilites for each class (per example)
    y -- target probabilites (usually 1-hot for each example)
    """
    return T.mean(-T.sum(y*T.log(os) + (1-y)*T.log(1-os), axis=1))

def scaled_cross_entropy(os, y):
    r"""
    Cross-entropy cost with inputs scaled from [-1, 1] to [0, 1].

    This is useful for using cross-entropy with tanh.
    
    os -- tanh output
    y -- target probabilites (usually 1-hot for each example)
    """
    return cross_entropy((os+1)/2, y)

def cross_entropy_softmax(os, y):
    r"""
    Cross-entropy cost for softmax output.
    
    os -- probabilites for each class (per example)
    y -- target probabilites (usually 1-hot for each example)
    """
    return T.mean(-T.sum(y*T.log(os), axis=1))

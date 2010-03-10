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

import theano.tensor as T

__all__ = ['mse', 'nll']

def nll(os, y):
    r"""
    Computes the negative log likelyhood.

    Inputs:
    os -- probabilites for each class
    y -- integer label for the good class
    """
    return -T.mean(T.log(os)[T.arange(y.shape[0]),y])

def mse(os, y):
    r"""
    Mean square error between `os` and `y`.
    """
    return T.mean((os-y)**2)

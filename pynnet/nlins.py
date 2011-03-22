from .base import *

__all__ = ['tanh', 'sigmoid', 'softmax', 'none']

def tanh(x):
    r"""
    Returns the elementwise tanh of `x`.

    Tests:
    >>> x = T.fmatrix('x')
    >>> exp = tanh(x)
    >>> theano.pp(exp)
    'tanh(x)'
    >>> f = theano.function([x], exp, allow_input_downcast=True)
    >>> inp = numpy.random.random((3, 5))
    >>> r = f(inp)
    >>> r.shape
    (3, 5)
    >>> abs(numpy.tanh(inp) - r).max() < 1e-5
    True
    """
    return T.tanh(x)

def sigmoid(x):
    r"""
    Returns the elementwise tanh of `x`.

    Tests:
    >>> x = T.fmatrix('x')
    >>> exp = sigmoid(x)
    >>> theano.pp(exp)
    'sigmoid(x)'
    >>> f = theano.function([x], exp, allow_input_downcast=True)
    >>> inp = numpy.random.random((3, 5))
    >>> r = f(inp)
    >>> r.shape
    (3, 5)
    >>> abs((numpy.tanh(inp/2)+1)/2 - r).max() < 1e-5
    True
    """
    return T.nnet.sigmoid(x)

def softmax(x):
    r"""
    Returns the elementwise tanh of `x`.

    Tests:
    >>> x = T.fmatrix('x')
    >>> exp = softmax(x)
    >>> theano.pp(exp)
    'Softmax(x)'
    >>> f = theano.function([x], exp, allow_input_downcast=True)
    >>> inp = numpy.random.random((3, 5))
    >>> r = f(inp)
    >>> r.shape
    (3, 5)
    >>> abs(numpy.e**inp/numpy.sum(numpy.e**inp,axis=1).reshape((3,1)) - r).max() < 1e-5
    True
    """
    return T.nnet.softmax(x)

def none(x):
    r"""
    Returns the elementwise tanh of `x`.

    Tests:
    >>> x = T.fmatrix('x')
    >>> exp = none(x)
    >>> theano.pp(exp)
    'x'
    >>> f = theano.function([x], exp, allow_input_downcast=True)
    >>> inp = numpy.random.random((3, 5))
    >>> r = f(inp)
    >>> r.shape
    (3, 5)
    >>> abs(inp - r).max() < 1e-5 # converted to float32 so not ==
    True
    """
    return x

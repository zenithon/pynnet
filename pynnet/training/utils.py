from pynnet.base import *
from itertools import izip

__all__ = ['get_updates']

def get_updates(params, err, alpha):
    r"""
    Returns a dictionary of updates suitable for theano.function.

    The updates are what would be done in one step of backprop over
    parameters `params` with error function `err` and step `alpha`.

    A typical call of this function over a network looks like this:
    
       updts = get_updates(params, cost, 0.01)
    
    Tests:
    >>> W = theano.shared(numpy.random.random((12, 8)), name='W')
    >>> b = theano.shared(numpy.random.random((8,)), name='b')
    >>> import theano.tensor as T
    >>> xs = T.matrix('x')
    >>> ys = T.matrix('y')
    >>> x = numpy.random.random((50, 12))
    >>> y = numpy.random.random((50, 8))
    >>> err = T.mean((T.tanh(T.dot(xs, W)+b) - ys)**2)
    >>> up = get_updates([W, b], err, 0.125)
    >>> f = theano.function([xs, ys], err, updates=up)
    >>> f(x, y) > f(x, y)
    True
    >>> up = get_updates([W, b], err, 0.03)
    >>> W.dtype == up[W].dtype
    True
    >>> v = numpy.array(0.1, dtype='float32')
    >>> up = get_updates([W, b], err, v)
    >>> W.dtype == up[W].dtype
    True
    """
    a = theano.tensor.cast(theano.tensor.as_tensor_variable(alpha),
                           dtype='float32')
    gparams = theano.tensor.grad(err, params)
    return dict((p, p - gp*a) for p, gp in izip(params, gparams))

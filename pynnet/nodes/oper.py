from .base import *

import operator

__all__ = ['AddNode', 'SubNode', 'MulNode', 'DivNode', 'SumNode', 'MeanNode']

AddNode = make_trivial(operator.add)
SubNode = make_trivial(operator.sub)
MulNode = make_trivial(operator.mul)
DivNode = make_trivial(operator.truediv)

class SumNode(BaseNode):
    r"""
    Sum over its input.

    If axis is provided, will sum over that axis or all if None is
    given.

    Examples:
    >>> x = T.fvector('x')
    >>> s = SumNode(x)
    >>> s = SumNode(x, 0)
    """
    def __init__(self, inp, axis=None, name=None):
        BaseNode.__init__(self, [inp], name)
        self.axis = axis

    def transform(self, inp):
        r"""
        >>> x = T.dmatrix('x')
        >>> s = SumNode(x)
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array(3.0)
        >>> s = SumNode(x, 1)
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array([ 2.,  1.])
        >>> s = SumNode(x, (0,1))
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array(3.0)
        """
        return inp.sum(self.axis)

class MeanNode(BaseNode):
    r"""
    Mean over its input.

    If axis is provided, will compute the mean over that axis or all
    if None is given.

    Examples:
    >>> x = T.fmatrix('x')
    >>> s = MeanNode(x)
    >>> s = MeanNode(x, 1)
    >>> s = MeanNode(x, (0,1))
    """
    def __init__(self, inp, axis=None, name=None):
        BaseNode.__init__(self, [inp], name)
        self.axis = axis

    def transform(self, inp):
        r"""
        >>> x = T.dmatrix('x')
        >>> s = MeanNode(x)
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array(0.75)
        >>> s = MeanNode(x, 1)
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array([ 1. ,  0.5])
        >>> s = MeanNode(x, (0,1))
        >>> f = theano.function([x], s.output)
        >>> f([[0.5, 1.5], [0.25, 0.75]])
        array(0.75)
        """
        return inp.mean(self.axis)

from .base import *
from pynnet.nlins import *

__all__ = ['SimpleNode', 'SharedNode']

class SharedNode(BaseNode):
    r"""
    Specialized node that works with passed-in W and b.

    Examples:
    >>> W = T.fmatrix()
    >>> b = T.fvector()
    >>> x = T.fmatrix('x')
    >>> h = SharedNode(x, W, b)

    >>> W = theano.shared(numpy.random.random((3, 2)))
    >>> b = theano.shared(numpy.random.random((2,)))
    >>> h = SharedNode(x, W, b, nlin=tanh)

    Attributes: 

    `W` -- (theano matrix, read-write) can be any theano expression
           that gives a matrix of the appropriate size.  The given
           expression is not automatically treated as a gradient
           parameter and is not saved with the node.  It is your
           responsability to ensure that this happens if you need it.
    `b` -- (theano vector, read-write) can be any theano expression
           that gives a vector of the appropriate size.  The same
           precautions as for `W` apply.
    `nlin` -- (function, read-write) must be a function that will
              receive as input a theano expression gives back a theano
              expression of the same shape.  Apart from the shape
              restriction any computation can be preformed on the
              input.  This is saved with the node.
    """
    def __init__(self, input, W, b, nlin=tanh, name=None):
        r"""
        Tests:
        >>> W = T.fmatrix()
        >>> b = T.fvector()
        >>> x = T.fmatrix('x')
        >>> h = SharedNode(x, W, b)
        >>> h.params
        []
        >>> ht = SharedNode(x, W, b)
        >>> ht.name != h.name # this tests BaseNode auto-naming (a bit)
        True
        >>> ht.nlin
        <function tanh at ...>
        >>> h2 = test_saveload(ht)
        >>> h2.nlin
        <function tanh at ...>
        >>> h2.params
        []
        """
        if not isinstance(input, (list, tuple)):
            input = [input]
            W = [W]
        BaseNode.__init__(self, input, name)
        self.nlin = nlin
        self.W = W
        self.b = b

    def transform(self, *inp):
        r"""
        Returns the dot product of the input expression with the given
        expression.
        
        Tests:
        >>> W = T.fmatrix('W')
        >>> b = T.fvector('b')
        >>> x = T.fmatrix('x')
        >>> h = SharedNode(x, W, b)
        >>> theano.pp(h.output)
        'tanh((b + (x \\dot W)))'
        """
        return self.nlin(sum((T.dot(x, w) for x, w in zip(inp, self.W)), self.b))

class SimpleNode(SharedNode):
    r"""
    Typical hidden node of a MLP: units are fully-connected and have
    an activation function. Weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).

    An extension was implemented to provide for multiple inputs.  If
    you specify `input` as a list and `n_in` as a list of
    corresponding length, a W matrix will be created for each input
    and the results will be added together before passing through the
    non-linearity function.  This is slighty faster than joining the
    nodes befaore passing through a single W matrix.
    
    Examples:
    >>> x = T.fmatrix('x')
    >>> h = SimpleNode(x, 20, 16)
    >>> h = SimpleNode(x, 50, 40, nlin=sigmoid)
    >>> h = SimpleNode(x, 3, 2, dtype='float32')
    >>> x2 = T.fmatrix('x2')
    >>> h = SimpleNode([x, x2], [20, 10], 20)
    """
    def __init__(self, input, n_in, n_out, nlin=tanh, rng=numpy.random,
                 dtype=theano.config.floatX, name=None):
        r"""
        Tests: 
        >>> x = T.fmatrix('x')
        >>> h = SimpleNode(x, 2, 1)
        >>> h.params
        [W0, b]
        >>> h.W[0].get_value().shape
        (2, 1)
        >>> h2 = test_saveload(h)
        >>> h2.W[0].get_value().shape
        (2, 1)
        >>> h2.params
        [W0, b]
        >>> x2 = T.fmatrix('x2')
        >>> h = SimpleNode([x, x2], [20, 10], 20)
        >>> h.params
        [W0, W1, b]
        """
        if not isinstance(input, (list, tuple)):
            input = [input]
            n_in = [n_in]
        else:
            assert len(input) == len(n_in)

        W = [self.make_W(i, n, n_out, rng, dtype) for i, n in enumerate(n_in)]
        b_values = numpy.zeros((n_out,), dtype=dtype)
        b = theano.shared(value=b_values, name='b')
        SharedNode.__init__(self, input, W, b, nlin=nlin, name=name)
        self.local_params = self.W + [self.b]

    @classmethod
    def make_W(cls, i, n_in, n_out, rng, dtype):
        r"""
        Make a W matrix suitable for the given parameters.

        This uses sqrt(6./(n_in+n_out)) as the range for the values.
        """
        w_range = numpy.sqrt(6./(n_in+n_out))
        W_values = rng.uniform(low=-w_range, high=w_range,
                               size=(n_in, n_out)).astype(dtype)
        return theano.shared(value=W_values, name='W'+str(i))

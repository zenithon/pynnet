from base import *
from pynnet.nlins import *

__all__ = ['SimpleLayer', 'SharedLayer']

import theano.tensor as T

class SharedLayer(BaseLayer):
    r"""
    Specialized layer that works with passed-in W and b.

    Examples:
    >>> W = T.fmatrix()
    >>> b = T.fvector()
    >>> h = SharedLayer(W, b)

    >>> W = theano.shared(numpy.random.random((3, 2)))
    >>> b = theano.shared(numpy.random.random((2,)))
    >>> h = SharedLayer(W, b, activation=tanh)

    Attributes: 

    `W` -- (theano matrix, read-write) can be any theano expression
           that gives a matrix of the appropriate size.  The given
           expression is not automatically treated as a gradient
           parameter and is not saved with the layer.  It is your
           responsability to ensure that this happens if you need it.
    `b` -- (theano vector, read-write) can be any theano expression
           that gives a vector of the appropriate size.  The same
           precautions as for `W` apply.
    `activation` -- (function, read-write) must be a function that
                    will receive as input a theano expression gives
                    back a theano expression of the same shape.  Apart
                    from the shape restriction any computation can be
                    preformed on the input.  This is saved with the
                    layer.
    """
    def __init__(self, W, b, activation=tanh, rng=numpy.random, name=None):
        r"""
        Tests:
        >>> W = T.fmatrix()
        >>> b = T.fvector()
        >>> h = SharedLayer(W, b)
        >>> ht = SharedLayer(W, b)
        >>> ht.name != h.name # this tests BaseLayer auto-naming (a bit)
        True
        >>> ht.activation
        <function tanh at ...>
        >>> h2 = test_saveload(ht)
        >>> h2.activation
        <function tanh at ...>
        """
        BaseLayer.__init__(self, name)
        self.activation = activation
        self.W = W
        self.b = b

    def _save_(self, file):
        file.write('SL1')
        psave(self.activation, file)

    def _load_(self, file):
        s = file.read(3)
        if s != 'SL1':
            raise ValueError('wrong cookie for SharedLayer')
        self.activation = pload(file)
    
    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> W = theano.shared(value=numpy.random.random((3,2)).astype(numpy.float32), name='W')
        >>> b = theano.shared(value=numpy.random.random((2,)).astype(numpy.float32), name='b')
        >>> h = SharedLayer(W, b)
        >>> x = T.fmatrix('x')
	>>> h.build(x, input_shape=(4, 3))
        >>> h.params
        []
        >>> h.input
        x
	>>> h.output_shape
	(4, 2)
        >>> theano.pp(h.output)
        'tanh(((x \\dot W) + b))'
        >>> f = theano.function([x], h.output)
        >>> r = f(numpy.random.random((4, 3)))
        >>> r.shape
        (4, 2)
        >>> r.dtype
        dtype('float32')
        >>> h.build(x)
	>>> h.output_shape
        """
	if input_shape:
            if len(input_shape) != 2:
                raise ValueError('Expecting a 2-dimension input_shape, got %s'%(input_shape,))
            if input_shape[1] != self.W.value.shape[0]:
                raise ValueError('Wrong dimensions for matrix multiplication, (%d != %d)'%(input_shape[1], self.W.value.shape[0]))
            self.output_shape = (input_shape[0], self.W.value.shape[1])
        else:
            self.output_shape = None
        self.input = input
        self.output = self.activation(T.dot(self.input, self.W) + self.b)
        self.params = []

class SimpleLayer(SharedLayer):
    r"""
    Typical hidden layer of a MLP: units are fully-connected and have
    an activation function. Weight matrix W is of shape (n_in,n_out)
    and the bias vector b is of shape (n_out,).
    
    Examples:
    >>> h = SimpleLayer(20, 16)
    >>> h = SimpleLayer(50, 40, activation=sigmoid)
    >>> h = SimpleLayer(3, 2, dtype=numpy.float32)

    Attributes:
    `W` -- (shared matrix, read-only) Shared connection weights
           matrix.
    `b` -- (shared vector, read-only) Shared bias vector.
    `activation` -- (function, read-write) must be a function that
                    will receive as input a theano expression gives
                    back a theano expression of the same shape.  Apart
                    from the shape restriction any computation can be
                    preformed on the input.
    """
    def __init__(self, n_in, n_out, activation=tanh, rng=numpy.random,
                 dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> h = SimpleLayer(2, 1)
        >>> h.W.value.shape
        (2, 1)
        >>> h2 = test_saveload(h)
        >>> h2.W.value.shape
        (2, 1)
        """
        w_range = numpy.sqrt(6./(n_in+n_out))
        W_values = rng.uniform(low=-w_range, high=w_range,
                               size=(n_in, n_out)).astype(dtype)
        W = theano.shared(value=W_values, name='W')
        b_values = numpy.zeros((n_out,), dtype=dtype)
        b = theano.shared(value=b_values, name='b')
        SharedLayer.__init__(self, W, b, activation=activation,
                             rng=rng, name=name)
    
    def _save_(self, file):
        file.write('HL2')
        numpy.save(file, self.W.value)
        numpy.save(file, self.b.value)

    def _load_(self, file):
        s = file.read(3)
        if s != 'HL2':
            raise ValueError('wrong cookie for SimpleLayer')
        self.W = theano.shared(value=numpy.load(file), name='W')
        self.b = theano.shared(value=numpy.load(file), name='b')
    
    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> h = SimpleLayer(3, 2, dtype=numpy.float32)
        >>> x = T.fmatrix('x')
        >>> h.build(x, input_shape=(4, 3))
        >>> h.params
        [W, b]
        >>> h.input
        x
        >>> h.output_shape
        (4, 2)
        >>> theano.pp(h.output)
        'tanh(((x \\dot W) + b))'
        >>> f = theano.function([x], h.output)
        >>> r = f(numpy.random.random((4, 3)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (4, 2)
        >>> h.build(x)
        >>> h.output_shape
        """
        SharedLayer.build(self, input, input_shape)
        self.params += [self.W, self.b]

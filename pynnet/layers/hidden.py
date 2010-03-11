from pynnet.base import *
from pynnet.nlins import *

__all__ = ['SimpleLayer', 'SharedLayer']

import theano.tensor as T

class SharedLayer(BaseObject):
    def __init__(self, W, b, activation=tanh, rng=numpy.random):
        r"""
        Specialized layer that works with a shared W matrix.

        Examples:
        >>> W = T.fmatrix()
        >>> b = T.fvector()
        >>> h = SharedLayer(W, b)

        Tests:
        >>> h = SharedLayer(W, b)
        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> h.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> h2 = SharedLayer.loadf(f2)
        >>> f2.close()
        """
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
        self.activation= pload(file)
    
    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> W = theano.shared(value=numpy.random.random((3,2)).astype(numpy.float32), name='W')
        >>> b = theano.shared(value=numpy.random.random((2,)).astype(numpy.float32), name='b')
        >>> h = SharedLayer(W, b)
        >>> x = T.fmatrix('x')
        >>> h.build(x)
        >>> h.params
        []
        >>> h.input
        x
        >>> theano.pp(h.output)
        'tanh(((x \\dot W) + b))'
        >>> f = theano.function([x], h.output)
        >>> r = f(numpy.random.random((4, 3)))
        >>> r.shape
        (4, 2)
        >>> r.dtype
        dtype('float32')
        """
        self.input = input
        self.output = self.activation(T.dot(self.input, self.W) + self.b)
        self.params = []

class SimpleLayer(SharedLayer):
    def __init__(self, n_in, n_out, activation=tanh, rng=numpy.random,
                 dtype=theano.config.floatX):
        r"""
        Typical hidden layer of a MLP: units are fully-connected and have
        an activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        Parameters:
        `n_in` -- (int) dimensionality of input
        `n_out` -- (int) dimensionality of output
        `activation` -- (function) function to apply to the output
        `rng` -- (numpy.random.RandomState) random generator to use
                 for initialization

        Examples:
        >>> h = SimpleLayer(20, 16)

        Tests:
        >>> h = SimpleLayer(2, 1)
        >>> h.W.value.shape
        (2, 1)
        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> h.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> h2 = SimpleLayer.loadf(f2)
        >>> f2.close()
        >>> h2.W.value.shape
        (2, 1)
        """
        w_range = numpy.sqrt(6./(n_in+n_out))
        W_values = rng.uniform(low=-w_range, high=w_range,
                               size=(n_in, n_out)).astype(dtype)
        W = theano.shared(value=W_values, name='W')
        b_values = numpy.zeros((n_out,), dtype=dtype)
        b = theano.shared(value=b_values, name='b')
        SharedLayer.__init__(self, W, b, activation=activation, rng=rng)
    
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
    
    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> h = SimpleLayer(3, 2, dtype=numpy.float32)
        >>> x = T.fmatrix('x')
        >>> h.build(x)
        >>> h.params
        [W, b]
        >>> h.input
        x
        >>> theano.pp(h.output)
        'tanh(((x \\dot W) + b))'
        >>> f = theano.function([x], h.output)
        >>> r = f(numpy.random.random((4, 3)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (4, 2)
        """
        SharedLayer.build(self, input)
        self.params += [self.W, self.b]

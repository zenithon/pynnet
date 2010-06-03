from base import *
from simple import SimpleLayer
from pynnet.nlins import *

__all__ = ['RecurrentLayer']

class RecurrentLayer(SimpleLayer):
    r"""
    This is a recurrent layer with a one tap delay.  This means it
    gets it own output from the previous step in addition to the
    input provided at each step.
    
    The memory is automatically updated and starts with a zero fill.
    
    Note that the input must be one example at a time (i.e. of
    dimension 1xN).  You will get errors otherwise.
    
    Examples:
    >>> r = RecurrentLayer(20, 10)
    >>> r = RecurrentLayer(10, 10, activation=sigmoid)
    >>> r = RecurrentLayer(3, 2, dtype='float32')
    
    Attributes:
    * All the attributes of `SimpleLayer` and
    `memory` -- (shared matrix, read-write) Shared memory matrix.
                Overwrite with zeros to clear memory.
    """
    def __init__(self, n_in, n_out, activation=tanh, rng=numpy.random,
                 dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> r = RecurrentLayer(10, 10, dtype='float32')
        >>> r.memory.value
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)
        >>> r2 = test_saveload(r)
        >>> r2.memory.value
        array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]], dtype=float32)
        """
        SimpleLayer.__init__(self, n_in+n_out, n_out, activation, rng=rng, 
                             dtype=dtype, name=name)
        self.memory = theano.shared(numpy.zeros((1,n_out), dtype=dtype),
                                    name='memory')

    def _save_(self, file):
        numpy.save(file, self.memory.value)
    
    def _load1_(self, file):
        self.memory = theano.shared(numpy.load(file), name='memory')
    _load_ = _load1_

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> r = RecurrentLayer(3, 2, dtype='float32')
        >>> x = theano.tensor.fmatrix('x')
        >>> r.build(x, input_shape=(1, 3))
        >>> r.params
        [W, b]
        >>> r.input
        x
        >>> r.output_shape
        (1, 2)
        >>> theano.pp(r.output)
        'tanh(((join(1, x, memory) \\dot W) + b))'
        >>> f = theano.function([x], r.output)
        >>> v = f(numpy.random.random((1, 3)))
        >>> v.dtype
        dtype('float32')
        >>> v.shape
        (1, 2)
        >>> (r.memory.value == v).all()
        True
        >>> r.build(x)
        >>> r.output_shape
        """
        assert input_shape is None or len(input_shape) == 2
        if input_shape is not None:
            assert input_shape[0] == 1
            input_shape = (1, input_shape[1]+self.memory.value.shape[1])
        SimpleLayer.build(self, theano.tensor.join(1, input, self.memory),
                          input_shape)
        self.input = input
        self.memory.default_update = self.output
        

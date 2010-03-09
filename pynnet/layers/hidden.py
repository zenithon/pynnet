from pynnet.base import *
from pynnet.nlins import tanh

__all__ = ['HiddenLayer']

import theano.tensor as T

class HiddenLayer(BaseObject):
    def __init__(self, n_in, n_out, activation=tanh, rng=numpy.random,
                 dtype=theano.config.floatX):
        r"""
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        
        Parameters:
        `n_in` -- (int) dimensionality of input
        `n_out` -- (int) dimensionality of output
        `activation` -- (function) function to apply to the output
        `rng` -- (numpy.random.RandomState) random generator to use
                 for initialization

        Examples:
        >>> h = HiddenLayer(20, 16)

        Tests:
        >>> h = HiddenLayer(2, 1)
        >>> h.W.value.shape
        (2, 1)
        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> h.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> h2 = HiddenLayer.loadf(f2)
        >>> f2.close()
        >>> h2.W.value.shape
        (2, 1)
        """
        self.activation = activation
        w_range = numpy.sqrt(6./(n_in+n_out))
        W_values = rng.uniform(low=-w_range, high=w_range,
                               size=(n_in, n_out)).astype(dtype)
        self.W = theano.shared(value=W_values, name='W')

        b_values = numpy.zeros((n_out,), dtype=dtype)
        self.b = theano.shared(value=b_values, name='b')
    
    def _save_(self, file):
        file.write('HL1')
        psave(self.activation, file)
        numpy.save(file, self.W.value)
        numpy.save(file, self.b.value)

    def _load_(self, file):
        s = file.read(3)
        if s != 'HL1':
            raise ValueError('Wrong cookie, probably not a HiddenLayer save')
        self.activation = pload(file)
        self.W = theano.shared(value=numpy.load(file), name='W')
        self.b = theano.shared(value=numpy.load(file), name='b')
    
    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.

        Tests:
        >>> h = HiddenLayer(3, 2)
        >>> x = T.fmatrix('x')
        >>> h.build(x)
        >>> h.params
        [W, b]
        >>> h.input
        x
        >>> theano.pp(h.output)
        'tanh(((x \\dot W) + b))'
        """
        self.input = input
        self.output = self.activation(T.dot(self.input, self.W) + self.b)
        self.params = [self.W, self.b]

from base import *
from pynnet.nodes import SimpleNode, SharedNode, ConvNode, SharedConvNode
from pynnet.nlins import tanh
from pynnet.errors import mse

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['CorruptNode', 'autoencoder', 'conv_autoencoder']

class CorruptNode(BaseNode):
    r"""
    Node that corrupts its input.

    Examples:
    >>> x = T.fmatrix('x')
    >>> c = CorruptNode(x, 0.25)
    >>> c = CorruptNode(x, 0.0)

    Attributes: 
    `noise` -- (float, read-write) The noise level as a probability of
               destroying any given input.  Must be kept between 0 and
               1.
    """
    def __init__(self, input, noise, theano_rng=RandomStreams(), name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> c = CorruptNode(x, 0.25)
        >>> c.noise
        0.25
        >>> c2 = test_saveload(c)
        >>> c2.noise
        0.25
        """
        BaseNode.__init__(self, [input], name)
        self.noise = noise
        self.theano_rng = theano_rng

    def transform(self, input):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> c = CorruptNode(x, 0.25)
        >>> theano.pp(c.output)
        '(x * RandomFunction{binomial}(<RandomStateType>, x.shape, 1, 0.75))'
        >>> f = theano.function([x], c.output)
        >>> r = f(numpy.random.random((10, 12)))
        >>> r.dtype
        dtype('float32')
        >>> c.noise = 0.0
        >>> theano.pp(c.output)
        'x'
        """
        if self.noise == 0.0:
            return input
        else:
            idx = self.theano_rng.binomial(size=input.shape, n=1, 
                                           p=1-self.noise, dtype='int8')
            return input * idx

def autoencoder(input, n_in, n_out, tied=False, nlin=tanh,
                dtype=theano.config.floatX, rng=numpy.random):
    r"""
    Utility function to build an (single-node) autoencoder.
    
    This function returns a tuple with the encoding and
    reconstruction parts of the output.

    Examples:
    >>> x = T.fmatrix('x')
    >>> enc, dec = autoencoder(x, 3, 2)
    >>> enc, dec = autoencoder(x, 6, 4, tied=True)
    
    Tests:
    >>> enc, dec = autoencoder(x, 20, 16, tied=True)
    >>> enc.params
    [W, b]
    >>> dec.params
    [b2, W, b]
    >>> enc.nlin
    <function tanh at ...>
    >>> theano.pp(dec.W)
    'W.T'
    """
    encode = SimpleNode(input, n_in, n_out, nlin=nlin,
                         dtype=dtype, rng=rng)
    if tied:
        b = theano.shared(value=numpy.random.random((n_in,)).astype(dtype),
                          name='b2')
        decode = SharedNode(encode, encode.W.T, b, nlin=nlin)
        decode.local_params.append(b)
    else:
        decode = SimpleNode(encode, n_out, n_in,
                             nlin=nlin, dtype=dtype, rng=rng)
    return encode, decode
    
def conv_autoencoder(input, filter_size, num_filt, num_in=1, rng=numpy.random,
                     nlin=tanh, dtype=theano.config.floatX, name=None):
    r"""
    Utility function to build an (single-node) autoencoder.
    
    This function returns a tuple with the encoding and
    reconstruction parts of the output.

    Examples:
    >>> x = T.tensor4('x', dtype='float32')
    >>> enc, dec = conv_autoencoder(x, (5,5), 3)
    >>> enc, dec = conv_autoencoder(x, (4,3), 8)
    >>> enc, dec = conv_autoencoder(x, (4,4), 4, dtype='float32')

    Tests:
    >>> enc, dec = conv_autoencoder(x, (5,5), 3, dtype='float32')
    >>> enc.filter_shape
    (3, 1, 5, 5)
    >>> dec.filter_shape
    (1, 3, 5, 5)
    >>> dec.nlin
    <function tanh at ...>
    """
    encode = ConvNode(input, filter_size=filter_size, num_filt=num_filt,
                       num_in=num_in, nlin=nlin, rng=rng,
                       mode='valid', dtype=dtype)
    dec_in = SharedConvNode(input, encode.filter, encode.b,
                             encode.filter_shape, nlin=nlin,
                             mode='full')
    decode = ConvNode(dec_in, filter_size=filter_size, num_filt=num_in,
                       dtype=dtype, num_in=num_filt, nlin=nlin, 
                       rng=rng, mode='valid')
    return encode, decode
    

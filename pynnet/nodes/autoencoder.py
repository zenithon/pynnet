from .base import *
from .simple import SimpleNode, SharedNode
from .conv import ConvNode, SharedConvNode
from .recurrent import RecurrentInput, RecurrentOutput

from pynnet.nlins import tanh
from pynnet.errors import mse

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['CorruptNode', 'autoencoder', 'recurrent_autoencoder', 
           'conv_autoencoder']

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
        '(x * RandomFunction{binomial}(<RandomStateType>, int32(x.shape), 1, 0.75))'
        >>> f = theano.function([x], c.output, allow_input_downcast=True)
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

def autoencoder(input, n_in, n_out, noise=0.0, tied=False, nlin=tanh,
                dtype=theano.config.floatX, rng=numpy.random):
    r"""
    Utility function to build an autoencoder.
    
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
    [W, b, b2]
    >>> enc.nlin
    <function tanh at ...>
    >>> theano.pp(dec.W)
    'W.T'
    >>> theano.pp(enc.output)
    'tanh(((x \\dot W) + b))'
    >>> theano.pp(dec.output)
    'tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2))'
    """
    noiser = CorruptNode(input, noise)
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
    return encode, decode.replace({input: noiser})

def recurrent_autoencoder(input, n_in, n_out, noise=0.0, tied=False, nlin=tanh,
                          dtype=theano.config.floatX, rng=numpy.random):
    r"""
    Utility function to build a recurrent autoencoder.
    
    This function returns a tuple with the encoding output and
    pretraining cost.

    Examples:
    >>> x = T.fmatrix('x')
    >>> enc, dec, rec_in = recurrent_autoencoder(x, 3, 2)
    >>> enc, dec, rec_in = recurrent_autoencoder(x, 6, 4, tied=True)
    
    Tests:
    >>> enc, dec, rec_in = recurrent_autoencoder(x, 20, 16, tied=True)
    >>> enc.params
    [W, b]
    >>> dec.params
    [W, b, b2]
    >>> theano.pp(dec.W)
    'W.T'
    >>> theano.pp(enc.output)
    'scan(?_steps, x, memory, W, b)'
    >>> theano.pp(dec.output)
    'tanh(((scan(?_steps, x, memory, W, b) \\dot W.T) + b2))'
    """
    tag = object()
    rec_in = RecurrentInput(input, tag)
    noiser = CorruptNode(rec_in, noise)
    encode = SimpleNode(rec_in, n_in, n_out, nlin=nlin,
                         dtype=dtype, rng=rng)
    rec_enc = RecurrentOutput(encode, tag, outshp=(n_out,), dtype=dtype)
    if tied:
        b = theano.shared(value=numpy.random.random((n_in,)).astype(dtype),
                          name='b2')
        decode = SharedNode(rec_enc, encode.W.T, b, nlin=nlin)
        decode.local_params.append(b)
    else:
        decode = SimpleNode(rec_enc, n_out, n_in,
                             nlin=nlin, dtype=dtype, rng=rng)
    return rec_enc, decode.replace({rec_in: noiser}), rec_in

def conv_autoencoder(input, filter_size, num_filt, num_in=1, noise=0.0, 
                     nlin=tanh, rng=numpy.random, dtype=theano.config.floatX):
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
    noiser = CorruptNode(input, noise)
    encode = ConvNode(input, filter_size=filter_size, num_filt=num_filt,
                      num_in=num_in, nlin=nlin, rng=rng,
                      mode='valid', dtype=dtype)
    dec_in = SharedConvNode(noiser, encode.filter, encode.b,
                            encode.filter_shape, nlin=nlin,
                            mode='full')
    dec_in.local_params.append(encode.filter)
    dec_in.local_params.append(encode.b)
    decode = ConvNode(dec_in, filter_size=filter_size, num_filt=num_in,
                      dtype=dtype, num_in=num_filt, nlin=nlin, 
                      rng=rng, mode='valid')
    return encode, decode

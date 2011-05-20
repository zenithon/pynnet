from .base import *
from .simple import SimpleNode, SharedNode
from .conv import ConvNode, SharedConvNode
from .recurrent import RecurrentMemory, RecurrentNode, RecurrentInput, RecurrentOutput
from .oper import MeanNode

from pynnet.nlins import tanh
from .errors import mse, binary_cross_entropy

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['CorruptNode', 'autoencoder', 'RecurrentAutoencoder', 
           'recurrent_autoencoder', 'conv_autoencoder']

class CorruptNode(BaseNode):
    r"""
    Node that corrupts its input.

    Examples:
    >>> x = T.matrix('x')
    >>> c = CorruptNode(x, 0.25)
    >>> c = CorruptNode(x, 0.0)
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
    [W0, b]
    >>> dec.params
    [W0, b, b2]
    >>> dec.params[2].get_value()[0]
    0.0
    >>> enc.nlin
    <function tanh at ...>
    >>> theano.pp(dec.W[0])
    'W0.T'
    >>> theano.pp(enc.output)
    'tanh((b + (x \\dot W0)))'
    >>> theano.pp(dec.output)
    'tanh((b2 + (tanh((b + (x \\dot W0))) \\dot W0.T)))'
    """
    noiser = CorruptNode(input, noise)
    encode = SimpleNode(input, n_in, n_out, nlin=nlin,
                         dtype=dtype, rng=rng)
    if tied:
        b2 = theano.shared(value=numpy.zeros((n_in,), dtype=dtype),
                          name='b2')
        decode = SharedNode(encode, encode.W[0].T, b2, nlin=nlin)
        decode.local_params.append(b2)
    else:
        decode = SimpleNode(encode, n_out, n_in,
                             nlin=nlin, dtype=dtype, rng=rng)
    return encode, decode.replace({input: noiser})

class RecurrentAutoencoder(BaseObject):
    r"""
    This class is a factory for building recurrent autoencoders.  The
    returned object is not a node but only a container for the various
    useful parts that compose the autoencoder.

    After building the object you can access the attributes
      - encode
      - decode_in
      - decode_state
      - cost_in
      - cost_state
    which are all nodes to use in your model.  

    If you need to do something not covered by the prebuilt nodes, you
    can also use the following attributes to use in your graph:
     - input
     - noisy_input
     - mem
    
    (Note that using mem requires the use of a RecurrentNode).
    
    Examples:
    >>> x = T.matrix('x')
    >>> rae = RecurrentAutoencoder(x, 3, 2)
    >>> rae = RecurrentAutoencoder(x, 6, 4, tied=True)
    >>> rae = RecurrentAutoencoder(x, 7, 5, noise=0.2)
    >>> rae = RecurrentAutoencoder(x, 4, 1, recost=binary_cross_entropy)
    """
    def __init__(self, inp, n_in, n_out, noise=0.0, tied=False, nlin=tanh,
                 recost=mse, dtype=theano.config.floatX, rng=numpy.random):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> rae = RecurrentAutoencoder(x, 20, 16, tied=True, dtype='float32')
        >>> rae.encode.params
        [W0, W1, b]
        >>> rae.decode_in.params
        [W0, W1, b, b0]
        >>> rae.decode_state.params
        [W0, W1, b, b1]
        >>> f = theano.function([x], rae.encode.output)
        >>> xval = numpy.random.random((3, 20)).astype('float32')
        >>> y = f(xval)
        >>> f = theano.function([x], rae.decode_in.output)
        >>> y = f(xval)
        >>> f = theano.function([x], rae.decode_state.output)
        >>> y = f(xval)
        >>> f = theano.function([x], rae.cost_in.output)
        >>> y = f(xval)
        >>> f = theano.function([x], rae.cost_state.output)
        >>> y = f(xval)
        """
        self.noisy_input = CorruptNode(inp, noise)
        self.input = self.noisy_input.inputs[0]
        self.mem = RecurrentMemory(numpy.zeros((n_out,), dtype=dtype))
        encg = SimpleNode([self.input, self.mem], [n_in, n_out], n_out, 
                          nlin=nlin, dtype=dtype, rng=rng)
        self.mem.subgraph = encg
        self.encode = RecurrentNode([self.input], [], self.mem, encg)
        self.noisy_mem = RecurrentMemory(numpy.zeros((n_out,), dtype=dtype))
        newencg = encg.replace({self.input: self.noisy_input, self.mem: self.noisy_mem})
        self.noisy_mem.subgraph = newencg
        if tied:
            b0 = theano.shared(value=numpy.zeros((n_in,), dtype=dtype),
                               name='b0')
            decing = SharedNode(newencg, encg.W[0].T, b0, nlin=nlin)
            decing.local_params.append(b0)
            b1 = theano.shared(value=numpy.zeros((n_out,), dtype=dtype),
                               name='b1')
            decstateg = SharedNode(newencg, encg.W[1].T, b1, nlin=nlin)
            decstateg.local_params.append(b1)
        else:
            decing = SimpleNode(newencg, n_out, n_in, nlin=nlin, dtype=dtype,
                                rng=rng)
            decstateg = SimpleNode(newencg, n_out, n_out, nlin=nlin,
                                   dtype=dtype, rng=rng)
        self.decode_in = RecurrentNode([self.input], [], self.noisy_mem, 
                                       decing)
        self.decode_state = RecurrentNode([self.input], [], self.noisy_mem,
                                          decstateg)
        self.cost_in_rec = RecurrentNode([self.input], [], self.noisy_mem,
                                         recost(inp, decing), nopad=True)
        self.cost_in = MeanNode(self.cost_in_rec)
        self.cost_state_rec = RecurrentNode([self.input], [], self.noisy_mem,
                                            recost(self.noisy_mem, decstateg),
                                            nopad=True)
        self.cost_state = MeanNode(self.cost_state_rec)
    
    def clear(self):
        r"""
        Clears the memory from all associated nodes.
        """
        self.mem.clear()
        self.noisy_mem.clear()

def recurrent_autoencoder(inp, n_in, n_out, noise=0.0, tied=False, nlin=tanh,
                          dtype=theano.config.floatX, rng=numpy.random):
    r"""
    Utility function to build a recurrent autoencoder.
    
    This function returns a tuple with the encoding output and
    pretraining cost.

    Examples:
    >>> x = T.fmatrix('x')
    >>> enc, dec = recurrent_autoencoder(x, 3, 2)
    >>> enc, dec = recurrent_autoencoder(x, 6, 4, tied=True)
    
    Tests:
    >>> enc, dec = recurrent_autoencoder(x, 20, 16, tied=True)
    >>> enc.params
    [W0, b]
    >>> dec.params
    [W0, b, b2]
    >>> dec.params[2].get_value()[0]
    0.0
    >>> theano.pp(dec.W[0])
    'W0.T'
    >>> f = theano.function([x], enc.output)
    >>> xval = numpy.random.random((3, 20)).astype('float32')
    >>> y = f(xval)
    >>> f2 = theano.function([x], dec.output)
    >>> y = f2(xval)
    """
    tag = object()
    rec_in = RecurrentInput(inp, tag)
    encode = SimpleNode(rec_in, n_in+n_out, n_out, nlin=nlin,
                         dtype=dtype, rng=rng)
    rec_enc = RecurrentOutput(encode, tag, outshp=(n_out,), dtype=dtype)
    if tied:
        b2 = theano.shared(value=numpy.zeros((n_in+n_out,), dtype=dtype),
                          name='b2')
        decode = SharedNode(rec_enc, encode.W[0].T, b2, nlin=nlin)
        decode.local_params.append(b2)
    else:
        decode = SimpleNode(rec_enc, n_out, n_in+n_out,
                             nlin=nlin, dtype=dtype, rng=rng)
    
    noiser = CorruptNode(rec_in, noise)
    return rec_enc, decode.replace({rec_in: noiser})

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

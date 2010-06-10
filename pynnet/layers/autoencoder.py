from base import *
from pynnet.layers import SimpleLayer, SharedLayer, ConvLayer, SharedConvLayer
from pynnet.net import NNet
from pynnet.nlins import tanh
from pynnet.errors import mse

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['CorruptLayer', 'Autoencoder', 'ConvAutoencoder']

class CorruptLayer(BaseLayer):
    r"""
    Layer that corrupts its input.

    Examples:
    >>> c = CorruptLayer(0.25)
    >>> c = CorruptLayer(0.0)

    Attributes: 
    `noise` -- (float, read-write) The noise level as a probability of
               destroying any given input.  Must be kept between 0 and
               1.
    """
    def __init__(self, noise, theano_rng=RandomStreams(), name=None):
        r"""
        Tests:
        >>> c = CorruptLayer(0.25)
        >>> c.noise
        0.25
        >>> c2 = test_saveload(c)
        >>> c2.noise
        0.25
        """
        BaseLayer.__init__(self, name)
        self.noise = noise
        self.theano_rng = theano_rng

    def _save_(self, file):
        psave((self.noise, self.theano_rng), file)

    def _load1_(self, file):
        self.noise, self.theano_rng = pload(file)

    _load_ = _load1_
    
    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> c = CorruptLayer(0.25)
        >>> x = T.fmatrix('x')
        >>> c.build(x, (10, 12))
        >>> c.params
        []
        >>> c.input
        x
        >>> c.output_shape
        (10, 12)
        >>> theano.pp(c.output)
        '(x * RandomFunction{binomial}(<RandomStateType>, [10 12], 1, 0.75))'
        >>> f = theano.function([x], c.output)
        >>> r = f(numpy.random.random((10, 12)))
        >>> r.shape
        (10, 12)
        >>> r.dtype
        dtype('float32')
        >>> c.build(x)
        >>> c.output_shape
        >>> c.noise = 0.0
        >>> c.build(x)
        >>> theano.pp(c.output)
        'x'
        """
        self.input = input
        if self.noise == 0.0:
            self.output = self.input
        else:
            if input_shape:
                idx = self.theano_rng.binomial(size=input_shape, n=1, 
                                               p=1-self.noise, dtype='int8')
            else:
                idx = self.theano_rng.binomial(size=input.shape, n=1, 
                                               p=1-self.noise, dtype='int8')
            self.output = self.input * idx
        self.output_shape = input_shape
        self.params = []

class Autoencoder(NNet):
    r"""
    Autoencoder layer.
    
    This layer also acts as a network if you want to do pretraining.
    
    Examples:
    >>> a = Autoencoder(32, 20)
    >>> a = Autoencoder(20, 16, tied=True, noise=0.2)

    Attributes:
    `noise` -- (float, read-write) The noise level as a probability of
               destroying any given input.  This is applied for
               pretraining only. Must be kept between 0 and 1.
    `tied` -- (bool, read-only) wether the weights of the encoding and
              the decoding layer are tied.
    `W` -- (shared matrix, read-only) The matrix for the encoding
           layer.
    `W2` -- (theano expression, read-only) The matrix for the decoding
            layer.  May either be the transpose of `W` or another
            shared matrix depending on the `tied` value passed to the
            constructor.
    `b` -- (shared vector, read-only) The bias vector for the encoding
           layer.
    `b2` -- (shared vector, read-only) The bias vector for the
            decoding layer.
    `activation` -- (function, read-write) The activation of the
                    encoding layer. (See `SimpleLayer` docs for
                    details)
    `activation2` -- (function, read-write) The activation of the
                     decoding layer. (See `SimpleLayer` docs for
                     details)
    `err` -- (function, read-write) The error function for the
             pretraining cost.  See the `NNet` documentation for
             details.
    """
    class noise(prop):
        def fget(self):
            return self.layers[0].noise
        def fset(self, val):
            self.layers[0].noise = val
    class W(prop):
        def fget(self):
            return self.layers[1].W
    class b(prop):
        def fget(self):
            return self.layers[1].b
    class W2(prop):
        def fget(self):
            return self.layers[2].W
    class b2(prop):
        def fget(self):
            return self.layers[2].b
    class activation(prop):
        def fget(self):
            return self.layers[1].activation
        def fset(self, val):
            self.layers[1].activation = val
    class activation2(prop):
        def fget(self):
            return self.layers[2].activation
        def fset(self, val):
            self.layers[2].activation = val

    def __init__(self, n_in, n_out, tied=False, nlin=tanh, noise=0.0, 
                 err=mse, dtype=theano.config.floatX, name=None,
                 rng=numpy.random, noise_rng=RandomStreams()):
        r"""
        Tests:
        >>> a = Autoencoder(20, 16, tied=True, noise=0.01)
        >>> a.noise
        0.01
        >>> a.W.value.shape
        (20, 16)
        >>> a.b.value.shape
        (16,)
        >>> a.activation
        <function tanh at ...>
        >>> a2 = test_saveload(a)
        >>> a2.noise
        0.01
        >>> theano.pp(a2.W2)
        'W.T'
        >>> a2.b2
        b2
        """
        self.tied = tied
        layer1 = SimpleLayer(n_in, n_out, activation=nlin, dtype=dtype, rng=rng)
        if self.tied:
            self._b = theano.shared(value=numpy.random.random((n_in,)).astype(dtype), name='b2')
            layer2 = SharedLayer(layer1.W.T, self._b, activation=nlin)
        else:
            layer2 = SimpleLayer(n_out, n_in, activation=nlin, dtype=dtype, rng=rng)
        NNet.__init__(self, [CorruptLayer(noise, theano_rng=noise_rng), layer1, layer2], 
                      error=err, name=name)

    def _save_(self, file):
        psave(self.tied, file)
        if self.tied:
            numpy.save(file, self._b.value)

    def _load1_(self, file):
        self.tied = pload(file)

        if self.tied:
            self._b = theano.shared(value=numpy.load(file), name='b2')
            self.layers[2].W = self.layers[1].W.T
            self.layers[2].b = self._b

    _load_ = _load1_

    def build(self, input, input_shape=None):
        r"""
        Build the layer with input expression `input`.

        Also build the network parts for pretraining.

        Tests:
        >>> x = T.fmatrix('x')
        >>> ae = Autoencoder(10, 8, tied=True, dtype=numpy.float32)
        >>> ae.build(x)
        >>> ae.input
        x
        >>> ae.pre_params
        [W, b, b2]
        >>> ae.params
        [W, b]
        >>> ae.output_shape
        >>> theano.pp(ae.output)
        'tanh(((x \\dot W) + b))'
        >>> theano.pp(ae.cost)
        '((sum(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2)) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2).shape)[0]) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2).shape)[1])'
        >>> ae = Autoencoder(3, 2, tied=False, noise=0.25, dtype=numpy.float32)
        >>> ae.build(x, (4, 3))
        >>> ae.input
        x
        >>> ae.pre_params
        [W, b, W, b]
        >>> ae.params
        [W, b]
        >>> ae.output_shape
        (4, 2)
        >>> theano.pp(ae.output)
        'tanh(((x \\dot W) + b))'
        >>> theano.pp(ae.cost)
        '((sum(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, [4 3], 1, 0.75)) \\dot W) + b)) \\dot W) + b)) - x) ** 2)) / float32(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, [4 3], 1, 0.75)) \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape)[0]) / float32(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, [4 3], 1, 0.75)) \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape)[1])'
        >>> f = theano.function([x], [ae.output, ae.cost])
        >>> r = f(numpy.random.random((4, 3)))
        >>> r[0].shape
        (4, 2)
        >>> r[0].dtype
        dtype('float32')
        """
        NNet.build(self, input, input, input_shape)
        self.pre_params = self.params
        self.layers[1].build(input, input_shape)
        self.output = self.layers[1].output
        self.output_shape = self.layers[1].output_shape
        self.params = self.layers[1].params
        if self.tied:
            self.pre_params += [self.b2]

class ConvAutoencoder(NNet):
    r"""
    Convolutional autoencoder layer.
    
    Also acts as a network if you want to do pretraining.
    
    Examples:
    >>> ca = ConvAutoencoder((5,5), 3)
    >>> ca = ConvAutoencoder((4,3), 8, noise=0.25)
    >>> ca = ConvAutoencoder((4,4), 4, dtype=numpy.float32)

    Attributes:
    `noise` -- (float, read-write) The corruption level of the input.
               Used only in pretraining.
    `filter` -- (shared tensor4, read-only) The filters used for
                encoding.
    `filter_shape` -- (complete shape, read-only) The shape of
                      `filter`.
    `b` -- (shared vector, read-only) The bias for each encoding
           filter.
    `filter2` -- (shared tensor4, read-only) The filters used for
                 decoding.
    `filter2_shape` -- (complete shape, read-only) The shape of
                       `filter2`.
    `b2` -- (shared vector, read-only) The bias for each decoding
            filter.
    `nlin` -- (function, read-write) The activation of the encoding
              layer. (See `ConvLayer` docs for details)
    `nlin2` -- (function, read-write) The activation of the decoding
               layer. (See `ConvLayer` docs for details)
    `err` -- (function, read-write) The error function for the
             pretraining cost.  See the `NNet` documentation for
             details.
    """
    class noise(prop):
        def fget(self):
            return self.layers[0].noise
        def fset(self, val):
            self.layers[0].noise = val
    class filter(prop):
        def fget(self):
            return self.layer.filter
    class filter_shape(prop):
        def fget(self):
            return self.layer.filter_shape
    class b(prop):
        def fget(self):
            return self.layer.b
    class filter2(prop):
        def fget(self):
            return self.layers[2].filter
    class filter2_shape(prop):
        def fget(self):
            return self.layers[2].filter_shape
    class b2(prop):
        def fget(self):
            return self.layers[2].b
    class nlin(prop):
        def fget(self):
            return self.layer.nlin
        def fset(self, val):
            self.layer.nlin = val
    class nlin2(prop):
        def fget(self):
            return self.layers[2].nlin
        def fset(self, val):
            self.layers[2].nlin = val

    def __init__(self, filter_size, num_filt, num_in=1, rng=numpy.random, 
                 nlin=tanh, err=mse, noise=0.0, 
                 dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> ca = ConvAutoencoder((5,5), 3)
        >>> ca.noise
        0.0
        >>> ca.filter_shape
        (3, 1, 5, 5)
        >>> ca.filter.value.shape
        (3, 1, 5, 5)
        >>> ca.b.value.shape
        (3,)
        >>> ca.filter2_shape
        (1, 3, 5, 5)
        >>> ca.b2.value.shape
        (1,)
        >>> ca.nlin
        <function tanh at ...>
        >>> ca2 = test_saveload(ca)
        >>> ca2.filter.value.shape
        (3, 1, 5, 5)
        """
        self.layer = ConvLayer(filter_size=filter_size, num_filt=num_filt,
                               num_in=num_in, nlin=nlin, rng=rng, 
                               mode='valid', dtype=dtype)
        layer1 = SharedConvLayer(self.layer.filter, self.layer.b, 
                                 self.layer.filter_shape, nlin=nlin,
                                 mode='full')
        layer2 = ConvLayer(filter_size=filter_size, num_filt=num_in,
                           dtype=dtype, num_in=num_filt, nlin=nlin, 
                           rng=rng, mode='valid')
        NNet.__init__(self, [CorruptLayer(noise), layer1, layer2],
                      error=err, name=name)
    
    def _save_(self, file):
        self.layer.savef(file)
    
    def _load1_(self, file):
        self.layer = loadf(file)
        self.layers[1].filter = self.layer.filter
        self.layers[1].b = self.layer.b
    
    _load_ = _load1_

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expression `input`

        Also builds the cost for pretraining.

        Tests:
        >>> cae = ConvAutoencoder((3,3), 2, dtype=numpy.float32)
        >>> x = T.tensor4('x', dtype='float32')
        >>> cae.build(x, input_shape=(3, 1, 32, 32))
        >>> cae.input
        x
        >>> cae.pre_params
        [b, filter, b, filter]
        >>> cae.params
        [b, filter]
        >>> cae.output_shape
        (3, 2, 30, 30)
        >>> f = theano.function([x], [cae.output, cae.cost])
        >>> r = f(numpy.random.random((3, 1, 32, 32)))
        >>> r[0].shape
        (3, 2, 30, 30)
        >>> r[0].dtype
        dtype('float32')
        >>> cae.build(x)
        >>> cae.output_shape
        >>> theano.pp(cae.output)
        "tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b)))"
        >>> theano.pp(cae.cost)
        "((((sum(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2)) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[0]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[1]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[2]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[3])"
        """
        NNet.build(self, input, input, input_shape)
        self.layer.build(input, input_shape)
        self.output = self.layer.output
        self.output_shape = self.layer.output_shape
        self.pre_params = self.params + self.layer.params
        self.params = self.layer.params

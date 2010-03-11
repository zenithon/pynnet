from pynnet.base import *
from pynnet.layers import SimpleLayer, SharedLayer, ConvLayer, SharedConvLayer
from pynnet.net import NNet
from pynnet.nlins import tanh
from pynnet.errors import mse

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['CorruptLayer', 'Autoencoder', 'ConvAutoencoder']

class CorruptLayer(BaseObject):
    def __init__(self, noisyness, theano_rng=RandomStreams()):
        r"""
        Layer that corrupts its input.

        Examples:
        >>> c = CorruptLayer(noisyness=0.25)

        Tests:
        >>> c.noisyness
        0.25
        >>> c2 = test_saveload(c)
        >>> c2.noisyness
        0.25
        """
        self.noisyness = noisyness
        self.theano_rng = theano_rng

    def _save_(self, file):
        file.write('CL1')
        psave((self.noisyness, self.theano_rng), file)

    def _load_(self, file):
        c = file.read(3)
        if c != 'CL1':
            raise ValueError('wrong cookie for CorruptLayer')
        self.noisyness, self.theano_rng = pload(file)
    
    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> c = CorruptLayer(0.2)
        >>> x = T.fmatrix('x')
        >>> c.build(x)
        >>> c.params
        []
        >>> c.input
        x
        >>> theano.pp(c.output)
        '(x * RandomFunction{binomial}(<RandomStateType>, x.shape, 1, 0.8))'
        >>> f = theano.function([x], c.output)
        >>> r = f(numpy.random.random((10, 12)))
        >>> r.shape
        (10, 12)
        >>> r.dtype
        dtype('float32')
        """
        self.input = input
        idx = self.theano_rng.binomial(size=input.shape, n=1, 
                                       prob=1-self.noisyness, dtype='int8')
        self.output = self.input * idx
        self.params = []
        

class Autoencoder(NNet):
    def __init__(self, n_in, n_out, tied=False, nlin=tanh, noisyness=0.0, 
                 err=mse, dtype=theano.config.floatX):
        r"""
        Autoencoder layer.
        
        This layer also acts as a network if you want to do pretraining.

        Examples:
        >>> a = Autoencoder(32, 20)
        >>> a = Autoencoder(20, 16, tied=True, noisyness=0.2)

        Tests:
        >>> a.layers
        [<pynnet.layers.autoencoder.CorruptLayer object at ...>, <pynnet.layers.hidden.SimpleLayer object at ...>, <pynnet.layers.hidden.SharedLayer object at ...>]
        >>> a2 = test_saveload(a)
        >>> a2.layers
        [<pynnet.layers.autoencoder.CorruptLayer object at ...>, <pynnet.layers.hidden.SimpleLayer object at ...>, <pynnet.layers.hidden.SharedLayer object at ...>]
        >>> theano.pp(a2.layers[-1].W)
        'W.T'
        >>> a2.layers[-1].b
        b2
        """
        self.tied = tied
        layer1 = SimpleLayer(n_in, n_out, activation=nlin, dtype=dtype)
        if self.tied:
            self.b = theano.shared(value=numpy.random.random((n_in,)).astype(dtype), name='b2')
            layer2 = SharedLayer(layer1.W.T, self.b, activation=nlin)
        else:
            layer2 = SimpleLayer(n_out, n_in, activation=nlin, dtype=dtype)
        layers = []
        if noisyness != 0.0:
            layers += [CorruptLayer(noisyness)]
        layers += [layer1, layer2]
        NNet.__init__(self, layers, error=err)

    def _save_(self, file):
        file.write('AE2')
        psave(self.tied, file)
        if self.tied:
            numpy.save(file, self.b)

    def _load_(self, file):
        s = file.read(3)
        if s != 'AE2':
            raise ValueError('wrong cookie for Autoencoder')
        self.tied = pload(file)

        if self.tied:
            self.b = theano.shared(value=numpy.load(file), name='b2')
            self.layers[-1].W = self.layers[-2].W.T
            self.layers[-1].b = self.b

    def build(self, input):
        r"""
        Build the layer with input expression `input`.

        Also build the network parts for pretraining.

        Tests:
        >>> x = T.fmatrix('x')
        >>> ae = Autoencoder(10, 8, tied=True, dtype=numpy.float32)
        >>> ae.build(x)
        >>> ae.input
        x
        >>> ae.params
        [W, b, b2]
        >>> theano.pp(ae.output)
        'tanh(((x \\dot W) + b))'
        >>> theano.pp(ae.cost)
        '((sum(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2)) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2).shape)[0]) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W.T) + b2)) - x) ** 2).shape)[1])'
        >>> ae = Autoencoder(3, 2, tied=False, noisyness=0.3, dtype=numpy.float32)
        >>> ae.build(x)
        >>> ae.input
        x
        >>> ae.params
        [W, b, W, b]
        >>> theano.pp(ae.output)
        'tanh(((x \\dot W) + b))'
        >>> theano.pp(ae.cost)
        '((sum(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, x.shape, 1, 0.7)) \\dot W) + b)) \\dot W) + b)) - x) ** 2)) / float32(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, x.shape, 1, 0.7)) \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape)[0]) / float32(((tanh(((tanh((((x * RandomFunction{binomial}(<RandomStateType>, x.shape, 1, 0.7)) \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape)[1])'
        >>> f = theano.function([x], [ae.output, ae.cost])
        >>> r = f(numpy.random.random((4, 3)))
        >>> r[0].shape
        (4, 2)
        >>> r[0].dtype
        dtype('float32')
        """
        NNet.build(self, input, input)
        self.layers[-2].build(input)
        self.output = self.layers[-2].output
        if self.tied:
            self.params += [self.b]
        
class ConvAutoencoder(NNet):
    def __init__(self, filter_size, num_filt, rng=numpy.random, nlin=tanh,
                 err=mse, noisyness=0.0, dtype=theano.config.floatX):
        r"""
        Convolutional autoencoder layer.

        Also acts as a network if you want to do pretraining.
        
        Examples:
        >>> ca = ConvAutoencoder((5,5), 3)

        Tests:
        >>> ca.layers
        [<pynnet.layers.conv.SharedConvLayer object at ...>, <pynnet.layers.conv.ConvLayer object at ...>]
        >>> ca2 = test_saveload(ca)
        >>> ca2.layers
        [<pynnet.layers.conv.SharedConvLayer object at ...>, <pynnet.layers.conv.ConvLayer object at ...>]
        """
        self.layer = ConvLayer(filter_size=filter_size, num_filt=num_filt,
                               nlin=nlin, rng=rng, mode='valid', dtype=dtype)
        layer1 = SharedConvLayer(self.layer.filter, self.layer.b, 
                                 self.layer.filter_shape, nlin=nlin,
                                 mode='full')
        layer2 = ConvLayer(filter_size=filter_size, num_filt=1, dtype=dtype,
                           num_in=num_filt, nlin=nlin, rng=rng, mode='valid')
        layers = []
        if noisyness != 0.0:
            layers += [CorruptLayer(noisyness)]
        layers += [layer1, layer2]
        NNet.__init__(self, layers, err)

    def _save_(self, file):
        file.write('CAE1')
        self.layer.savef(file)

    def _load_(self, file):
        s = file.read(4)
        if s != 'CAE1':
            raise ValueError('wrong cookie for ConvAutoencoder')
        self.layer = ConvLayer.loadf(file)
        self.layers[-1].filter = self.layer.filter
        self.layers[-1].b = self.layer.b

    def build(self, input):
        r"""
        Builds the layer with input expression `input`

        Also builds the cost for pretraining.

        Tests:
        >>> cae = ConvAutoencoder((3,3), 2, dtype=numpy.float32)
        >>> x = T.tensor4('x', dtype='float32')
        >>> cae.build(x)
        >>> cae.input
        x
        >>> cae.params
        [b, filter]
        >>> theano.pp(cae.output)
        "tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b)))"
        >>> theano.pp(cae.cost)
        "((((sum(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2)) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[0]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[1]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[2]) / float32(((tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 1),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(tanh((ConvOp{('imshp', None),('kshp', (3, 3)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (3, 3)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))), filter) + DimShuffle{0, x, x}(b))) - x) ** 2).shape)[3])"
        >>> f = theano.function([x], [cae.output, cae.cost])
        >>> r = f(numpy.random.random((3, 1, 32, 32)))
        >>> r[0].shape
        (3, 2, 30, 30)
        >>> r[0].dtype
        dtype('float32')
        """
        NNet.build(self, input, input)
        self.layer.build(input)
        self.output = self.layer.output
        self.params = self.layer.params

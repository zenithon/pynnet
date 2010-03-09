from pynnet.base import *
from pynnet.nlins import *

__all__ = ['ReshapeLayer', 'ConvLayer', 'MaxPoolLayer']

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class ReshapeLayer(BaseObject):
    def __init__(self, new_shape):
        r"""
        Layer to reshape the input to the given new shape.

        Examples:

        Will reshape a (N, 1024) matrix to a (N, 1, 32, 32) one 
        >>> r = ReshapeLayer((None, 1, 32, 32))

        Will convert to a (1024,) shape matrix
        >>> r = ReshapeLayer((1024,))
        
        Tests:
        >>> r = ReshapeLayer((None, 1, 32, 32))
        >>> r.outshape
        (None, 1, 32, 32)
        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> r.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> r2 = ReshapeLayer.loadf(f2)
        >>> f2.close()
        >>> r2.outshape
        (None, 1, 32, 32)
        """
        self.outshape = new_shape

    def _save_(self, file):
        file.write('RL1')
        psave(self.outshape, file)
    
    def _load_(self, file):
        c = file.read(3)
        if c != 'RL1':
            raise ValueError('Wrong magic for ReshapeLayer')
        self.outshape = pload(file)

    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> r = ReshapeLayer((None, None))
        >>> x = T.fmatrix('x')
        >>> r.build(x)
        >>> r.params
        []
        >>> r.input
        x
        >>> theano.pp(r.output)
        'Reshape{2}(x, join(0, Rebroadcast{0}(x.shape[0]), Rebroadcast{0}(x.shape[1])))'
        """
        self.input = input
        nshape = list(self.outshape)
        for i, v in enumerate(self.outshape):
            if v is None:
                nshape[i] = self.input.shape[i]
            
        self.output = input.reshape(tuple(nshape))
        self.params = []

class ConvLayer(BaseObject):
    def __init__(self, filter_size, num_filt, num_in=1, nlin=none,
                 dtype=theano.config.floatX, mode='valid',
                 rng=numpy.random, filter=None, b=None):
        r"""
        Layer that performs a convolution over the input.

        The size and number of filters are configurable through the
        `filter_size` and `num_filt` parameters.  `filter_size` is a
        2-tuple giving the 2d size of one filter.  `num_filt` is
        obviously the number of filter to apply.  `nlin` is a
        nonlinearity to apply to the result of the convolution.

        You can also specify the mode of the convolution (`mode`
        parameter), valid options are 'valid' and 'full'.
        
        The `num_in` parameter is used to specify the number of input
        maps (in case you want to stack more than one ConvLayer).
        
        The `rng` parameter can be used to specify a specific numpy
        RandomState to use.
        
        The `filter` and `b` parameters are used if you want to share
        this layer's parameters with another. (NOTE: You usually don't
        want to do this)

        Examples:
        >>> c = ConvLayer(filter_size=(5,5), num_filt=3)
        >>> c = ConvLayer(filter_size=(12,12), num_filt=7, mode='full')

        Tests:
        >>> c = ConvLayer((5,5), 3)
        >>> c.filter_shape
        (3, 1, 5, 5)

        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> c.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> c2 = ConvLayer.loadf(f2)
        >>> f2.close()
        >>> c2.filter_shape
        (3, 1, 5, 5)
        """
        self.nlin = nlin
        self.mode = mode
        self.filter_shape = (num_filt, num_in)+filter_size
        w_range = 1./numpy.sqrt(numpy.prod(filter_size)*num_filt)
        if filter is None:
            filtv = rng.uniform(low=-w_range, high=w_range, 
                                size=self.filter_shape).astype(dtype)
            self.filter = theano.shared(value=filtv, name='filter')
        else:
            self.filter = filter
        if b is None:
            bval = rng.uniform(low=-.5, high=.5,
                               size=(num_filt,)).astype(dtype)
            self.b = theano.shared(value=bval, name='b')
            
        else:
            self.b = b
    
    def _save_(self, file):
        file.write('CL1')
        psave((self.filter_shape, self.nlin, self.mode), file)
        numpy.save(file, self.filter.value)
        numpy.save(file, self.b.value)
        
    def _load_(self, file):
        c = file.read(3)
        if c != 'CL1':
            raise ValueError('wrong magic for ConvLayer')
        self.filter_shape, self.nlin, self.mode = pload(file)
        self.filter = theano.shared(numpy.load(file))
        self.b = theano.shared(numpy.load(file))

    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> import pynnet
        >>> c = ConvLayer((5,5), 2, nlin=pynnet.nlins.tanh)
        >>> x = T.tensor4('x')
        >>> c.build(x)
        >>> c.params
        [b, filter]
        >>> c.input
        x
        >>> theano.pp(c.output)
        "tanh((ConvOp{('imshp', None),('kshp', (5, 5)),('nkern', 2),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b)))"
        """
        self.input = input
        conv_out = conv.conv2d(self.input, self.filter, border_mode=self.mode,
                               filter_shape=self.filter_shape)
        self.output = self.nlin(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = [self.b, self.filter]

class MaxPoolLayer(BaseObject):
    def __init__(self, pool_shape=(2,2)):
        r"""
        MaxPooling layer
        
        The image is split into windows of size `pool_shape` and the
        maximum for each window is returned.
        
        Examples:
        >>> m = MaxPoolLayer()
        >>> m = MaxPoolLayer((3, 4))

        Tests:
        >>> m = MaxPoolLayer((3, 2))
        >>> m.pool_shape
        (3, 2)
        >>> import StringIO
        >>> f = StringIO.StringIO()
        >>> m.savef(f)
        >>> f2 = StringIO.StringIO(f.getvalue())
        >>> f.close()
        >>> m2 = MaxPoolLayer.loadf(f2)
        >>> f2.close()
        >>> m2.pool_shape
        (3, 2)
        """
        self.pool_shape = pool_shape

    def _save_(self, file):
        file.write('MPL1')
        psave(self.pool_shape, file)

    def _load_(self, file):
        c = file.read(4)
        if c != 'MPL1':
            raise ValueError('wrong magic for MaxPoolLayer')
        self.pool_shape = pload(file)

    def build(self, input):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> m = MaxPoolLayer((1,2))
        >>> x = T.fmatrix('x')
        >>> m.build(x)
        >>> m.params
        []
        >>> m.input
        x
        >>> theano.pp(m.output)
        'Reshape{2}(DownsampleFactorMax{(1, 2),True}(Reshape{4}(x, join(0, Rebroadcast{0}(Prod(x.shape[:-2])), Rebroadcast{0}([1]), Rebroadcast{0}(x.shape[-2:])))), join(0, Rebroadcast{0}(x.shape[:-2]), Rebroadcast{0}(DownsampleFactorMax{(1, 2),True}(Reshape{4}(x, join(0, Rebroadcast{0}(Prod(x.shape[:-2])), Rebroadcast{0}([1]), Rebroadcast{0}(x.shape[-2:])))).shape[-2:])))'
        """
        self.input = input
        self.output = downsample.max_pool2D(input, self.pool_shape,
                                            ignore_border=True)
        self.params = []

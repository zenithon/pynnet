from base import *
from pynnet.nlins import *

__all__ = ['ReshapeLayer', 'ConvLayer', 'SharedConvLayer', 'MaxPoolLayer']

import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class ReshapeLayer(BaseLayer):
    r"""
    Layer to reshape the input to the given new shape.

    Examples:
    
    Will reshape a (N, 1024) matrix to a (N, 1, 32, 32) one 
    >>> r = ReshapeLayer((None, 1, 32, 32))
    
    Will convert to a (1024,) shape matrix
    >>> r = ReshapeLayer((1024,))
    
    Attributes: 
    `outshape` -- (partial shape, read-write) A partial shape tuple [a
                  shape with None in the place of the missing
                  dimensions.]
    """
    def __init__(self, new_shape, name=None):
        r"""
        Tests:
        >>> r = ReshapeLayer((None, 1, 32, 32))
        >>> r.outshape
        (None, 1, 32, 32)
        >>> r2 = test_saveload(r)
        >>> r2.outshape
        (None, 1, 32, 32)
        """
        BaseLayer.__init__(self, name)
        self.outshape = new_shape

    def _save_(self, file):
        file.write('RL1')
        psave(self.outshape, file)
    
    def _load_(self, file):
        c = file.read(3)
        if c != 'RL1':
            raise ValueError('Wrong magic for ReshapeLayer')
        self.outshape = pload(file)

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> r = ReshapeLayer((None, None))
        >>> x = T.tensor3('x')
        >>> r.build(x, input_shape=(3, 2, 1))
        >>> r.params
        []
        >>> r.input
        x
        >>> r.output_shape
        (3, 2)
        >>> theano.pp(r.output)
        'Reshape{2}(x, [3 2])'
        >>> f = theano.function([x], r.output)
        >>> mat = numpy.random.random((3, 2, 1))
        >>> mat2 = f(mat)
        >>> mat2.shape
        (3, 2)
        >>> r.build(x)
        >>> r.output_shape
        """
        self.input = input
        nshape = list(self.outshape)
        for i, v in enumerate(self.outshape):
            if v is None:
                if input_shape is None:
                    nshape[i] = self.input.shape[i]
                else:
                    nshape[i] = input_shape[i]
        if input_shape:
            self.output_shape = tuple(nshape)
        else:
            self.output_shape = None
        self.output = input.reshape(tuple(nshape))
        self.params = []

class SharedConvLayer(BaseLayer):
    r"""
    Shared version of ConvLayer
    
    Examples:
    >>> filter = T.tensor4()
    >>> b = T.fvector()
    >>> c = SharedConvLayer(filter, b, (3, 1, 5, 5))

    Attributes: 
    `filter` -- (theano tensor4, read-write) A theano expression
                giving a 4D tensor representing the filters to apply
                for this convolution step.  *Not saved.*
    `filter_shape` -- (complete shape or None, read-write) The shape
                      of the filter with no missing dimensions.  Must
                      be kept in sync with the `filter` attribute.
    `b` -- (theano vector, read-write) A theano expression giving a
           vector of weights for each filter.  *Not saved.*
    `mode` -- ('full' or 'valid', read-write) A string representing
              the convolution output mode.
    `nlin` -- (function, read-write) must be a function that will
              receive as input a theano expression gives back a theano
              expression of the same shape.  Apart from the shape
              restriction any computation can be preformed on the
              input.
    """
    def __init__(self, filter, b, filter_shape, nlin=none, mode='valid',
                 name=None):
        r"""
        Tests:
        >>> filter = T.tensor4()
        >>> b = T.fvector()
        >>> c = SharedConvLayer(filter, b, (3, 1, 5, 5))
        >>> c.filter_shape
        (3, 1, 5, 5)
        >>> c2 = test_saveload(c)
        >>> c2.filter_shape
        (3, 1, 5, 5)
        """
        BaseLayer.__init__(self, name)
        self.nlin = nlin
        self.mode = mode
        self.filter_shape = filter_shape
        self.filter = filter
        self.b = b

    @classmethod
    def getoutshape(cls, filter_shape, image_shape, mode):
        r"""
        Returns the output shape for a convolution of the specified
        `mode` and arguments.

        Tests:
        >>> SharedConvLayer.getoutshape((4,1,5,5), (100,1,32,32), 'valid')
        (100, 4, 28, 28)
        >>> SharedConvLayer.getoutshape((4,1,5,5), (100,1,32,32), 'full')
        (100, 4, 36, 36)
        """
        if mode == 'valid':
            b = 1
        else:
            b = -1
        return (image_shape[0], filter_shape[0],
                image_shape[2]-b*filter_shape[2]+b,
                image_shape[3]-b*filter_shape[3]+b)
    
    def _save_(self, file):
        file.write('SCL3')
        psave((self.filter_shape, self.nlin, self.mode), file)
        
    def _load_(self, file):
        c = file.read(4)
        if c == 'SCL2':
            _, self.filter_shape, self.nlin, self.mode = pload(file)
        elif c == 'SCL3':
            self.filter_shape, self.nlin, self.mode = pload(file)
        else:
            raise ValueError('wrong cookie for SharedConvLayer')

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> filter = T.tensor4('filter')
        >>> b = T.fvector('b')
        >>> c = SharedConvLayer(filter, b, (4, 1, 5, 5))
        >>> x = T.tensor4('x')
        >>> c.build(x, (8, 1, 32, 32))
        >>> c.params
        []
        >>> c.input
        x
        >>> c.output_shape
        (8, 4, 28, 28)
        >>> theano.pp(c.output)
        "(ConvOp{('imshp', (1, 32, 32)),('kshp', (5, 5)),('nkern', 4),('bsize', 8),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 4),('unroll_kern', 4),('unroll_patch', False),('imshp_logical', (1, 32, 32)),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))"
        >>> c = SharedConvLayer(filter, b, (4, 1, 5, 5), mode='full')
        >>> c.build(x, (8, 1, 32, 32))
        >>> c.output_shape
        (8, 4, 36, 36)
        >>> theano.pp(c.output)
        "(ConvOp{('imshp', (1, 32, 32)),('kshp', (5, 5)),('nkern', 4),('bsize', 8),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 4),('unroll_kern', 4),('unroll_patch', False),('imshp_logical', (1, 32, 32)),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))"
        >>> c.build(x)
        >>> c.output_shape
        """
        self.input = input
        if input_shape and self.filter_shape:
            # These values seem to be the best or close for G5, x86 and x64
            # will have to check for other type of machines.
            un_p = False
            un_b = 4
            un_k = 4
            self.output_shape = self.getoutshape(self.filter_shape, 
                                                 input_shape,
                                                 self.mode)
        else:
            un_p = True
            un_b = 0
            un_k = 0
            self.output_shape = None
        conv_out = conv.conv2d(self.input, self.filter, border_mode=self.mode,
                               filter_shape=self.filter_shape,
                               image_shape=input_shape,
                               unroll_patch=un_p, unroll_batch=un_b,
                               unroll_kern=un_k)
        self.output = self.nlin(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        self.params = []

class ConvLayer(SharedConvLayer):
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
    
    Examples:
    >>> c = ConvLayer(filter_size=(5,5), num_filt=3)
    >>> c = ConvLayer(filter_size=(12,12), num_filt=7, mode='full')

    Attributes:
    `filter` -- (shared tensor4, read-only) A theano expression
                giving a 4D tensor representing the filters to apply
                for this convolution step.
    `filter_shape` -- (complete shape, read-only) The shape of the
                      filter with no missing dimensions.
    `b` -- (shared vector, read-only) A theano expression giving a
           vector of weights for each filter.
    `mode` -- ('full' or 'valid', read-write) A string representing
              the convolution output mode.
    `nlin` -- (function, read-write) must be a function that will
              receive as input a theano expression gives back a theano
              expression of the same shape.  Apart from the shape
              restriction any computation can be preformed on the
              input.
    """
    def __init__(self, filter_size, num_filt, num_in=1, nlin=none,
                 dtype=theano.config.floatX, mode='valid',
                 rng=numpy.random, name=None):
        r"""
        Tests:
        >>> c = ConvLayer((5,5), 3)
        >>> c.filter_shape
        (3, 1, 5, 5)
        
        >>> c2 = test_saveload(c)
        >>> c2.filter_shape
        (3, 1, 5, 5)
        """
        filter_shape = (num_filt, num_in)+filter_size
        w_range = 1./numpy.sqrt(numpy.prod(filter_size)*num_filt)
        filtv = rng.uniform(low=-w_range, high=w_range, 
                            size=filter_shape).astype(dtype)
        filter = theano.shared(value=filtv, name='filter')
        bval = rng.uniform(low=-.5, high=.5,
                           size=(num_filt,)).astype(dtype)
        b = theano.shared(value=bval, name='b')
        SharedConvLayer.__init__(self, filter, b, filter_shape, nlin=nlin, 
                                 mode=mode, name=name)
    
    def _save_(self, file):
        file.write('CL2')
        numpy.save(file, self.filter.value)
        numpy.save(file, self.b.value)
        
    def _load_(self, file):
        c = file.read(3)
        if c != 'CL2':
            raise ValueError('wrong magic for ConvLayer')
        self.filter = theano.shared(numpy.load(file), name='filter')
        self.b = theano.shared(numpy.load(file), name='b')

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> import pynnet
        >>> c = ConvLayer((5,5), 2, nlin=pynnet.nlins.tanh)
        >>> x = T.tensor4('x')
        >>> c.build(x, (2, 1, 28, 28))
        >>> c.params
        [b, filter]
        >>> c.input
        x
        >>> c.output_shape
        (2, 2, 24, 24)
        >>> theano.pp(c.output)
        "tanh((ConvOp{('imshp', (1, 28, 28)),('kshp', (5, 5)),('nkern', 2),('bsize', 2),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 2),('unroll_kern', 2),('unroll_patch', False),('imshp_logical', (1, 28, 28)),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b)))"
        >>> f = theano.function([x], c.output)
        >>> r = f(numpy.random.random((2, 1, 28, 28)))
        >>> r.shape
        (2, 2, 24, 24)
        >>> c.build(x)
        >>> c.output_shape
        """
        SharedConvLayer.build(self, input, input_shape)
        self.params = [self.b, self.filter]

class MaxPoolLayer(BaseLayer):
    r"""
    MaxPooling layer
    
    The matrix inputs (over the last 2 dimensions of the input) are
    split into windows of size `pool_shape` and the maximum for each
    window is returned.
    
    Examples:
    >>> m = MaxPoolLayer()
    >>> m = MaxPoolLayer((3, 4))

    Attributes:
    `pool_shape` -- (2-tuple, read-write) The size of the windows.
    """
    def __init__(self, pool_shape=(2,2), name=None):
        r"""
        Tests:
        >>> m = MaxPoolLayer((3, 2))
        >>> m.pool_shape
        (3, 2)
        >>> m2 = test_saveload(m)
        >>> m2.pool_shape
        (3, 2)
        """
        BaseLayer.__init__(self, name)
        self.pool_shape = pool_shape

    def _save_(self, file):
        file.write('MPL1')
        psave(self.pool_shape, file)

    def _load_(self, file):
        c = file.read(4)
        if c != 'MPL1':
            raise ValueError('wrong magic for MaxPoolLayer')
        self.pool_shape = pload(file)

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> m = MaxPoolLayer((1,2))
        >>> x = T.fmatrix('x')
        >>> m.build(x, (32, 32))
        >>> m.params
        []
        >>> m.input
        x
        >>> m.output_shape
        (32, 16)
        >>> theano.pp(m.output)
        'Reshape{2}(DownsampleFactorMax{(1, 2),True}(Reshape{4}(x, join(0, Rebroadcast{0}(Prod(x.shape[:-2])), Rebroadcast{0}([1]), Rebroadcast{0}(x.shape[-2:])))), join(0, Rebroadcast{0}(x.shape[:-2]), Rebroadcast{0}(DownsampleFactorMax{(1, 2),True}(Reshape{4}(x, join(0, Rebroadcast{0}(Prod(x.shape[:-2])), Rebroadcast{0}([1]), Rebroadcast{0}(x.shape[-2:])))).shape[-2:])))'
        >>> f = theano.function([x], m.output)
        >>> r = f(numpy.random.random((32, 32)))
        >>> r.shape
        (32, 16)
        >>> m.build(x)
        >>> m.output_shape
        >>> m = MaxPoolLayer((2,3))
        >>> m.build(x, (4, 3, 22, 21))
        >>> m.output_shape
        (4, 3, 11, 7)
        """
        self.input = input
        self.output = downsample.max_pool2D(input, self.pool_shape,
                                            ignore_border=True)
        if input_shape:
            outsh = list(input_shape)
            for i in range(-1, -len(self.pool_shape)-1, -1):
                outsh[i] /= self.pool_shape[i]
            self.output_shape = tuple(outsh)
        else:
            self.output_shape = None
        self.params = []

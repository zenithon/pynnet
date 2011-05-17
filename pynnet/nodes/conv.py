from .base import *
from pynnet.nlins import *

__all__ = ['ReshapeNode', 'ConvNode', 'SharedConvNode', 'MaxPoolNode']

from theano.tensor.nnet import conv
from theano.tensor.signal import downsample

class ReshapeNode(BaseNode):
    r"""
    Node to reshape the input to the given new shape.

    Examples:
    >>> x = T.fmatrix('x')

    Will reshape a (N, 1024) matrix to a (N, 1, 32, 32) one 
    >>> r = ReshapeNode(x, (None, 1, 32, 32))
    
    Will convert to a (1024,) shape matrix
    >>> r = ReshapeNode(x, (1024,))
    
    Attributes: 
    `outshape` -- (partial shape, read-write) A partial shape tuple [a
                  shape with None in the place of the missing
                  dimensions.]
    """
    def __init__(self, input, new_shape, name=None):
        r"""
        Tests:
        >>> x = T.tensor3('x')
        >>> r = ReshapeNode(x, (None, 1, 32, 32))
        >>> r.outshape
        (None, 1, 32, 32)
        >>> r.params
        []
        >>> r2 = test_saveload(r)
        >>> r2.outshape
        (None, 1, 32, 32)
        >>> r.params
        []
        """
        BaseNode.__init__(self, [input], name)
        self.outshape = new_shape

    def transform(self, input):
        r"""
        Builds the node with input expresstion `input`.
        
        Tests:
        >>> x = T.tensor3('x')
        >>> r = ReshapeNode(x, (None, None))
        >>> theano.pp(r.output)
        'Reshape{2}(x, [x.shape[0], x.shape[1]])'
        >>> f = theano.function([x], r.output, allow_input_downcast=True)
        >>> mat = numpy.random.random((3, 2, 1))
        >>> mat2 = f(mat)
        >>> mat2.shape
        (3, 2)
        """
        nshape = list(self.outshape)
        for i, v in enumerate(self.outshape):
            if v is None:
                nshape[i] = input.shape[i]
        return input.reshape(tuple(nshape))

class SharedConvNode(BaseNode):
    r"""
    Shared version of ConvNode
    
    Examples:
    >>> filter = T.tensor4()
    >>> b = T.fvector()
    >>> x = T.tensor4('x')
    >>> c = SharedConvNode(x, filter, b, (3, 1, 5, 5))

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
    def __init__(self, input, filter, b, filter_shape, nlin=none, mode='valid',
                 name=None):
        r"""
        Tests:
        >>> filter = T.tensor4()
        >>> b = T.fvector()
        >>> x = T.tensor4('x')
        >>> c = SharedConvNode(x, filter, b, (3, 1, 5, 5))
        >>> c.filter_shape
        (3, 1, 5, 5)
        >>> c2 = test_saveload(c)
        >>> c2.filter_shape
        (3, 1, 5, 5)
        """
        BaseNode.__init__(self, [input], name)
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
        >>> SharedConvNode.getoutshape((4,1,5,5), (100,1,32,32), 'valid')
        (100, 4, 28, 28)
        >>> SharedConvNode.getoutshape((4,1,5,5), (100,1,32,32), 'full')
        (100, 4, 36, 36)
        """
        if mode == 'valid':
            b = 1
        else:
            b = -1
        return (image_shape[0], filter_shape[0],
                image_shape[2]-b*filter_shape[2]+b,
                image_shape[3]-b*filter_shape[3]+b)
    
    def transform(self, input):
        r"""
        Builds the node with input expresstion `input`.
        
        Tests:
        >>> filter = T.tensor4('filter')
        >>> b = T.fvector('b')
        >>> x = T.tensor4('x')
        >>> c = SharedConvNode(x, filter, b, (4, 1, 5, 5))
        >>> theano.pp(c.output)
        "(ConvOp{('imshp', None),('kshp', (5, 5)),('nkern', 4),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'valid'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))"
        >>> c = SharedConvNode(x, filter, b, (4, 1, 5, 5), mode='full')
        >>> theano.pp(c.output)
        "(ConvOp{('imshp', None),('kshp', (5, 5)),('nkern', 4),('bsize', None),('dx', 1),('dy', 1),('out_mode', 'full'),('unroll_batch', 0),('unroll_kern', 0),('unroll_patch', True),('imshp_logical', None),('kshp_logical', (5, 5)),('kshp_logical_top_aligned', True)}(x, filter) + DimShuffle{0, x, x}(b))"
        """
        conv_out = conv.conv2d(input, self.filter, border_mode=self.mode,
                               filter_shape=self.filter_shape,
                               image_shape=None, unroll_patch=True,
                               unroll_batch=0, unroll_kern=0)
        return self.nlin(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))

class ConvNode(SharedConvNode):
    r"""
    Node that performs a convolution over the input.
    
    The size and number of filters are configurable through the
    `filter_size` and `num_filt` parameters.  `filter_size` is a
    2-tuple giving the 2d size of one filter.  `num_filt` is
    obviously the number of filter to apply.  `nlin` is a
    nonlinearity to apply to the result of the convolution.
    
    You can also specify the mode of the convolution (`mode`
    parameter), valid options are 'valid' and 'full'.
    
    The `num_in` parameter is used to specify the number of input
    maps (in case you want to stack more than one ConvNode).
    
    The `rng` parameter can be used to specify a specific numpy
    RandomState to use.
    
    Examples:
    >>> x = T.tensor4('x')
    >>> c = ConvNode(x, filter_size=(5,5), num_filt=3)
    >>> c = ConvNode(x, filter_size=(12,12), num_filt=7, mode='full')

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
    def __init__(self, input, filter_size, num_filt, num_in=1, nlin=none,
                 dtype=theano.config.floatX, mode='valid',
                 rng=numpy.random, name=None):
        r"""
        Tests:
        >>> x = T.tensor4('x')
        >>> c = ConvNode(x, (5,5), 3)
        >>> c.filter_shape
        (3, 1, 5, 5)
        >>> c2 = test_saveload(c)
        >>> c2.filter_shape
        (3, 1, 5, 5)
        >>> c.params
        [b, filter]
        """
        filter_shape = (num_filt, num_in)+filter_size
        w_range = 1./numpy.sqrt(numpy.prod(filter_size)*num_filt)
        filtv = rng.uniform(low=-w_range, high=w_range, 
                            size=filter_shape).astype(dtype)
        filter = theano.shared(value=filtv, name='filter')
        bval = rng.uniform(low=-.5, high=.5,
                           size=(num_filt,)).astype(dtype)
        b = theano.shared(value=bval, name='b')
        SharedConvNode.__init__(self, input, filter, b, filter_shape, 
                                 nlin=nlin, mode=mode, name=name)
        self.local_params = [filter, b]

class MaxPoolNode(BaseNode):
    r"""
    MaxPooling node
    
    The matrix inputs (over the last 2 dimensions of the input) are
    split into windows of size `pool_shape` and the maximum for each
    window is returned.
    
    Examples:
    >>> x = T.fmatrix('x')
    >>> m = MaxPoolNode(x)
    >>> m = MaxPoolNode(x, (3, 4))

    Attributes:
    `pool_shape` -- (2-tuple, read-write) The size of the windows.
    """
    def __init__(self, input, pool_shape=(2,2), name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> m = MaxPoolNode(x, (3, 2))
        >>> m.pool_shape
        (3, 2)
        >>> m2 = test_saveload(m)
        >>> m2.pool_shape
        (3, 2)
        """
        BaseNode.__init__(self, [input], name)
        self.pool_shape = pool_shape

    def transform(self, input, input_shape=None):
        r"""
        Builds the node with input expresstion `input`.
        
        Tests:
        >>> x = T.fmatrix('x')
        >>> m = MaxPoolNode(x, (1,2))
        >>> f = theano.function([x], m.output, allow_input_downcast=True)
        >>> r = f(numpy.random.random((32, 32)))
        >>> r.shape
        (32, 16)
        """
        return downsample.max_pool_2d(input, self.pool_shape,
                                      ignore_border=True)

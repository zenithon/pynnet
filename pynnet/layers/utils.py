from base import *
from itertools import izip

# Since these layers are more or less patches or fixes for small
# problems we do not export them.  They should only be used explicitly
# in wrapper functions.
__all__ = []

class IdentityLayer(BaseLayer):
    r"""
    Class that copies input to output.  Should only be useful when combining
    with layers like OpLayer or JoinLayer when you need the input untouched.

    Examples:
    >>> i = IdentityLayer()
    """
    def __init__(self, name=None):
        r"""
        Tests:
        >>> i = IdentityLayer()
        >>> i2 = test_saveload(i)
        """
        BaseLayer.__init__(self, name)

    def build(self, input, input_shape=None):
        r"""
        Build layer with input expression `input`.
        
        Tests:
        >>> i = IdentityLayer()
        >>> x = T.fmatrix('x')
        >>> i.build(x, (1, 2))
        >>> i.input
        x
        >>> i.output_shape
        (1, 2)
        >>> i.params
        []
        >>> i.output
        x
        """
        self.input = input
        self.output = input
        self.output_shape = input_shape
        self.params = []

class iClass(object):
    r"""
    Helper class to pass slices to methods.

    Examples:
    >>> i[0:1]
    slice(0, 1, None)
    >>> i[:,3:]
    (slice(None, None, None), slice(3, None, None))
    """
    def __getitem__(self, slice):
        r"""
        Returns the slice argument.
        """
        return slice

i = iClass()

class SplitLayerHelper(BaseLayer):
    r"""
    Helper class for SplitLayer.  Represents additional splits.
    """
    def __init__(self, name):
        r"""
        Tests:
        >>> s = SplitLayerHelper('SplitLayer1-2')
        >>> s.is_built
        False
        >>> s2 = test_saveload(s)
        >>> s2.is_built
        False
        """
        BaseLayer.__init__(self, name)
        self.is_built = False
    
    def _save_(self, file):
        pass
    
    def _load1_(self, file):
        self.is_built = False
    
    _load_ = _load1_
    
    def build(self, input, input_shape=None):
        r"""
        :notests:
        """
        assert self.is_built, "Must build the parent SplitLayer before building dependent layers"

class SplitLayer(BaseLayer):
    r"""
    Class to split an input expression into one or more pieces.
    
    The regions are specified by a list of tuples representing indexes
    into the input. The split regions may overlap.

    The first split is the output of the main class and the other will
    be the outputs of the sublayers accessible though `helpers`.

    Note: To use the convinient syntax in the examples below import
    `i` from this module along with SplitLayer.  Otherwise you can
    pass instances of `slice` with the appropriate parameters.

    Note2: This class doesn't compute its output shape in any case.
    (since I am lazy)  If you want to do it I will be glad.

    Examples:
    >>> s = SplitLayer([i[:,0], i[:,1:]])
    >>> s = SplitLayer([i[:100,:], i[50:150,:], i[100:,:]])
    >>> s = SplitLayer([i[1], i[2], i[3]])

    Attributes:
    `splits` -- (list of slices, read-only) The list of indexes for
                each split.
    `helpers` -- (tuple of layers, read-only) Helper layers for splits
                 other than the first.
    """
    def __init__(self, splits, name=None):
        r"""
        Tests:
        >>> s = SplitLayer([i[0,2], i[1:,:]])
        >>> s.splits
        [(0, 2), (slice(1, None, None), slice(None, None, None))]
        >>> s2 = test_saveload(s)
        >>> s2.splits
        [(0, 2), (slice(1, None, None), slice(None, None, None))]
        """
        BaseLayer.__init__(self, name)
        self.splits = splits
        name = self.name
        self.helpers = tuple(SplitLayerHelper(name+'-'+str(i+2)) for i, _ in enumerate(splits[1:]))

    def _save_(self, file):
        psave(self.splits, file)

    def _load1_(self, file):
        self.splits = pload(file)

    _load_ = _load1_

    def build(self, input, input_shape=None):
        r"""
        Build layer with input expression `input`.
        
        Tests:
        >>> s = SplitLayer([i[1:3]])
        >>> x = T.fmatrix('x')
        >>> s.build(x)
        >>> s.input
        x
        >>> s.output_shape # always None for now
        >>> s.params
        []
        >>> theano.pp(s.output)
        'x[1:3]'
        >>> s = SplitLayer([i[:,0:2], i[:,2:]])
        >>> s.build(x)
        >>> s.helpers[0].input
        x
        >>> s.helpers[0].output_shape # also always none for now
        >>> s.helpers[0].params
        []
        >>> theano.pp(s.helpers[0].output)
        'x[:, 2:]'
        """
        self.input = input
        self.params = []
        self.output = input[self.splits[0]]
        if input_shape is None:
            self.output_shape = None
            helper_shape = [None for h in self.helpers]
        else:
            # computing the output shape is hard and I'm lazy
            # I'll do it later, maybe
            self.output_shape = None
            helper_shape = [None for h in self.helpers]
            
        for split, helper, shape in izip(self.splits[1:], self.helpers, helper_shape):
            helper.input = input
            helper.params = []
            helper.output = input[split]
            helper.output_shape = shape
            helper.is_built = True

class ParaLayer(CompositeLayer):
    r"""
    Class to run more than one layer in parallel.

    Each layer must take the same input shape.  The output shape must
    have the same 2nd dimension but may differ in the first.  The end
    result of each layer is concatenated along the first dimension.

    Examples:
    >>> p = ParaLayer([SimpleLayer(10, 3), SimpleLayer(10, 2)])

    Attributes:
    `sublayers` -- (list of layers, read-write) The list of layer to
                   run in parallel.
    """
    def __init__(self, sublayers, name=None):
        r"""
        >>> p = ParaLayer([SimpleLayer(10, 3), SimpleLayer(10, 2)])
        >>> len(p.sublayers)
        2
        >>> p2 = test_saveload(p)
        >>> len(p2.sublayers)
        2
        """
        CompositeLayer.__init__(self, name, sublayers)
        self.sublayers = sublayers
        
    def build(self, input, input_shape=None):
        r"""
        Build layer with input expression `input`.
        
        Tests:
        >>> p = ParaLayer([SimpleLayer(10, 2, dtype='float32'), SimpleLayer(10, 3, dtype='float32')])
        >>> x = T.fmatrix('x')
        >>> p.build(x, input_shape=(4,10))
        >>> p.input
        x
        >>> p.output_shape
        (4, 5)
        >>> p.params
        [W, b, W, b]
        >>> theano.pp(p.output)
        'join(1, tanh(((x \\dot W) + b)), tanh(((x \\dot W) + b)))'
        >>> f = theano.function([x], p.output)
        >>> r = f(numpy.random.random((4,10)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (4, 5)
        """
        self.input = input
        for l in self.sublayers:
            l.build(input, input_shape)
            
        self.output = T.join(1, *(l.output for l in self.sublayers))
        if input_shape is None:
            self.output_shape = None
        else:
            self.output_shape = (self.sublayers[0].output_shape[0],
                               sum(l.output_shape[1] for l in self.sublayers))
        self.params = sum((l.params for l in self.sublayers), [])

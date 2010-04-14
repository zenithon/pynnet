from base import *

__all__ = ['LayerStack']

class LayerStack(CompositeLayer):
    r"""
    Stack of layers that acts as a layer.
    
    Examples:
    >>> from pynnet.layers import *
    >>> l = LayerStack([ReshapeLayer((None, 1, 32, 32)), 
    ...                 ConvLayer((5,5), 4)])
    >>> l2 = LayerStack([SimpleLayer(1024, 1024), l])

    Attributes:
    `layers` -- (list, read-write) The list of layers in their stack
                order.
    """
    def __init__(self, layers, name=None):
        r"""
        Tests:
        >>> from pynnet.layers import *
        >>> l = LayerStack([ReshapeLayer((None, 1, 32, 32)), 
        ...                 ConvLayer((5,5), 4, name='cl')])
        >>> l.layers
        [ReshapeLayer..., cl]
        >>> ll = test_saveload(l)
        >>> ll.layers
        [ReshapeLayer..., cl]
        >>> ll.get_layer('cl')
        cl
        """
        CompositeLayer.__init__(self, name, layers)
        self.layers = layers

    def _save_(self, file):
        file.write('LS1')
        psave([l.__class__ for l in self.layers], file)
        for l in self.layers:
            l.savef(file)

    def _load_(self, file):
        c = file.read(3)
        if c != 'LS1':
            raise ValueError('wrong cookie for LayerStack')
        lclass = pload(file)
        self.layers = [c.loadf(file) for c in lclass]
        self.add(self.layers)

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expression `input`.
        
        Tests:
        >>> from pynnet.layers import *
        >>> import theano
        >>> x = theano.tensor.tensor3('x', dtype='float32')
        >>> s = LayerStack([ReshapeLayer((None, 1024)),
        ...                 SimpleLayer(1024, 1024, dtype=numpy.float32)])
        >>> s.build(x, input_shape=(3, 32, 32))
        >>> s.input
        x
        >>> s.params
        [W, b]
        >>> s.output_shape
        (3, 1024)
        >>> theano.pp(s.output)
        'tanh(((Reshape{2}(x, [   3 1024]) \\dot W) + b))'
        >>> f = theano.function([x], s.output)
        >>> r = f(numpy.random.random((3, 32, 32)))
        >>> r.shape
        (3, 1024)
        >>> r.dtype
        dtype('float32')
        >>> s.build(x)
        >>> s.output_shape
        """
        self.input = input
        for l in self.layers:
            l.build(input, input_shape)
            input = l.output
            input_shape = l.output_shape
        self.output = input
        self.output_shape = input_shape
        self.params = sum((l.params for l in self.layers), [])

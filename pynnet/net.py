from base import *
from pynnet.layers.composite import LayerStack

__all__ = ['NNet']

class NNet(LayerStack):
    r"""
    Create a new fully-connected neural network with the given parameters.
    
    Arguements:
    layers -- A list of layers.
    error -- The error function to use.
    
    Examples:
    
    A xor net
    >>> from pynnet.layers import *
    >>> n = NNet([SimpleLayer(2,2, nlin=nlins.tanh),
    ...           SimpleLayer(2,1, nlin=nlins.none)],
    ...          error=errors.mse)
    
    A net with no hidden layers
    >>> n = NNet([SimpleLayer(5, 2)], errors.mse)
    
    A more complex net
    >>> n = NNet([SimpleLayer(20, 50, nlin=nlins.tanh),
    ...           SimpleLayer(50, 50, nlin=nlins.sigmoid),
    ...           SimpleLayer(50, 10, nlin=nlins.sigmoid),
    ...           SimpleLayer(50, 1, nlin=nlins.none)],
    ...          error=errors.nll)

    Attributes:
    `err` -- (function, read-write) must be a function that will
             recive as input a theano expression for a matrix and will
             give back a theano expression for a scalar.  Apart from
             the shape restriction, any computation may be performed
             on the input.
    `layers` -- (list, read-write) The list of layers in their stack
                order. 
    """
    def __init__(self, layers, error, name=None):
        r"""
        Tests:
        >>> net = NNet([SimpleLayer(2,2, nlin=nlins.tanh),
        ...             SimpleLayer(2,1, nlin=nlins.none)],
        ...            error=errors.mse)
        >>> net.layers
        [SimpleLayer..., SimpleLayer...]
        >>> net2 = test_saveload(net)
        >>> net2.layers
        [SimpleLayer..., SimpleLayer...]
        """
        LayerStack.__init__(self, layers, name=name)
        self.err = error
    
    def _save_(self, file):
        file.write('NN3')
        psave(self.err, file)
    
    def _load_(self, file):
        s = file.read(3)
        if s != 'NN3':
            raise ValueError('wrong cookie for NNet')
        self.err = pload(file)
    
    def build(self, input, target, input_shape=None):
        r""" 
        Build the network from input `input`, with cost against
        `target`.
        
        Tests:
        >>> x = theano.tensor.fmatrix('x')
        >>> y = theano.tensor.fvector('y')
        >>> n = NNet([SimpleLayer(3,2, dtype=numpy.float32),
        ...           SimpleLayer(2,3, dtype=numpy.float32)],
        ...          error=errors.mse)
        >>> n.build(x, y)
        >>> n.input
        x
        >>> n.params
        [W, b, W, b]
        >>> theano.pp(n.output)
        'tanh(((tanh(((x \\dot W) + b)) \\dot W) + b))'
        >>> theano.pp(n.cost)
        '((sum(((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2)) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2).shape)[0]) / float32(((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2).shape)[1])'
        """
        LayerStack.build(self, input, input_shape)
        self.cost = self.err(self.output, target)

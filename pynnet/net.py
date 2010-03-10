from base import *

from errors import *

__all__ = ['NNet']

class NNet(BaseObject):
    def __init__(self, layers, error=mse):
        r"""
        Create a new fully-connected neural network with the given parameters.
        
        Arguements:
        layers -- A list of layers.
        error -- (default: mse) The error function to use.
        
        Examples:
        
        A xor net
        >>> from pynnet.layers import *
        >>> n = NNet([SimpleLayer(2,2, activation=nlins.tanh),
        ...           SimpleLayer(2,1, activation=nlins.none)])
        
        A net with no hidden layers
        >>> n = NNet([SimpleLayer(5, 2)])
        
        A more complex net
        >>> n = NNet([SimpleLayer(20, 50, activation=nlins.tanh),
        ...           SimpleLayer(50, 50, activation=nlins.sigmoid),
        ...           SimpleLayer(50, 10, activation=nlins.sigmoid),
        ...           SimpleLayer(50, 1, activation=nlins.none)], error=nll)
        
        TESTS::
        >>> net = NNet([SimpleLayer(2,2, activation=nlins.tanh),
        ...             SimpleLayer(2,1, activation=nlins.none)])
        >>> net2 = test_saveload(net)
        >>> net2.layers
        [<pynnet.layers.hidden.SimpleLayer object at ...>, <pynnet.layers.hidden.SimpleLayer object at ...>]
        """
        self.layers = layers
        self.err = error
    
    def _save_(self, file):
        r"""save state to a file"""
        file.write('NN2')
        psave((self.err, [l.__class__ for l in self.layers]), file)
        for l in self.layers:
            l.savef(file)
    
    def _load_(self, file):
        r"""load state from a file"""
        s = file.read(3)
        if s != 'NN2':
            raise ValueError('wrong cookie for NNet')
        self.err, lclass = pload(file)
        self.layers = [c.loadf(file) for c in lclass]
    
    def build(self, input, target):
        r""" 
        Build the network from input `input`, with cost against
        `target`.
        
        Tests:
        >>> x = theano.tensor.fmatrix('x')
        >>> y = theano.tensor.fvector('y')
        >>> n = NNet([SimpleLayer(3,2),
        ...           SimpleLayer(2,3)])
        >>> n.build(x, y)
        >>> n.input
        x
        >>> n.params
        [W, b, W, b]
        >>> theano.pp(n.output)
        'tanh(((tanh(((x \\dot W) + b)) \\dot W) + b))'
        >>> theano.pp(n.cost)
        '((sum(((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2)) / ((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2).shape[0]) / ((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - y) ** 2).shape[1])'
        """
        self.input = input
        for l in self.layers:
            l.build(input)
            input = l.output
        self.output = input
        self.params = sum((l.params for l in self.layers), [])
        self.cost = self.err(self.output, target)

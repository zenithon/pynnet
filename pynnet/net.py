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
        >>> n = NNet([Layer(2,2, activation=nlins.tanh),
        ...           Layer(2,1, activation=nlins.none)])
        
        A net with no hidden layers
        >>> n = NNet([Layer(5, 2)])
        
        A more complex net
        >>> n = NNet([Layer(20, 50, activation=nlins.tanh),
        ...           Layer(50, 50, activation=nlins.sigmoid),
        ...           Layer(50, 10, activation=nlins.sigmoid),
        ...           Layer(50, 1, activation=nlins.none)], error=nll)
        
        TESTS::
        >>> net = NNet([Layer(2,2, activation=nlins.tanh),
        ...             Layer(2,1, activation=nlins.none)])
        >>> net2 = test_saveload(net)
        >>> net2.layers
        [<pynnet.layers.hidden.Layer object at ...>, <pynnet.layers.hidden.Layer object at ...>]
        """
        self.layers = layers
        self.err = error
    
    def _save_(self, file):
        r"""save state to a file"""
        file.write('NN2')
        psave((self.layers, [l.__class__ for l in self.layers]), file)
        for l in self.layers:
            l.savef(file)
    
    def _load_(self, file):
        r"""load state from a file"""
        s = file.read(3)
        if s != 'NN2':
            raise ValueError('wrong cookie for NNet')
        self.err, lclass = pload(file)
        self.layers = [c.loadf(file) for c in lclass]
    
    def build(self, input):
        r"""
        Build the network from input `input`
        
        Tests:
        >>> x = theano.tensor.fmatrix('x')
        >>> n = NNet([Layer(3,2),
        ...           Layer(2,3)])
        >>> n.build(x)
        >>> n.input
        x
        >>> n.params
        [W, b, W, b]
        >>> theano.pp(n.output)
        'tanh(((tanh(((x \\dot W) + b)) \\dot W) + b))'
        >>> theano.pp(n.cost)
        '((sum(((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - x) ** 2)) / ((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape[0]) / ((tanh(((tanh(((x \\dot W) + b)) \\dot W) + b)) - x) ** 2).shape[1])'
        """
        self.input = input
        for l in self.layers:
            l.build(input)
            input = l.output
        self.output = input
        self.params = sum((l.params for l in self.layers), [])
        self.cost = self.err(self.output, self.input)

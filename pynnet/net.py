from base import *

__all__ = ['NNet']

class NNet(BaseObject):
    r"""
    Create a new fully-connected neural network with the given parameters.
    
    Arguements:
    layers -- A list of layers.
    error -- The error function to use.
    
    Examples:

    >>> x = T.fmatrix('x')
    >>> y = T.fmatrix('y')
    
    A xor net

    >>> l1 = SimpleNode(x, 2, 2, nlin=nlins.tanh)
    >>> l2 = SimpleNode(l1, 2, 1, nlin=nlins.none)
    >>> n = NNet(l2, y, error=errors.mse)
    
    A net with no hidden layers
    >>> n = NNet(SimpleNode(x, 5, 2), y, errors.mse)
    
    A more complex net
    >>> l1 = SimpleNode(x, 20, 50, nlin=nlins.tanh)
    >>> l2 = SimpleNode(l1, 50, 50, nlin=nlins.sigmoid)
    >>> l3 = SimpleNode(l2, 50, 10, nlin=nlins.sigmoid)
    >>> l4 = SimpleNode(l3, 50, 1, nlin=nlins.none)
    >>> n = NNet(l4, y, error=errors.nll)

    Attributes:
    `err` -- (function, read-write) must be a function that will
             recive as input a theano expression for a matrix and will
             give back a theano expression for a scalar.  Apart from
             the shape restriction, any computation may be performed
             on the input.
    `graph` -- (node, read-write) The graph of nodes that make up the
               net.
    `target` -- (theano expr, read-write) The target value used by the
                cost.
    """
    def __init__(self, graph, target, error):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> y = T.fmatrix('y')
        >>> l1 = SimpleNode(x, 2, 2, nlin=nlins.tanh)
        >>> l2 = SimpleNode(l1, 2, 1, nlin=nlins.none)
        >>> n = NNet(l2, y, error=errors.mse)
        >>> theano.pp(n.cost)
        '((sum(((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2)) / ((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2).shape[0]) / ((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2).shape[1])'
        >>> n2 = test_saveload(n)
        >>> theano.pp(n2.cost)
        '((sum(((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2)) / ((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2).shape[0]) / ((((tanh(((x \\dot W) + b)) \\dot W) + b) - y) ** 2).shape[1])'
        """
        self.graph = graph
        self.target = target
        self.error = error

    class cost(prop):
        def fget(self):
            return self.error(self.graph.output, self.target)
    class params(prop):
        def fget(self):
            return self.graph.params
    class output(prop):
        def fget(self):
            return self.graph.output
        

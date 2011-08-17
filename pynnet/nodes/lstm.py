from .base import *
from pynnet.nlins import tanh, sigmoid
from .simple import SimpleNode
from .recurrent import RecurrentNode, RecurrentMemory
from .oper import AddNode, MulNode

__all__ = ['LSTMNode']

broad0 = T.Rebroadcast((0, True))

class _broad0(BaseNode):
    def __init__(self, input):
        BaseNode.__init__(self, [input], None)
    transform = broad0

class LSTMNode(RecurrentNode):
    r"""
    Node representing a single LSTM block.

    This node includes a peephole and a forget gate.

    Examples:
    >>> x = T.fmatrix('x')
    >>> l = LSTMNode(x, 20, 5, dtype='float32')
    """
    def __init__(self, input, in_size, n_cells, mapnlin=tanh, out_bias=0.0,
                 rng=numpy.random, name=None, dtype=theano.config.floatX):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> l = LSTMNode(x, 10, 2, dtype='float32')
        >>> f = theano.function([x], l.output)
        >>> r = f(numpy.random.random((8, 10)).astype('float32'))
        >>> r.shape
        (8, 2)
        """
        mem = RecurrentMemory(numpy.zeros((n_cells,), dtype=dtype))
        map_in = SimpleNode(input, in_size, n_cells, nlin=mapnlin,
                            dtype=dtype, rng=rng)
        gate_in = _broad0(SimpleNode([input, mem], [in_size, n_cells], 1,
                                     nlin=sigmoid, dtype=dtype, rng=rng))
        gate_forget = _broad0(SimpleNode([input, mem], [in_size, n_cells], 1,
                                         nlin=sigmoid, dtype=dtype, rng=rng))
        gate_out = _broad0(SimpleNode([input, mem], [in_size, n_cells], 1,
                                      nlin=sigmoid, dtype=dtype, rng=rng,
                                      b_init=out_bias))
        gin = MulNode(map_in, gate_in)
        gocec = MulNode(mem, gate_forget)
        gcec = AddNode(gin, gocec)
        gout = MulNode(gcec, gate_out)
        mem.subgraph = gcec
        
        RecurrentNode.__init__(self, [input], [], mem, gout, 
                               name=name, nopad=True)

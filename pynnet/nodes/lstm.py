from pynnet.nodes.base import *
from pynnet.nlins import tanh, sigmoid
from pynnet.nodes import SimpleNode, RecurrentWrapper

__all__ = ['lstm_block']

Broad1 = T.Rebroadcast((1, True))

class CEC(BaseNode):
    r"""
    The CEC class for LSTMs.

    See `lstm_block` or `lstm_block_peep` for user-friendly wrappers
    or the code of these functions for example usage.

    Inputs:
    `map_in` -- (node, read-only) The mapping applied to the input
                before going in the CEC.
    `gate_in` -- (node, read-only) The gating function for input.
    `gate_out` -- (node, read-only) The gating function for output.

    Attributes:
    `cec` -- (shared var, read-only) The memory.
    """
    def __init__(self, input, gate_in, gate_forget, gate_out, size, name=None,
                 dtype=theano.config.floatX):
        BaseNode.__init__(self, [input, gate_in, gate_forget, gate_out], name)
        self.cec = theano.shared(numpy.zeros((size,), dtype=dtype), 
                                 name='cec')
    
    def clear(self):
        val = self.cec.value.copy()
        val[:] = 0
        self.cec.value = val

    def transform(self, input, gate_in, gate_forget, gate_out):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> g = SimpleNode(x, 3, 1, dtype='float32', nlin=sigmoid)
        >>> c = CEC(x, g, g, g, 3, dtype='float32')
        >>> theano.pp(c.output)
        '(Rebroadcast{?,1}(sigmoid(((x \\dot W) + b))) * <theano.scan.Scan object at ...>(?_steps, Rebroadcast{?,1}(sigmoid(((x \\dot W) + b))), (x * Rebroadcast{?,1}(sigmoid(((x \\dot W) + b)))), cec))'
        >>> f = theano.function([x], c.output)
        >>> r = f(numpy.random.random((4, 3)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (4, 3)
        """
        final_in = input * Broad1(gate_in)

        def cecf(forget, cell_input, outp):
            return outp * forget + cell_input

        outs, upds = theano.scan(cecf,
                                 sequences=[Broad1(gate_forget), final_in],
                                 outputs_info=[self.cec])

        for s, u in upds.iteritems():
            s.default_update = u
        self.cec.default_update = outs[-1]

        return Broad1(gate_out) * outs

def lstm_block(input, in_size, n_cells, rng=numpy.random, 
               dtype=theano.config.floatX):
    r"""
    An single LSTM block with memory cells sharing gates.

    Examples:
    >>> x = T.fmatrix('x')
    >>> l = lstm_block(x, 20, 2)
    >>> l2 = test_saveload(l)
    """
    map_in = SimpleNode(input, in_size, n_cells)
    gate_in = SimpleNode(input, in_size, 1, nlin=sigmoid,
                         dtype=dtype, rng=rng)
    gate_forget = SimpleNode(input, in_size, 1, nlin=sigmoid,
                             dtype=dtype, rng=rng)
    gate_out = SimpleNode(input, in_size, 1, nlin=sigmoid,
                          dtype=dtype, rng=rng)
    block = CEC(map_in, gate_in, gate_forget, gate_out, n_cells, dtype=dtype)
    return block

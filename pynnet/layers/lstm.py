from pynnet.layers.base import *
from pynnet.nlins import tanh, sigmoid
from pynnet.layers import SimpleLayer, RecurrentWrapper

__all__ = ['LSTMBlock']

class LSTMBlock(BaseLayer):
    r"""
    An LSTM block of memory cells.

    Examples:
    >>> l = LSTMBlock(20, 2)

    Attributes:
    `map_in` -- (SimpleLayer, read-only [you can modify attributes])
                The mapping applied to the input before going in the CEC.
    `gate_in` -- (SimpleLayer, read-only) The gating function for input.
    `gate_out` -- (SimpleLayer, read-only) The gating function for output.
    `h` -- (function, read-write) the nonlinearity applied to the
           output of the CEC.
    `cec` -- (Layer, read-only) The central memory unit.  This is
             where magic happens.
    `peephole` -- (boolean, read-only) Reports whether the block was
                  initialzed with a peephole from the CECs to the
                  gates (cannot be changed after initialzation)
    """
    def __init__(self, in_size, n_cells=1, g=tanh, h=tanh, peephole=False,
                 name=None, dtype=theano.config.floatX, rng=numpy.random):
        r"""
        Tests:
        >>> l = LSTMBlock(10, 2)
        >>> l.map_in.W.value.shape
        (10, 2)
        >>> l2 = test_saveload(l)
        >>> l.map_in.W.value.shape
        (10, 2)
        """
        BaseLayer.__init__(self, name)
        self.peephole = peephole
        self.map_in = SimpleLayer(in_size, n_cells, nlin=g,
                                  dtype=dtype, rng=rng)
        if self.peephole:
            in_size += n_cells
        self.gate_in = SimpleLayer(in_size, n_cells, nlin=sigmoid,
                                   dtype=dtype, rng=rng)
        self.gate_forget = SimpleLayer(in_size, n_cells, nlin=sigmoid,
                                       dtype=dtype, rng=rng)
        self.cec = theano.shared(numpy.zeros((n_cells,), dtype=dtype), 
                                 name='cec')
        self.gate_out = SimpleLayer(in_size, n_cells, nlin=sigmoid,
                                    dtype=dtype, rng=rng)

    def _save_(self, file):
        self.map_in.savef(file)
        self.gate_in.savef(file)
        self.gate_forget.savef(file)
        psave((self.cec.value.shape, self.cec.value.dtype, self.peephole), file)
        self.gate_out.savef(file)

    def _load1_(self, file):
        self.map_in = loadf(file)
        self.gate_in = loadf(file)
        self.gate_forget = loadf(file)
        shp, dtype = pload(file)
        self.peephole = False
        self.cec = theano.shared(numpy.zeros(shp, dtype=dtype), name='cec')
        self.gate_out = loadf(file)

    def _load2_(self, file):
        self.map_in = loadf(file)
        self.gate_in = loadf(file)
        self.gate_forget = loadf(file)
        shp, dtype, self.peephole = pload(file)
        self.cec = theano.shared(numpy.zeros(shp, dtype=dtype), name='cec')
        self.gate_out = loadf(file)

    _load_ = _load2_

    def build(self, input, input_shape=None):
        r"""
        Builds the layer with input expresstion `input`.
        
        Tests:
        >>> l = LSTMBlock(12, 3, dtype='float32', peephole=False)
        >>> x = T.fmatrix('x')
        >>> l.build(x, input_shape=(10, 12))
        >>> l.params
        [W, b, W, b, W, b, W, b]
        >>> l.input
        x
        >>> l.output_shape
        (10, 3)
        >>> theano.pp(l.output)
        '(sigmoid(((x \\dot W) + b)) * <theano.scan.Scan object at ...>(?_steps, sigmoid(((x \\dot W) + b)), (tanh(((x \\dot W) + b)) * sigmoid(((x \\dot W) + b))), cec))'
        >>> f = theano.function([x], l.output)
        >>> r = f(numpy.random.random((10, 12)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (10, 3)
        >>> l.build(x)
        >>> l.output_shape

        # Now again with peephole=True
        >>> l = LSTMBlock(12, 3, dtype='float32', peephole=True)
        >>> x = T.fmatrix('x')
        >>> l.build(x, input_shape=(10, 12))
        >>> l.params
        [W, b, W, b, W, b, W, b]
        >>> l.input
        x
        >>> l.output_shape
        (10, 3)
        >>> theano.pp(l.output)
        '(sigmoid(((join(1, x, <theano.scan.Scan object at ...>(?_steps, x, tanh(((x \\dot W) + b)), cec, W, b, W, b)) \\dot W) + b)) * <theano.scan.Scan object at ...>(?_steps, x, tanh(((x \\dot W) + b)), cec, W, b, W, b))'
        >>> f = theano.function([x], l.output)
        >>> r = f(numpy.random.random((10, 12)))
        >>> r.dtype
        dtype('float32')
        >>> r.shape
        (10, 3)
        >>> l.build(x)
        >>> l.output_shape
        """
        self.input = input
        self.map_in.build(input, input_shape)
        if self.peephole:
            def block(inp, cell_inp, outp):
                inp = T.join(0, inp, outp)
                if input_shape is None:
                    ishp = None
                else:
                    ishp = (1, self.map_in.output_shape[1]+input_shape[1])
                self.gate_in.build(inp, ishp)
                self.gate_forget.build(inp, ishp)
                return ((self.gate_in.output*cell_inp) + \
                    (self.gate_forget.output*outp))
            outs, upds = theano.scan(block,
                                     sequences=[input, self.map_in.output],
                                     outputs_info=[self.cec])
            if input_shape is None:
                oshp = None
            else:
                oshp = (input_shape[0], input_shape[1]+self.map_in.output_shape[1])
            self.gate_out.build(T.join(1, input, outs), oshp)
        else:
            self.gate_in.build(input, input_shape)
            final_in = self.map_in.output * self.gate_in.output
            self.gate_forget.build(input, input_shape)
        
            def cecf(forget, cell_input, outp):
                return outp * forget + cell_input
            outs, upds = theano.scan(cecf,
                                     sequences=[self.gate_forget.output, final_in],
                                     outputs_info=[self.cec])

            self.gate_out.build(input, input_shape)

        for s, u in upds.iteritems():
            s.default_update = u
        
        self.cec.default_update = outs[-1]

        if input_shape is None:
            self.output_shape = None
        else:
            assert len(input_shape) == 2, "LSTM needs 2d input"
            self.output_shape = self.gate_out.output_shape

        self.output = self.gate_out.output * outs
        self.params = self.gate_in.params + self.map_in.params + \
                       self.gate_forget.params + self.gate_out.params
        

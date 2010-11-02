from pynnet.nodes.base import *
from pynnet.nodes.simple import SimpleNode
from pynnet.nlins import *
from pynnet.errors import cross_entropy

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['DelayNode', 'RecurrentWrapper']

class DelayNode(BaseNode):
    r"""
    This is a node that inserts a delay of its input in the graph.
    
    Examples:
    >>> x = T.fmatrix()
    >>> d = DelayNode(x, 3, [[1, 2], [2, 2], [2, 1]])
    """
    def __init__(self, input, delay, init_vals, name=None):
        BaseNode.__init__(self, [input], name)
        self.delay = delay
        self.memory = theano.shared(init_vals, 'delaymem')

    def transform(self, input):
        j = T.join(0, self.memory, input)
        self.memory.default_update = j[-self.delay:]
        return j[:-self.delay]

class RecurrentWrapper(BaseNode):
    r"""
    This is a recurrent node with a one tap delay.  This means it
    gets it own output from the previous step in addition to the
    input provided at each step.

    The memory is automatically updated and starts with a zero fill.
    If you want to clear the memory at some point, use the clear()
    function.  It will work on any backend and with any shape.  You
    may have problems on the GPU (and maybe elsewhere) otherwise.

    The recurrent part of the graph is built by the provided
    `subgraph_builder` function.

    This wrapper will not work for subgraphs with more than one input
    or output at the moment.  There are plans to fix that in the
    future.

    Attributes:
    `subgraph_builder` -- (function, read-write) a function which
                          builds the recursive part of the graph.
    `mem_init` -- (array_like, read-only) Initial value for the memory.
    """
    def __init__(self, input, subgraph_builder, mem_init=None, 
                 outshp=None, dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> x_n = InputNode(x)
        >>> r = RecurrentWrapper(x, lambda x_n: SimpleNode(x_n, 10, 5, dtype='float32'),
        ...                      outshp=(5,), dtype='float32')
        >>> r.memory.value
        array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)
        """
        BaseNode.__init__(self, [input], name)
        if mem_init is None:
            mem_init = numpy.zeros(outshp, dtype=dtype)
        self.mem_init = mem_init
        self.memory = theano.shared(self.mem_init.copy(), name='memory')
        self.subgraph_builder = subgraph_builder
        self.local_params = self.subgraph_builder(*self.inputs).params

    def clear(self):
        r"""
        Resets the memory to initial value.
        """
        self.memory.value = self.mem_init.copy()
    
    def transform(self, input):
        r"""
        Builds the node with input expresstion `input`.
        
        Tests:
        >>> x = T.fmatrix('x')
        >>> x_n = InputNode(x)
        >>> r = RecurrentWrapper(x, lambda inp: SimpleNode(inp, 5, 2, dtype='float32'),
        ...                      outshp=(2,), dtype='float32')
        >>> r.params
        [W, b]
        >>> theano.pp(r.output)
        '<theano.scan.Scan object at ...>(?_steps, x, memory, W, b)'
        >>> f = theano.function([x], r.output)
        >>> v = f(numpy.random.random((4, 3)))
        >>> v.dtype
        dtype('float32')
        >>> v.shape
        (4, 2)
        >>> (r.memory.value == v[-1]).all()
        True
        >>> r=RecurrentWrapper(x, lambda inp: RecurrentWrapper(inp, lambda inp2: SimpleNode(inp2, 6,2), outshp=(2,)), outshp=(2,))
        >>> f = theano.function([x], r.output)
        >>> v = f(numpy.random.random((3, 2)))
        >>> v.shape
        (3, 2)
        >>> (r.memory.value == v[-1]).all()
        True
        """
        def f(inp, mem):
            i = InputNode(T.unbroadcast(T.shape_padleft(T.join(0,inp,mem)),0))
            g = self.subgraph_builder(i)
            return g.output[0]
        
        outs,upds = theano.scan(f,sequences=[input],outputs_info=[self.memory])
        
        for s, u in upds.iteritems():
            s.default_update = u
        self.memory.default_update = outs[-1]
        return outs

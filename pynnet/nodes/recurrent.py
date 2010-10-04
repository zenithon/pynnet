from base import *
from simple import SimpleNode
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

    No example usage is provided as the relation between arguments is
    rather complex.  The recommended usage is to use the provided
    wrapper functions.  See the documentation for __init__ and
    transform if you need to call this directly.

    This wrapper will not work for subgraph with more than one input.

    Attributes:
    `subgraph` -- (node, read-only) the node upon which this one is
                  based.
    `mem_init` -- (array_like, read-only) Initial value for the memory.
    """
    def __init__(self, input, subgraph, subgraph_input, mem_init=None, 
                 outshp=None, dtype=theano.config.floatX, name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> x_n = InputNode(x)
        >>> r = RecurrentWrapper(x, SimpleNode(x_n, 10, 5, dtype='float32'),
        ...                      x_n, outshp=(5,), dtype='float32')
        >>> r.memory.value
        array([ 0.,  0.,  0.,  0.,  0.], dtype=float32)
        """
        BaseNode.__init__(self, [input], name)
        if mem_init is None:
            mem_init = numpy.zeros(outshp, dtype=dtype)
        self.mem_init = mem_init
        self.memory = theano.shared(self.mem_init.copy(), name='memory')
        self.subgraph = subgraph
        self.subgraph_input = subgraph_input
        self.local_params = self.subgraph.params

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
        >>> r = RecurrentWrapper(x, SimpleNode(x_n, 5, 2, dtype='float32'),
        ...                      x_n, outshp=(2,), dtype='float32')
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
        >>> x_n2 = InputNode(x)
        >>> r=RecurrentWrapper(x, RecurrentWrapper(x_n, SimpleNode(x_n2, 6,2), x_n2, outshp=(2,)), x_n, outshp=(2,))
        >>> f = theano.function([x], r.output)
        >>> v = f(numpy.random.random((3, 2)))
        >>> v.shape
        (3, 2)
        >>> (r.memory.value == v[-1]).all()
        True
        >>> (r.subgraph.memory.value == v[-1]).all()
        True
        """
        def f(inp, mem):
            i = InputNode(T.unbroadcast(T.shape_padleft(T.join(0,inp,mem)),0))
            g = self.subgraph.replace({self.subgraph_input: i})
            return g.output[0]
        
        outs,upds = theano.scan(f,sequences=[input],outputs_info=[self.memory])
        
        for s, u in upds.iteritems():
            s.default_update = u
        self.memory.default_update = outs[-1]
        return outs

from .base import *
from .simple import SimpleNode
from pynnet.nlins import *
from pynnet.errors import cross_entropy

from theano.tensor.shared_randomstreams import RandomStreams

__all__ = ['DelayNode', 'RecurrentInput', 'RecurrentOutput', 'RecurrentWrapper']

class DelayNode(BaseNode):
    r"""
    This is a node that inserts a delay of its input in the graph.
    
    Examples:
    >>> x = T.fmatrix()
    >>> d = DelayNode(x, 3, numpy.array([[1, 2], [2, 2], [2, 1]], dtype='float32'))
    """
    def __init__(self, input, delay, init_vals, name=None):
        r"""
        >>> x = T.matrix()
        >>> d = DelayNode(x, 1, numpy.array([[1, 2, 3]], dtype='float32'))
        >>> d.delay
        1
        >>> d.memory.get_value()
        array([[ 1.,  2.,  3.]], dtype=float32)
        """
        BaseNode.__init__(self, [input], name)
        self.delay = delay
        self.memory = theano.shared(init_vals, 'delaymem')

    def transform(self, input):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> d = DelayNode(x, 2, numpy.random.random((2, 2)).astype('float32'))
        >>> d.params
        []
        >>> theano.pp(d.output)
        'join(0, delaymem, x)[:-2]'
        >>> f = theano.function([x], d.output, allow_input_downcast=True)
        >>> inp = numpy.random.random((5, 2)).astype('float32')
        >>> v = f(inp)
        >>> v.dtype
        dtype('float32')
        >>> v.shape
        (5, 2)
        >>> (d.memory.get_value() == inp[-2:]).all()
        True
        >>> (v[2:] == inp[:-2]).all()
        True
        """
        j = T.join(0, self.memory, input)
        self.memory.default_update = j[-self.delay:]
        return j[:-self.delay]

class RecurrentInput(BaseNode):
    r"""
    Node used to mark the point where recurrent input is inserted.

    For use in conjunction with RecurrentOutput.  The tag parameter
    serves to match a RecurrentOutput with the corresponding
    RecurrentInput.  More than one recurrent loop can be nested as
    long as the nesting is proper and they do not share the same tag.
    
    Examples:
    >>> x = T.fmatrix()
    >>> tag = object()
    >>> rx = RecurrentInput(x, tag)
    >>> o = SimpleNode(rx, 5, 2)
    >>> ro = RecurrentOutput(o, tag, outshp=(2,))

    You can then use `ro` as usual for the rest of the graph.
    
    Attributes:
    `tag` -- (object, read-write) some object to match this
             RecurrentInput with its corresponding RecurrentOutput
    """
    def __init__(self, input, tag, name=None):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> tag = object()
        >>> rx = RecurrentInput(x, tag)
        >>> o = SimpleNode(rx, 5, 2)
        >>> ro = RecurrentOutput(o, tag, outshp=(2,))
        >>> theano.pp(ro.output)
        'scan(?_steps, x, memory, W, b)'
        """
        BaseNode.__init__(self, [input], name)
        self.tag = tag

class RecurrentOutput(BaseNode):
    r"""
    See documentation for RecurrentInput.
    """
    def __init__(self, input, tag, outshp=None, mem_init=None, name=None,
                 dtype=theano.config.floatX):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> tag = object()
        >>> rx = RecurrentInput(x, tag)
        >>> o = SimpleNode(rx, 5, 2)
        >>> ro = RecurrentOutput(o, tag, outshp=(2,))
        >>> theano.pp(ro.output)
        'scan(?_steps, x, memory, W, b)'
        >>> ro.memory.get_value()
        array([ 0.,  0.])
        """
        BaseNode.__init__(self, [input], name)
        self.tag = tag
        self.mem_init = mem_init or numpy.zeros(outshp, dtype=dtype)
        self.memory = theano.shared(self.mem_init.copy(), name='memory')
        self._inp = cell(None)

    def clear(self):
        r"""
        Resets the memory to the initial value.
        """
        self.memory.set_value(self.mem_init.copy())

    def _walker(self, node):
        r"""
        :nodoc:
        """
        if node.tag == self.tag:
            if self._inp is not node:
                assert self._inp.val is None
                self._inp.val = node
    
    class output(prop):
        def fget(self):
            if 'output' not in self._cache:
                self.walk(self._walker, RecurrentInput)
                assert self._inp.val is not None
                def f(inp, mem):
                    i = InputNode(T.unbroadcast(T.shape_padleft(T.join(0,inp,mem)),0),
                                  allow_complex=True)
                    g = self.inputs[0].replace({self._inp.val: i})
                    return g.output[0]

                outs, updt = theano.scan(f, sequences=[self._inp.val.inputs[0].output], outputs_info=[self.memory])
                
                for s, u in updt.iteritems():
                    s.default_update = u
                self.memory.default_update = outs[-1]
                # clear for the next run
                self._inp.val = None
                self._cache['output'] = outs
            return self._cache['output']

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
        >>> r = RecurrentWrapper(x, lambda x_n: SimpleNode(x_n, 10, 5, dtype='float32'),
        ...                      outshp=(5,), dtype='float32')
        >>> r.memory.get_value()
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
        Resets the memory to the initial value.
        """
        self.memory.set_value(self.mem_init.copy())
    
    def transform(self, input):
        r"""
        Tests:
        >>> x = T.fmatrix('x')
        >>> r = RecurrentWrapper(x, lambda inp: SimpleNode(inp, 5, 2, dtype='float32'),
        ...                      outshp=(2,), dtype='float32')
        >>> r.params
        [W, b]
        >>> theano.pp(r.output)
        'scan(?_steps, x, memory, W, b)'
        >>> f = theano.function([x], r.output, allow_input_downcast=True)
        >>> v = f(numpy.random.random((4, 3)))
        >>> v.dtype
        dtype('float32')
        >>> v.shape
        (4, 2)
        >>> (r.memory.get_value() == v[-1]).all()
        True
        >>> r=RecurrentWrapper(x, lambda inp: RecurrentWrapper(inp, lambda inp2: SimpleNode(inp2, 6,2), outshp=(2,)), outshp=(2,))
        >>> f = theano.function([x], r.output, allow_input_downcast=True)
        >>> v = f(numpy.random.random((3, 2)))
        >>> v.shape
        (3, 2)
        >>> (r.memory.get_value() == v[-1]).all()
        True
        """
        def f(inp, mem):
            i = InputNode(T.unbroadcast(T.shape_padleft(T.join(0,inp,mem)),0),
                          allow_complex=True)
            g = self.subgraph_builder(i)
            return g.output[0]
        
        outs,upds = theano.scan(f,sequences=[input],outputs_info=[self.memory])
        
        for s, u in upds.iteritems():
            s.default_update = u
        self.memory.default_update = outs[-1]
        return outs

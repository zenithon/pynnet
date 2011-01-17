from pynnet.base import *
import pynnet.base

import warnings

import copy

__all__ = ['BaseNode', 'InputNode']+pynnet.base.__all__

cdict = dict()

class BaseNode(BaseObject):
    r"""
    Convenient base class for nodes that sets the required `name`
    attribute in the constructor.

    If you pass it None for `name` in the constructor then a suitable
    unique name based on the class name of the object will be
    generated.  Note that no verification is made that the provided
    names are unique but, it is assumed so in other parts of the code.

    Attributes:
    `name` -- (string, read-only) the name of the node (unique)
    `inputs` -- (tuple, read-write) the list of inputs to the node
    """
    def __init__(self, inputs, name):
        r"""
        Tests:
        >>> b = BaseNode([], None)
        >>> b.name
        'BaseNode1'
        >>> b = BaseNode([], 'll')
        >>> b.name
        'll'
        >>> b = BaseNode([], None)
        >>> b.name
        'BaseNode2'
        >>> b2 = test_saveload(b)
        >>> b2.name
        'BaseNode2'
        >>> c = BaseNode([BaseNode([BaseNode([], 'b')], 'a'),
        ...                BaseNode([], 'zombie')], 'g')
        >>> sorted(c._dict.iterkeys())
        ['a', 'b', 'zombie']
        >>> all([k == v.name for k,v in c._dict.iteritems()])
        True
        """
        self._cache = dict()
        if name is None:
            cname = type(self).__name__
            count = cdict.setdefault(type(self), 1)
            name = '%s%d'%(cname, count)
            cdict[type(self)] += 1
        self.name = name
        self.inputs = tuple(InputNode(input) if not isinstance(input, BaseNode)
                            else input for input in inputs)
        self.local_params = []

    def __setattr__(self, name, val):
        r"""
        Clear the cache on attribute setting.
        """
        BaseObject.__setattr__(self, name, val)
        self._cache.clear()

    def get_node(self, name):
        r"""
        Returns the node corresponding to `name`.
        
        Raises KeyError if there is no corresponding node.

        Tests:
        >>> c = BaseNode([BaseNode([], 'a'), BaseNode([], 'b')], None)
        >>> c.get_node('b')
        b
        """
        return self._dict[name]

    def replace(self, replace_map):
        r"""
        Replace all instances of the specified node in the graph with
        the new node provided.
        
        Tests:
        >>> n = BaseNode([], 'n')
        >>> n2 = BaseNode([n], 'n2')
        >>> n2_new = n2.replace({n: BaseNode([], 'n3')})
        >>> n2_new.inputs[0].name
        'n3'
        >>> n2_new.get_node('n3')
        n3
        >>> n2.inputs[0].name
        'n'
        """
        if self in replace_map:
            return replace_map[self]
        else:
            res = copy.copy(self)
            res.inputs = tuple(i.replace(replace_map) for i in res.inputs)
            return res

    class _dict(prop):
        def fget(self):
            if 'dict' not in self._cache:
                d = dict()
                d.update((l.name, l) for l in self.inputs)
                for l in self.inputs:
                    d.update(l._dict)
                self._cache['dict'] = d
            return self._cache['dict']

    class output(prop):
        def fget(self):
            if 'output' not in self._cache:
                self._cache['output'] = self.transform(*[i.output for i in self.inputs])
            return self._cache['output']

    def _as_TensorVariable(self):
        r"""
        Theano function to be able to use nodes directly in a graph.
        """
        return self.output

    class params(prop):
        def fget(self):
            if 'params' not in self._cache:
                s = set(self.local_params)
                for i in self.inputs:
                    s.update(i.params)
                self._cache['params'] = s
            return sorted(self._cache['params'], key=repr)

    class inputs(prop):
        def fget(self):
            return self._inputs

        def fset(self, val):
            self._inputs = tuple(val)

    def transform(self, *input_vars):
        r"""
        Raises NotImplementedError.

        Tests:
        >>> b = BaseNode([], None)
        >>> b.output
        Traceback (most recent call last):
          ...
        NotImplementedError
        """
        raise NotImplementedError()

    def __str__(self):
        return self.name

    __repr__ = __str__ # for now

class InputNode(BaseNode):
    r"""
    Node to hold a symbolic input to the graph.

    Theano expressions built using nodes that depend on this one will
    have to provide a value for the expression in some way.

    Examples:
    >>> x = T.fmatrix('x')
    >>> x_l = InputNode(x, 'x')
    >>> y_l = InputNode(T.ivector())
    """
    def __init__(self, expr, allow_complex=False, name=None):
        r"""
        >>> x = InputNode(T.fmatrix())
        >>> x.name
        'InputNode...'
        >>> y = InputNode(T.fmatrix())
        >>> x.name != y.name
        True
        """
        if not allow_complex and expr.owner is not None:
            warnings.warn("Passing a theano expression to InputNode might be a bug\nUse allow_complex=True to silence this warning.", RuntimeWarning)
        BaseNode.__init__(self, [], name)
        self.expr = expr

    def transform(self):
        r"""
        Tests:
        
        >>> x = InputNode(T.fmatrix('x'), 'x')
        >>> theano.pp(x.output)
        'x'
        """
        return self.expr

    def __hash__(self):
        r"""
        :nodoc:
        """
        return hash(self.expr)

    def __eq__(self, other):
        r"""
        :nodoc:
        """
        return type(self) == type(other) and self.expr == other.expr

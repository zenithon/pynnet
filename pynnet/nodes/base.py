from pynnet.base import *
import pynnet.base

import copy

__all__ = ['BaseNode', 'InputNode', 'prop']+pynnet.base.__all__

class prop_meta(type):
    r""" 
    Metaclass to allow easy property specifications.  See doc for
    `prop`.
    """
    def __new__(meta, class_name, bases, new_attrs):
        r"""
        :nodoc:

        Tested by `prop`.
        """
        if bases == (object,):
            # The prop class itself
            return type.__new__(meta, class_name, bases, new_attrs)
        fget = new_attrs.get('fget')
        fset = new_attrs.get('fset')
        fdel = new_attrs.get('fdel')
        fdoc = new_attrs.get('__doc__')
        return property(fget, fset, fdel, fdoc)

class prop(object):
    r"""
    Allows easy specification of properties without cluttering the
    class namespace.
    
    The set method is specified with 'fset', the get with 'fget' and
    the del with 'fdel'.  All are optional.  Documentation for the
    attribute is taken from the class documentation, if specified.
    Any other attributes of the class are ignored and will not be
    preserved.

    The 'self' argument these methods take refer to the enclosing
    class of the attribute, not the attribute 'class'.
    
    Example/test:
    >>> class Angle(object):
    ...     def __init__(self, rad):
    ...         self._rad = rad
    ...     
    ...     class rad(prop):
    ...         r'The angle in radians.'
    ...         def fget(self): # here self is an 'Angle' object, not 'rad'
    ...             return self._rad
    ...         def fset(self, val): # same here
    ...             if isinstance(val, Angle):
    ...                 val = val.rad
    ...             self._rad = val
    >>> Angle.rad.__doc__
    'The angle in radians.'
    >>> a = Angle(0.0)
    >>> a.rad
    0.0
    >>> a.rad = 1.5
    >>> a.rad
    1.5
    """
    __metaclass__ = prop_meta

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
    `inputs` -- (list, read-only) the list of inputs to the node
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
        if name is None:
            cname = type(self).__name__
            count = cdict.setdefault(type(self), 1)
            name = '%s%d'%(cname, count)
            cdict[type(self)] += 1
        self.name = name
        self.inputs = [InputNode(input) if not isinstance(input, BaseNode)
                       else input for input in inputs]
        self._dict = dict()
        self._dict.update((l.name, l) for l in self.inputs)
        for l in self.inputs:
            self._dict.update(l._dict)
        self.local_params = []

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
        if self in replace_map:
            return replace_map[self]
        else:
            res = copy.copy(self)
            res.inputs = [i.replace(replace_map) for i in res.inputs]
            return res
    
    class output(prop):
        def fget(self):
            return self.transform(*[input.output for input in self.inputs])

    class params(prop):
        def fget(self):
            return self.local_params + sum((i.params for i in self.inputs), [])

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
    def __init__(self, expr, name=None):
        BaseNode.__init__(self, [], name)
        self.expr = expr

    def transform(self):
        return self.expr

from pynnet.base import *
import pynnet.base

__all__ = ['BaseLayer', 'CompositeLayer', 'prop']+pynnet.base.__all__

class prop_meta(type):
    r""" 
    Metaclass to allow easy property specifications.  See doc for
    `prop`.
    """
    def __new__(meta, class_name, bases, new_attrs):
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
    Allows easy specification of properties without cluttering the class namespace.

    Example/test:
    >>> class Angle(object):
    ...     def __init__(self, rad):
    ...         self._rad = rad
    ...     
    ...     class rad(prop):
    ...         def fget(self):
    ...             return self._rad
    ...         def fset(self, val):
    ...             if isinstance(val, Angle):
    ...                 val = val.rad
    ...             self._rad = val
    >>> a = Angle(0.0)
    >>> a.rad
    0.0
    >>> a.rad = 1.5
    >>> a.rad
    1.5
    """
    __metaclass__ = prop_meta

cdict = dict()

class BaseLayer(BaseObject):
    r"""
    Convinient base class for layers that sets the required `name`
    attribute in the constructor.

    If you pass it None for `name` in the constructor then a suitable
    unique name based on the class name of the object will be
    generated.  Note that no verification is made that the provided
    names are unique but, it is assumed so in other parts of the code.

    Attributes:
    `name` -- (string, read-only) the name of the layer (unique)
    """
    def __init__(self, name):
        r"""
        Tests:
        >>> b = BaseLayer(None)
        >>> b.name
        'BaseLayer1'
        >>> b = BaseLayer('ll')
        >>> b.name
        'll'
        >>> b = BaseLayer(None)
        >>> b.name
        'BaseLayer2'
        >>> b2 = test_saveload(b)
        >>> b2.name
        'BaseLayer2'
        """
        if name is None:
            cname = type(self).__name__
            count = cdict.setdefault(type(self), 1)
            name = '%s%d'%(cname, count)
            cdict[type(self)] += 1
        self.name = name

    def _save_(self, file):
        file.write('BL1')
        psave(self.name, file)

    def _load_(self, file):
        c = file.read(3)
        if c != 'BL1':
            raise ValueError('wrong cookie for BaseLayer')
        self.name = pload(file)

    def build(self, input, input_shape=None):
        r"""
        Raises NotImplementedError.

        Tests:
        >>> b = BaseLayer(None)
        >>> b.build('x')
        Traceback (most recent call last):
          ...
        NotImplementedError
        """
        raise NotImplementedError

    def __str__(self):
        return self.name

    __repr__ = __str__ # for now

def flatten(l, ltypes=(list, tuple)):
    r"""
    >>> a = []
    >>> for i in xrange(2000):
    ...     a = [a, i]
    >>> flatten(a) == range(2000)
    True
    >>> flatten([[5, [6]], 7, [8]])
    [5, 6, 7, 8]
    >>> flatten([3, []])
    [3]
    """
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return l

class CompositeLayer(BaseLayer):
    r"""
    Base class for a layer that contains sublayers.
    
    Sets up a dictionnary mapping layer names to their instance that
    can be queried by the user using the `get_layer()` method.

    Attributes:
    None
    """
    def __init__(self, name, sublayers):
        r"""
        Tests:
        >>> c = CompositeLayer(None, [BaseLayer(None), BaseLayer('zombie')])
        >>> c.add(BaseLayer(None), [BaseLayer('spam')])
        >>> all([k == v.name for k,v in c._dict.iteritems()])
        True
        """
        BaseLayer.__init__(self, name)
        self._dict = dict()
        self.add(sublayers)

    def add(self, *layers):
        r"""
        Adds one or more layers to the dictionary of this CompositeLayer.

        The arguments can be any combination of layers, lists of
        layers, tuples of layers (recursively).

        This method should only be called by subclasses in their
        __init__ method with layers that are part of their
        functionality and should be exposed to the user.  While it is
        possible to call this at any time and with any argument, you
        may violate some assumptions (like the one that there is no
        cycle in the layer graph).
        
        (Tested by __init__.)
        """
        sublayers = flatten(layers)
        self._dict.update((l.name, l) for l in sublayers)
        for l in sublayers:
            if isinstance(l, CompositeLayer):
                self._dict.update(l._dict)
    
    def get_layer(self, name):
        r"""
        Returns the layer corresponding to `name`.
        
        Raises KeyError if there is no corresponding layer.

        Tests:
        >>> c = CompositeLayer(None, [BaseLayer('a'), BaseLayer('b')])
        >>> c.get_layer('b')
        b
        """
        return self._dict[name]

    def _save_(self, file):
        file.write('CompL1')
        psave(self._dict, file)

    def _load_(self, file):
        c = file.read(6)
        if c != 'CompL1':
            raise ValueError('wrong cookie for CompositeLayer')
        self._dict = pload(file)

from pynnet.base import *
import pynnet.base

__all__ = ['BaseLayer', 'CompositeLayer']+pynnet.base.__all__

def forward_access(obj, propname, doc=None, allow_del=False):
    def __get__(self):
        return getattr(obj, propname)

    def __set__(self, value):
        setattr(obj, propname, value)

    def __del__(self):
        delattr(obj, propname)
    if not allow_del:
        __del__ = None

    return property(__get__, __set__, __del__, doc)

def multi_forward(objs, propname, doc=None, allow_del=False):
    def __get__(self):
        return getattr(objs[0], propname)

    def __set__(self, value):
        for obj in objs:
            setattr(obj, propname, value)

    def __del__(self):
        for obj in objs:
            delattr(obj, propname)
    if not allow_del:
        __del__ = None

    return property(__get__, __set__, __del__, doc)

cdict = dict()

class BaseLayer(BaseObject):
    r"""
    Convinient base class for layers that sets the required `name`
    attribute in the constructor.

    If you pass it None for `name` then a suitable unique name based
    on the class name of the object will be generated.  Note that no
    verification is made that the provided names are unique but, it is
    assumed so in other parts of the code.
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

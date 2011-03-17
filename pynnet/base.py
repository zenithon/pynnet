from __future__ import with_statement

__all__ = ['BaseObject', 'theano', 'T', 'numpy', 'load', 'loadf',
           'test_saveload', 'prop', 'cell']

try:
    import cPickle as pickle
except ImportError:
    import pickle
try:
    import cStringIO as StringIO
except ImportError:
    import StringIO

from contextlib import closing
import tempfile, os, zipfile

import numpy
import theano
import theano.tensor as T

def zipadd(fn, zf, name):
    r"""
    Calls `fn` with a file pointer and adds the content of the file to
    `zf` (which is a zip archive) under name `name`

    :notests:
    """
    try:
        fid, fname = tempfile.mkstemp()
        fp = os.fdopen(fid, 'wb')
        fn(fp)
        fp.close()
        zf.write(fname, arcname=name)
    finally:
        if fp:
            fp.close()
        try:
            os.remove(fname)
        except OSError:
            # if there was an error creating the file, it will not be removable
            pass

class PersSave(object):
    r"""
    Part of the implementation of the numpy pickle hack.  

    Used in zsave below.
    """
    def __init__(self, zf):
        r"""
        :nodoc:
        """
        self.zf = zf
        self.count = 0

    def __call__(self, obj):
        r"""
        :nodoc:
        """
        if isinstance(obj, numpy.ndarray):
            name = 'array-'+str(self.count)
            self.count += 1
            def fn(fp):
                numpy.lib.format.write_array(fp, obj)
            zipadd(fn, self.zf, name)
            return name
        else:
            return None

class PersLoad(object):
    r"""
    Part of the implementation of the numpy pickle hack.
    
    Used in zload below.
    """
    def __init__(self, zf):
        r"""
        :nodoc:
        """
        self.zf = zf
    def __call__(self,id):
        r"""
        :nodoc:
        """
        return numpy.lib.format.read_array(self.zf.open(id))

def zsave(obj, file):
    r"""
    Saves `obj` to `file` (which is a file-like object).

    This uses pickle with a special case for numpy matrices to avoid
    using a lot of memory (and often crashing) while saving big
    matrices.

    :notests:
    """
    with closing(zipfile.ZipFile(file, 'w', zipfile.ZIP_DEFLATED, allowZip64=True)) as zf:
        def fn(fp):
            p = pickle.Pickler(fp, 2)
            p.persistent_id = PersSave(zf)
            p.dump(obj)
        zipadd(fn, zf, 'pkl')

def zload(file):
    r"""
    Loads a save created with `zsave`.

    See `zsave` for why this is done.

    :notests:
    """
    with closing(zipfile.ZipFile(file, 'r')) as zf:
        p = pickle.Unpickler(StringIO.StringIO(zf.open('pkl').read()))
        p.persistent_load = PersLoad(zf)
        return p.load()

def test_saveload(obj):
    r"""
    Saves and loads `obj` and returns the loaded copy.
    
    :notests:
    """
    f = StringIO.StringIO()
    obj.savef(f)
    f2 = StringIO.StringIO(f.getvalue())
    f.close()
    obj2 = loadf(f2)
    f2.close()
    return obj2

class BaseObject(object):
    def save(self, fname):
        r"""
        Save the object to disk.

        The named file will be created if not present and overwritten
        if present.

        Do NOT override this method, implement __getstate__ and
        __setstate__ as per pickle rules.
        """
        with open(fname, 'wb') as f:
            self.savef(f)

    def savef(self, f):
        r"""
        Save the object to the file-like object `f`.

        Do NOT override this method, implement __getstate__ and
        __setstate__ as per pickle rules.
        """
        zsave(self, f)
    
def load(fname):
    r"""
    Load an object from a save file.
    """
    with open(fname, 'rb') as f:
        return loadf(f)

def loadf(f):
    r"""
    Loads an object from the file-like object `f`.    
    """
    return zload(f)

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


class cell(object):
    r""" 
    This class is useful when you want to hold a reference to a value
    rather than the value itself.

    This is mostly used in the case of nodes that want to hold mutable
    implementation details without destroying their cache in the
    process.  Note that you have to be certain that this use is
    actually safe cache-wise before mucking about.
    """
    __slots__ = ('val')
    def __init__(self, val=None):
        self.val = val

from __future__ import with_statement

__all__ = ['BaseObject', 'theano', 'T', 'numpy', 'pickle', 'psave', 'pload',
           'test_saveload', 'load', 'loadf']

try:
    import cPickle as pickle
except ImportError:
    import pickle

def psave(obj, file):
    r"""
    Saves `obj` to `file` (which is a file-like object).

    :notests:
    """
    pickle.dump(obj, file, pickle.HIGHEST_PROTOCOL)

def pload(file):
    r"""
    Loads a save created with `psave`.

    :notests:
    """
    return pickle.load(file)

def test_saveload(obj):
    r"""
    Saves and loads `obj` and returns the loaded copy.
    
    :notests:
    """
    import StringIO
    f = StringIO.StringIO()
    obj.savef(f)
    f2 = StringIO.StringIO(f.getvalue())
    f.close()
    obj2 = loadf(f2)
    f2.close()
    return obj2

import numpy
import theano
import theano.tensor as T

def _pickle_method(method):
    r"""
    Helper function to allow pickling of methods.
    
    Tests:
    >>> _pickle_method(BaseObject._load_)
    (<built-in function getattr>, (<class 'pynnet.base.BaseObject'>, '_load_'))
    >>> BaseObject._test = psave
    >>> _pickle_method(BaseObject._test)
    (<function _unpickle_modname at ...>, ('pynnet.base', 'psave'))
    >>> del BaseObject._test
    """
    func_name = method.im_func.__name__
    obj = method.im_self
    if obj is None:
        obj = method.im_class
    if not hasattr(obj, func_name):
        return _unpickle_modname, (method.im_func.__module__, func_name)
    return getattr, (obj, func_name)

def _unpickle_modname(mod, name):
    r"""
    Helper function for some type of class methods.

    >>> _unpickle_modname('pynnet.base', 'psave')
    <function psave at ...>
    """
    import sys
    __import__(mod)
    mod = sys.modules[mod]
    return getattr(mod, name)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method)

class BaseObject(object):
    def _save_(self, file):
        r"""
        Save the state to a file.

        This receives a file object argument and should dump the
        variables and maybe a tag to the file stream.

        You can use the `pickle.dump()` method to save objects, but
        please use `numpy.save()`(and not `numpy.savez()`) to save
        numpy arrays if you must.  `numpy` and `pickle` do not mix
        well.  Do not write to the file before the position of the
        file pointer when you received the file objet.  Also leave the
        file pointer at the end of the written data when you are
        finished.  The `numpy` and `pickle` methods do this
        automatically.

        It is also generally a good idea to write some kind of tag to
        uniquely identify your class and prevent the loading of bad
        data.  This tag can also be used to identify the format
        version in case you ever decide to change it.

        You only need to care about the variables you define yourself.
        In particular do not call the `_save_()` method of your
        parent(s) class(es).  
        """
        file.write("SOSV1")

    def save(self, fname):
        r"""
        Save the object to disk.

        The named file will be created if not present and overwritten
        if present.

        Do NOT override this method, implement a `_save_()`
        method for your classes.
        """
        with open(fname, 'wb') as f:
            self.savef(f)

    def savef(self, f): 
        r"""
        Save the object to the file-like object `f`.

        Do NOT override this method, implement a `_save_()` method for
        your classes.
        """
        psave(type(self), f)
        for C in reversed(type(self).__mro__):
            if hasattr(C, '_save_'):
                psave(C._load_, f)
                C._save_(self, f)
        psave(None, f)
    
    def _load_(self, file):
        r"""
        Load the state from a file.

        You should load what you saved in the `_save_()` method.  Be
        careful to leave the file pointer at the end of your loaded
        data.  The `numpy` and `pickle` methods do this automatically.
        """
        str = file.read(5)
        if str != "SOSV1":
            raise ValueError('Not a save file or file is corrupted')

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
    cls = pload(f)
    obj = object.__new__(cls)
    loadfunc = pload(f)
    while loadfunc is not None:
        loadfunc(obj, f)
        loadfunc = pload(f)
    return obj


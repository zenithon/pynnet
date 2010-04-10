from __future__ import with_statement

__all__ = ['BaseObject', 'theano', 'numpy', 'pickle', 'psave', 'pload',
           'test_saveload']

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
    obj2 = obj.__class__.loadf(f2)
    f2.close()
    return obj2

import numpy
import theano

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
        if hasattr(self, '_vitual'):
            raise ValueError('Cannot save a virtual object.  Save the parent instead.')
        
        with open(fname, 'wb') as f:
            self.savef(f)

    def savef(self, f): 
        r"""
        Save the object to the file-like object `f`.

        Do NOT override this method, implement a `_save_()` method for
        your classes.
        """
        for C in reversed(type(self).__mro__):
            if hasattr(C, '_save_'):
                C._save_(self, f)
    
    @classmethod
    def load(cls, fname):
        r"""
        Load an object from a save file.

        The resulting object will have the same class as the calling
        class of this function.  If the saved object in the file is
        not of the appropriate class exceptions may be raised.

        Do NOT override this method, implement a `_load_()`
        method for your classes.

        Do NOT rely on being able to load an objet as a different
        class than the one it was before save() since that possibility
        may go away in the future.
        """
        with open(fname, 'rb') as f:
            return cls.loadf(f)

    @classmethod
    def loadf(cls, f):
        r"""
        Loads an object from the file-like object `f`.
        
        See the documentation for `load()` for a more complete
        description.

        Do NOT override this method, implement a `_load_()` method for
        your classes.
        """
        obj = object.__new__(cls)
        for C in reversed(type(obj).__mro__):
            if hasattr(C, '_load_'):
                C._load_(obj, f)
        return obj
    
    def _load_(self, file):
        r"""
        Load the state from a file.

        You should load what you saved in the `_save_()` method.  Be
        careful to leave the file pointer at the end of your loaded
        data.  The `numpy` and `pickle` methods do this automatically.
        """
        str = file.read(5)
        if str != "SOSV1":
            raise ValueError('Not a save file of file is corrupted')

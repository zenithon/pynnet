from __future__ import with_statement

__all__ = ['BaseObject', 'numpy', 'pickle']

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy

class BaseObject(object):
    def _save_(self, file):
        file.write("SOSV1")

    def save(self, fname):
        if hasattr(self, '_vitual'):
            raise ValueError('Cannot save virtual object.  Save the parent instead.')
        
        with open(fname, 'wb') as f:
            for C in reversed(type(self).__mro__):
                try:
                    C._save_(self, f)
                except AttributeError:
                    pass

    @classmethod
    def load(cls, fname):
        obj = object.__new__(cls)
        with open(fname, 'rb') as f:
            for C in reversed(type(obj).__mro__):
                try:
                    C._load_(obj, f)
                except AttributeError:
                    pass
        return obj

    def _load_(self, file):
        str = file.read(5)
        if str != "SOSV1":
            raise ValueError('Not a save file of file is corrupted')

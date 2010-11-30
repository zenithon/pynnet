from pynnet.base import *

import imp

import copy_reg
copy_reg.constructor(getattr)

class DataWrapper(object):
    __slots__ = ('name', 'dset', 'stream')
    def __init__(self, name, dset, stream):
        self.name = name
        self.dset = dset
        self.stream = stream
    
    def __call__(self, **kwargs):
        self.dset.shared(**kwargs)(self.stream._counter)

    def __reduce__(self):
        return (getattr, (self.stream, self.name))

class DatasetStream(object):
    _cache = dict()
    def __init__(self, path):
        self._path = path
        self._counter = theano.shared(0)
        self._counter.default_update = self._counter + 1

    def __getattribute__(self, name):
        if name[0] == '_':
            return object.__getattribute__(self, name)
        if name not in self._cache:
            try:
                fp, pathname, descr = imp.find_module(name, self._path)
            except ImportError:
                raise AttributeError('No such dataset: ', + name)
            try:
                mod = imp.load_module(name, fp, pathname, descr)
            except ImportError:
                raise ValueError('Could not load dataset ' + name)
            finally:
                if fp:
                    fp.close()
            self._cache[name] = DataWrapper(mod.dset, self._counter)
        return self._cache[name]


    def seek(self, pos):
        self._counter.value = pos

    def tell(self):
        return self._counter.value

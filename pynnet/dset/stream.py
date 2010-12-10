from pynnet.base import *

import imp


def load_dw(stream, name, part, kwargs):
    return stream.get(name, part, **kwargs)

import copy_reg
copy_reg.constructor(load_dw)

class DataWrapper(object):
    __slots__ = ('stream', 'name', 'part', 'kwargs', 'data')
    def __init__(self, stream, name, part, kwargs):
        self.stream = stream
        self.name = name
        self.part = part
        self.kwargs = kwargs
        self.mod = load_dset(self.stream._path, self.name)

    def get_data(self):
        getattr(self.mod, self.part)

    def _as_TensorVariable(self)
        s = self.mod.shared(self, self.kwargs)
        return s(self.stream._counter)

    def __reduce__(self):
        return (load_dw, (self.stream, self.name, self.part, self.kwargs))

_dset_cache = dict()

def load_dset(path, name):
    if name not in _dset_cache:
        try:
            fp, pathname, descr = imp.find_module(name, path)
        except ImportError:
            raise AttributeError('No such dataset: ', + name)
        try:
            mod = imp.load_module(name, fp, pathname, descr)
        except ImportError:
            raise ValueError('Could not load dataset ' + name)
        finally:
            if fp:
                fp.close()
        _dset_cache[name] = mod
    return _dset_cache[name]

class DatasetStream(object):
    def __init__(self, path):
        self._path = path
        self._counter = theano.shared(0)
        self._counter.default_update = self._counter + 1

    def get(name, part, **kwargs):
        return DataWrapper(self, name, part, kwargs)
    
    def seek(self, pos):
        self._counter.value = pos
    
    def tell(self):
        return int(self._counter.value)

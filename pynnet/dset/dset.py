from pynnet.base import *
from pynnet.dset.shared import MemoryDataset

import sys

__all__ = ['dset']

class dset(object):
    r"""
    Skeleton dataset class to wrap a piece of data.

    This wrapper is there to provide a named reference to the data and
    to save only this name while pickling.  On loading the data is
    fetched again by name.  This avoids saving large dataset matrices
    as part of a model during training or otherwise.

    No assumptions are made as to the nature of the data apart from
    those made in the `shared_class` (which is configurable).
    """
    def __init__(self, stream, name, shared_class=MemoryDataset):
        r"""
        Tests:
        >>> d = dset.dset('sys', 'modules') # Not a usual dataset, but will do
        >>> type(d.data)
        <type 'dict'>
        >>> import pickle
        >>> s = pickle.dumps(d, -1)
        >>> 'pynnet.base' in s # make sure the dict was not saved
        False
        >>> d2 = pickle.loads(s)
        >>> type(d2.data)
        <type 'dict'>
        """
        self.module = module
        self.name = name
        self.shared_class = shared_class
        self._set_data()

    def __getstate__(self):
        r"""
        :nodoc:
        """
        odict = self.__dict__.copy()
        del odict['data']
        return odict

    def __setstate__(self, d):
        r"""
        :nodoc:
        """
        self.__dict__.update(d)
        self._set_data()

    def _set_data(self):
        r"""
        :nodoc:
        """
        self.data = getattr(sys.modules[self.module], self.name)

    def shared(**kwargs):
        r"""
        Return a theano op-like object which you can use to supply the input
        in a graph.

        This is based in the `shared_class` parameter passed to
        `__init__` and nothing is assumed of the interface or the
        returned object.

        :notests:
        """
        return self.shared_class(self, **kwargs)

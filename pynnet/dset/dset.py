from pynnet.base import *

import sys

__all__ = ['MemoryDataset', 'ListDataset', 'dset']

class MemoryDataset(object):
    r"""
    Fake op to provide batches of examples to a graph based on an index.

    Tests:
    >>> class fake(object):
    ...    def __init__(self, d):
    ...        self.data = d
    >>> o = fake(numpy.random.random((100, 10)))
    >>> md = MemoryDataset(o, 10)
    >>> i = T.iscalar()
    >>> out = md(i)
    >>> f = theano.function([i], out)
    >>> (f(1) == o.data[10:20]).all()
    True
    """
    def __init__(self, dataset, batch_size):
        r"""
        :nodoc:
        """
        self.data = theano.shared(dataset.data)
        self.batch_size = batch_size

    def __call__(self, idx):
        r"""
        :nodoc:
        """
        return self.data[idx*self.batch_size:(idx+1)*self.batch_size]

class ListDataset(theano.Op):
    r"""
    Theano op to provides examples to a graph from a list.

    This is mainly for datasets that are a list of sequences.

    Tests:
    >>> class fake(object):
    ...    def __init__(self, d):
    ...        self.data = d
    >>> o = fake([numpy.random.random((2, 3)),
    ...           numpy.random.random((2, 3))])
    >>> md = ListDataset(o)
    >>> i = T.iscalar()
    >>> out = md(i)
    >>> f = theano.function([i], out)
    >>> (f(1) == o.data[1]).all()
    True
    """
    def __init__(self, dataset):
        r"""
        :nodoc:
        """
        self.data = dataset.data
        
    def make_node(self, idx):
        r"""
        :nodoc:
        """
        idx_ = T.as_tensor_variable(idx)
        out_ty = T.TensorType(self.data[0].dtype,
                              broadcastable=(False,)*len(self.data[0].shape))
        return theano.Apply(self,
                            inputs = [idx_],
                            outputs = [out_ty()])

    def perform(self, node, inputs, output_storage):
        r"""
        :nodoc:
        """
        idx, = inputs
        output_storage[0][0] = self.data[idx]

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
    def __init__(self, module, name, shared_class=MemoryDataset):
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

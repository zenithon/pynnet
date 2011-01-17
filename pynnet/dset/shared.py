from pynnet.base import *
from pynnet.dset import DataRef
import sys

def shared(value, **kwargs):
    if type(value) is not DataRef:
        def index_shared(idx):
            return value[idx]
        return index_shared
    return value.dset.shared_class(value, **kwargs)

def load_shared(klass, dataref, kwargs):
    return klass(dataref **kwargs)

class SharedBase(theano.Op):
    def __init__(self, dataref, **kwargs):
        self.dataref = dataref
        self.kwargs = kwargs
        self.setup(**kwargs)
    
    def make_node(self, idx):
        idx_ = T.as_tensor_variable(idx)
        return theano.Apply(self,
                            inputs = [idx_],
                            outputs = self.out_types)
    
    def __reduce__(self):
        return load_shared, \
            (self.__class__, self.dataref, self.kwargs)

class MemoryDataset(SharedBase):
    r"""
    Fake op to provide batches of examples to a graph based on an index.

    Tests:
    >>> class fake(object):
    ...    def __init__(self, d):
    ...        self.data = d
    ...    def get_data(self, id):
    ...        return self.data
    >>> o = fake(numpy.random.random((100, 10)))
    >>> md = MemoryDataset(o, None, batch_size=10)
    >>> i = T.iscalar()
    >>> out = md(i)
    >>> f = theano.function([i], out)
    >>> (f(1) == o.data[10:20]).all()
    True
    """
    def setup(self, batch_size):
        r"""
        :nodoc:
        """
        self.batch_size = batch_size
        self.data = theano.shared(data)
        self.out_types = [self.data.type]

    def _as_CudaNdarrayVariable(self):
        r"""
        :nodoc:
        """
        return self

    def perform(self, node, inputs, output_storage):
        r"""
        :nodoc:
        """
        idx, = inputs
        d = self.data.get_value(borrow=True, return_internal_type=True)
        n = int(d.shape[0]/self.batch_size)
        idx %= n
        output_storage[0][0] = d[idx*self.batch_size:(idx+1)*self.batch_size]

class ListDataset(SharedBase):
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
    def setup(self):
        r"""
        :nodoc:
        """
        self.out_types = [T.TensorType(self.data[0].dtype,
                                       broadcastable=(False,)*len(self.data[0].shape))]
    
    def perform(self, node, inputs, output_storage):
        r"""
        :nodoc:
        """
        idx, = inputs
        output_storage[0][0] = self.data[idx%len(self.data)]

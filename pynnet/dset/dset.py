from pynnet.base import *

import sys

__all__ = ['MemoryDataset', 'ListDataset', 'dset']

class MemoryDataset(theano.Op):
    def __init__(self, dataset, batch_size):
        self.data = theano.shared(dataset.data)
        self.batch_size = batch_size

    def make_node(self, idx):
        idx_ = theano.as_tensor_variable(idx)
        return theano.Apply(self,
                            inputs = [idx_],
                            outputs = [self.data.type()])

    def preform(self, node, inputs, output_storage):
        idx, = inputs
        self.output_storage[0][0] = self.data[idx*self.batch_size:(idx+1)*self.batch_size]

# This is for datasets which are lists of sequences
class ListDataset(theano.Op):
    def __init__(self, dataset):
        self.data = theano.Constant(dataset.data)
        
    def make_node(self, idx):
        idx_ = theano.as_tensor_variable(idx)
        return theano.Apply(self,
                            inputs = [idx_],
                            outputs = [T.TensorType(self.data[0].dtype,
                              broadcastable=(False,)*len(self.data[0].shape))])

    def preform(self, node, inputs, output_storage):
        idx, = inputs
        self.output_storage[0][0] = self.data[idx]

class dset(object):
    def __init__(self, module, name, shared_class=MemoryDataset):
        self.module = sys.modules[module]
        self.name = name
        self.shared_class = shared_class
        self._set_data()

    def __getstate__(self):
        odict = self.__dict__.copy()
        del odict['data']
        return odict

    def __setstate__(self, d):
        object.__setstate__(self, d)
        self._set_data()

    def _set_data(self):
        self.data = getattr(self.module, self.name)

    def shared(**kwargs):
        return self.shared_class(self, **kwargs)

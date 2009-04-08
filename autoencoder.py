import numpy

from base import *

from net import *
from simplenet import *
from nlins import *
from errors import *

__all__ = ['Autoencoder', 'StackedAutoencoder']

class Autoencoder(SimpleNet):
    r"""
    Simple autoencoder with only one hidden layer.
    """
    def __init__(self, ninputs, noutputs, nlin=tanh, noisyness=0.0, corrupt_value=0.0, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        SimpleNet.__init__(self, ninputs, noutputs, ninputs, hnlin=nlin, error=mse, onlin=nlin, alpha=alpha, lmbd=lmbd, dtype=dtype)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def _save_(self, file):
        file.write('AE1')
        pickle.dump((self.noisyness, self.corrupt_value), file)

    def _load_(self, file):
        s = file.read(3)
        if s == 'AE1':
            self.noisyness, self.corrupt_value = pickle.load(file)
        else:
            raise ValueError('Not a valid Autoencoder save file')

    def test(self, x):
        x = numpy.atleast_2d(x)
        return self._test(x, x)

    def code(self, x):
        x = numpy.atleast_2d(x)
        return self._fprop(x)[1]

    def epoch_bprop(self, x):
        x = numpy.atleast_2d(x)
        return self._epoch_bprop(x, x)

    def _epoch_bprop(self, x, y):
        if self.noisyness != 0.0:
            x[numpy.random.random(x.shape) < self.noisyness] = self.corrupt_value
        SimpleNet._epoch_bprop(self, x, y)

    def train_loop(self, x, epochs=100):
        x = numpy.atleast_2d(x)
        return self._train_loop(x, x, epochs)

class StackedAutoencoder(NNet):
    def __init__(self, sizes, nlins, noisyness=0.0, corrupt_value=0.0, alpha=0.01, lmbd=0.0, dtype=numpy.float32):
        self.net = NNet(self, sizes + sizes[-2::-1], nlins + nlins[::-1], alpha=alpha, lmbd=lmbd, dtype=dtype)
        self.noisyness = noisyness
        self.corrupt_value = corrupt_value

    def _save_(self, file):
        self.net._save_(file)
        file.write('SAE1')
        pickle.dump((self.noisyness, self.corrupt_value), file)
    
    def _load_(self, file):
        self.net._load_(file)
        s = file.read(4)
        if s == 'SAE1':
            self.noisyness, self.corrupt_value = pickle.load(file)
        else:
            raise ValueError('Not a valid Autoencoder save file')

    def test(self, x):
        x = numpy.atleast_2d(x)
        return self.net._test(x, x)

    def pretrain(self, x):
        x = numpy.atleast_2d(x)
        return self._pretrain(x)

    def _pretrain(self, x):
        ## code be here

    def epoch_bprop(self, x):
        x = numpy.atleast_2d(x)
        return self._epoch_bprop(x, x)

    def _epoch_bprop(self, x, y):
        if self.noisyness != 0.0:
            x[numpy.random.random(x.shape) < self.noisyness] = self.corrupt_value
        self.net._epoch_bprop(self, x, y)

    def train_loop(self, x, epochs=100):
        x = numpy.atleast_2d(x)
        return self.net._train_loop(x, x, epochs)
